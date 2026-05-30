"""WebSocket server for the lightweight RoboFactory BC policy."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import websockets.asyncio.server
import websockets.frames

from eval_utils.robofactory_bc_policy import RoboFactoryBCPolicy


@dataclasses.dataclass
class BCServerConfig:
    num_agents: int = 2
    image_resolution: tuple[int, int] = (240, 320)
    num_frames: int = 1
    action_horizon: int = 24
    action_dim: int = 16
    fps: int = 20


def _make_packer():
    try:
        from openpi_client import msgpack_numpy as _mn  # type: ignore

        packer = _mn.Packer()
        return packer.pack, _mn.unpackb
    except Exception:
        import msgpack
        import msgpack_numpy

        msgpack_numpy.patch()
        return (
            lambda obj: msgpack.packb(obj, use_bin_type=True),
            lambda buf: msgpack.unpackb(buf, raw=False),
        )


class BCWebsocketServer:
    def __init__(self, policy: RoboFactoryBCPolicy, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.cfg = BCServerConfig(
            action_horizon=policy.horizon,
            action_dim=policy.action_dim,
        )
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            logging.info("BC policy server listening on ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket):
        logging.info("Connection from %s opened", websocket.remote_address)
        pack, unpack = _make_packer()
        await websocket.send(pack(dataclasses.asdict(self.cfg)))
        while True:
            try:
                obs = unpack(await websocket.recv())
                endpoint = obs.pop("endpoint", "infer")
                if endpoint == "reset":
                    reply: Any = self.policy.reset(obs)
                else:
                    reply = self.policy.infer(obs)
                await websocket.send(pack(reply))
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                logging.error("Inference error:\n%s", tb)
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--progress-steps",
        type=int,
        default=None,
        help="Step count that maps rollout progress to 1.0. Defaults to checkpoint config.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    policy = RoboFactoryBCPolicy(args.ckpt, device=args.device, progress_steps=args.progress_steps)
    BCWebsocketServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
