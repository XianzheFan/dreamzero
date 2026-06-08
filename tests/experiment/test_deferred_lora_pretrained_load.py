from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("h5py")
pytest.importorskip("albumentations")
from groot.vla.experiment import base


class _DeferredActionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(defer_lora_injection=True)
        self.model = torch.nn.Module()
        self.injected = False

    def inject_lora_after_loading(self):
        wrapper = torch.nn.Module()
        wrapper.model = torch.nn.Linear(1, 1, bias=False)
        self.model.base_model = wrapper
        self.injected = True


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.pre = torch.nn.Linear(1, 1, bias=False)
        self.action_head = _DeferredActionHead()


def test_create_model_reloads_deferred_lora_keys_after_injection(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    save_file(
        {
            "pre.weight": torch.tensor([[3.0]]),
            "action_head.model.base_model.model.weight": torch.tensor([[7.0]]),
        },
        str(ckpt_dir / "model.safetensors"),
    )
    model = _TinyModel()
    monkeypatch.setattr(base, "instantiate", lambda _: model)

    cfg = SimpleNamespace(model=object(), pretrained_model_path=str(ckpt_dir))
    training_args = SimpleNamespace(output_dir=str(tmp_path / "out"))

    loaded = base.BaseExperiment.create_model(object(), cfg, training_args)

    assert loaded.action_head.injected
    torch.testing.assert_close(loaded.pre.weight, torch.tensor([[3.0]]))
    torch.testing.assert_close(
        loaded.action_head.model.base_model.model.weight,
        torch.tensor([[7.0]]),
    )
