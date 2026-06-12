from dataclasses import dataclass, field
import logging
import time
from typing import TypeAlias, cast
import os

from accelerate import load_checkpoint_and_dispatch

from einops import rearrange
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from safetensors.torch import load_file
import json
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)

WAN_HF_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN22_HF_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"


def hf_download(filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Download a file from the specified HuggingFace repo to HF cache."""
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return path


def ensure_file(path: str | None, hf_filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Return a valid local path: use `path` if it exists, otherwise download from HuggingFace."""
    if path is not None and os.path.exists(path):
        return path
    return hf_download(hf_filename, repo_id)

from torch.distributions import Beta
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchvision.transforms import v2
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.n1_5.action_head.base_action_head import ActionHead
from groot.vla.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
from groot.vla.model.dreamzero.modules.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from groot.vla.model.dreamzero.modules.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler


KVCacheType: TypeAlias = torch.Tensor

@dataclass
class WANPolicyHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )
    tiled: bool = field(default=True, metadata={"help": "Whether to use tiled input."})
    tile_size_height: int = field(default=34, metadata={"help": "Tile size height."})
    tile_size_width: int = field(default=34, metadata={"help": "Tile size width."})
    tile_stride_height: int = field(default=18, metadata={"help": "Tile stride height."})
    tile_stride_width: int = field(default=16, metadata={"help": "Tile stride width."})
    num_frame_per_block: int = field(default=1, metadata={"help": "Number of frames per block."})
    # Target video (H, W) for Wan22 resize. When set, videos are resized to this before VAE so latent
    # spatial size matches. Use height/width divisible by 32 for WanVideoVAE38 (16x) so latent H,W are even.
    target_video_height: int | None = field(default=None, metadata={"help": "Target video height for resize (e.g. 160 for even latent with VAE38)."})
    target_video_width: int | None = field(default=None, metadata={"help": "Target video width for resize (e.g. 320)."})

    lora_rank: int = field(default=4, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=4, metadata={"help": "LoRA alpha."})
    lora_target_modules: str = field(default="q,k,v,o,ffn.0,ffn.2")
    init_lora_weights: str = field(default="kaiming", metadata={"help": "LoRA initialization method."})
    train_architecture: str= field(default="lora", metadata={"help": "Train architecture."})
    skip_component_loading: bool = field(default=False, metadata={"help": "Skip loading individual component weights (used when loading from full pretrained model)."})

    use_gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing."})
    qformer_cfg: dict = field(default=None, metadata={"help": "Qformer configuration."})
    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    # High noise emphasis for BASE (coupled) training - applies Beta distribution to BOTH video and action together
    use_high_noise_emphasis: bool = field(
        default=False, metadata={"help": "Use Beta distribution for noise sampling (biases BOTH video and action towards high noise levels together)."}
    )
    high_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for high noise emphasis. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher = more high noise bias."}
    )
    # Decoupled noise sampling config for training-inference alignment
    # When enabled: video uses Beta(alpha,beta) biased towards high noise, action uses independent uniform
    decouple_video_action_noise: bool = field(
        default=False, metadata={"help": "Decouple video/action noise: video uses Beta distribution (high noise bias), action uses independent uniform."}
    )
    video_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for video noise. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher alpha = more bias to high noise."}
    )
    video_noise_beta_beta: float = field(
        default=1.0, metadata={"help": "Beta beta for video noise. Keep at 1.0."}
    )
    # Decoupled inference config - allows video to stay noisy while action fully denoises
    decouple_inference_noise: bool = field(
        default=False, metadata={"help": "Use decoupled noise schedules during inference (video stays noisy, action fully denoises)."}
    )
    video_inference_final_noise: float = field(
        default=0.8, metadata={"help": "Final noise level for video during decoupled inference (0.0-1.0). E.g., 0.8 means video ends at 80% noise."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    action_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Global multiplier for action diffusion loss."},
    )
    gripper_action_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Extra multiplier for gripper action dimensions."},
    )
    gripper_close_action_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": "Additional multiplier for gripper targets below the close threshold."
        },
    )
    gripper_close_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Normalized gripper target threshold below which the command is treated as close."
        },
    )
    gripper_clean_action_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": "Optional direct MSE on predicted clean gripper actions reconstructed from the flow output."
        },
    )
    gripper_clean_close_action_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": "Additional multiplier for close targets in the clean gripper action loss."
        },
    )
    gripper_clean_max_sigma: float = field(
        default=1.0,
        metadata={
            "help": "Only apply clean gripper action loss at action diffusion sigmas <= this value."
        },
    )
    gripper_binary_action_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": "Optional BCE-with-logits loss on clean gripper open/close targets."
        },
    )
    gripper_binary_close_action_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Additional multiplier for close targets in binary gripper loss."},
    )
    gripper_binary_logit_scale: float = field(
        default=4.0,
        metadata={
            "help": "Scale applied to normalized clean gripper predictions before BCE."
        },
    )
    gripper_binary_max_sigma: float = field(
        default=1.0,
        metadata={
            "help": "Only apply binary gripper loss at action diffusion sigmas <= this value."
        },
    )
    gripper_action_dims: tuple[int, ...] = field(
        default=(7,),
        metadata={"help": "Per-agent action dimensions treated as grippers."},
    )
    action_prefix_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Extra multiplier for early action-horizon steps."},
    )
    action_prefix_loss_len: int = field(
        default=0,
        metadata={"help": "Number of early action steps to upweight."},
    )
    first_close_joint_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": "Extra multiplier for non-gripper joint action loss near each agent's first close target."
        },
    )
    first_close_joint_loss_window_before: int = field(
        default=0,
        metadata={"help": "Number of steps before first close to upweight for joint action loss."},
    )
    first_close_joint_loss_window_after: int = field(
        default=0,
        metadata={"help": "Number of steps after first close to upweight for joint action loss."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)
    defer_lora_injection: bool = field(default=False, metadata={"help": "Defer LoRA injection until after loading pretrained weights."})

    vl_self_attention_cfg: dict = field(default=None)
    text_encoder_cfg: dict = field(default=None)
    image_encoder_cfg: dict = field(default=None)
    vae_cfg: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class WANPolicyHead(ActionHead):
    config_class = WANPolicyHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: WANPolicyHeadConfig,
    ):
        super().__init__()
        self.tiled = config.tiled
        self.tile_size_height = config.tile_size_height
        self.tile_size_width = config.tile_size_width
        self.tile_stride_height = config.tile_stride_height
        self.tile_stride_width = config.tile_stride_width
        self.num_frame_per_block = config.num_frame_per_block
        self.hidden_size = config.hidden_size
        self.num_frames = config.num_frames
        self.text_encoder = instantiate(config.text_encoder_cfg)
        self.image_encoder = instantiate(config.image_encoder_cfg)
        self.vae = instantiate(config.vae_cfg)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.model_names = ['text_encoder']

        self.num_inference_steps = 16 
        self.seed = 1140
        self.cfg_scale = 5.0
        self.denoising_strength = 1.0
        self.sigma_shift = 5.0
        self.kv_cache1: KVCacheType | None = None
        self.kv_cache_neg: KVCacheType | None = None
        self.crossattn_cache: KVCacheType | None = None
        self.crossattn_cache_neg: KVCacheType | None = None

        self.global_step = 0
        self.max_steps = 0
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.init_lora_weights = config.init_lora_weights
        self.train_architecture = config.train_architecture
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self.language = None
        self._ma_cached_token_agent_id = None
        self._ma_cached_token_agent_id_neg = None

        self.ip_rank = 0
        self.ip_size = 1
        self.ip_group = None
        
        self._device = "cuda"
        self.dynamic_cache_schedule = os.getenv("DYNAMIC_CACHE_SCHEDULE", "False").lower() == "true"


        num_dit_steps = 8
        if os.getenv("NUM_DIT_STEPS") is not None:
            num_dit_steps = int(os.getenv("NUM_DIT_STEPS"))
        if num_dit_steps == 5:
            self.dit_step_mask = [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False]
        elif num_dit_steps == 6:
            self.dit_step_mask = [True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 7:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 8:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True]
        else:
            self.dit_step_mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        assert self.dit_step_mask[0] == True, "first step must be True"

        self.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        if self.training:
            self.scheduler.set_timesteps(1000, training=True)
        
        
        self.input_embedding_dim = config.input_embedding_dim

        self.cpu_offload = False

        self.model = instantiate(config.diffusion_model_cfg)
        if hasattr(self.model, "_set_gradient_checkpointing"):
            self.model._set_gradient_checkpointing(
                self.model, self.use_gradient_checkpointing
            )
        elif hasattr(self.model, "gradient_checkpointing"):
            self.model.gradient_checkpointing = self.use_gradient_checkpointing
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        
        text_enc_path = ensure_file(
            self.text_encoder.text_encoder_pretrained_path,
            "models_t5_umt5-xxl-enc-bf16.pth",
        )
        self.text_encoder.load_state_dict(torch.load(text_enc_path, map_location='cpu'))

        img_enc_path = ensure_file(
            self.image_encoder.image_encoder_pretrained_path,
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        )
        self.image_encoder.model.load_state_dict(torch.load(img_enc_path, map_location='cpu'), strict=False)

        # Wan2.2 (WanVideoVAE38, z_dim=48) uses Wan2.2_VAE.pth; Wan2.1 uses Wan2.1_VAE.pth
        vae_hf_filename = "Wan2.2_VAE.pth" if getattr(self.vae, "z_dim", 16) == 48 else "Wan2.1_VAE.pth"
        vae_repo_id = WAN22_HF_REPO_ID if getattr(self.vae, "z_dim", 16) == 48 else WAN_HF_REPO_ID
        vae_path = ensure_file(
            self.vae.vae_pretrained_path,
            vae_hf_filename,
            repo_id=vae_repo_id,
        )
        self.vae.model.load_state_dict(torch.load(vae_path, map_location='cpu'))

        if not config.skip_component_loading:
            dit_dir = self.model.diffusion_model_pretrained_path
            # Wan2.2 (in_dim=48) uses Wan2.2-TI2V-5B repo; Wan2.1 uses Wan2.1-I2V-14B-480P
            dit_repo_id = WAN22_HF_REPO_ID if getattr(self.model, "in_dim", 16) == 48 else WAN_HF_REPO_ID
            if dit_dir is None or not os.path.isdir(dit_dir):
                index_path = hf_hub_download(repo_id=dit_repo_id, filename="diffusion_pytorch_model.safetensors.index.json")
                dit_dir = os.path.dirname(index_path)
                with open(index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in set(index["weight_map"].values()):
                    hf_hub_download(repo_id=dit_repo_id, filename=shard_file)

            if dit_dir is not None:
                safetensors_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors")
                safetensors_index_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors.index.json")
                state_dict = {}

                if os.path.exists(safetensors_index_path):
                    # Handle sharded safetensors
                    print(f"Loading sharded safetensors using index: {safetensors_index_path}")

                    with open(safetensors_index_path, 'r') as f:
                        index = json.load(f)

                    # Load each shard
                    for shard_file in set(index["weight_map"].values()):
                        shard_path = os.path.join(dit_dir, shard_file)
                        print(f"Loading shard: {shard_path}")
                        shard_state_dict = load_file(shard_path)
                        state_dict.update(shard_state_dict)

                elif os.path.exists(safetensors_path):
                    # Handle single safetensors file
                    print(f"Loading weights from safetensors: {safetensors_path}")
                    state_dict = load_file(safetensors_path)

                else:
                    raise ValueError(f"No safetensors file found at {safetensors_path} or {safetensors_index_path}")

                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    print(f"Missing keys when loading pretrained weights: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")

                print("Successfully loaded pretrained weights")
        else:
            print("Skipping individual component loading (loading from full pretrained model)")
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        # Video noise Beta distribution (biased towards high noise levels when enabled)
        self.video_beta_dist = Beta(config.video_noise_beta_alpha, config.video_noise_beta_beta)
        # High noise emphasis Beta distribution for coupled training (applies to both video and action)
        self.high_noise_beta_dist = Beta(config.high_noise_beta_alpha, 1.0)
        # self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self._noise_logged = False
        self.defer_lora_injection = config.defer_lora_injection
        print("defer_lora_injection@@", self.defer_lora_injection)
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        if self.train_architecture == "lora" and not self.defer_lora_injection:
            print("Adding LoRA to model")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
            self._enable_multi_agent_aux_trainable()
        elif self.train_architecture == "lora" and self.defer_lora_injection:
            print("Deferring LoRA injection until after pretrained weights are loaded")
        else:
            self.print_trainable_params()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if not self.defer_lora_injection:
            self.print_trainable_params()


    def print_trainable_params(self):
        """Print trainable parameters of the diffusion model."""
        trainable_params = []
        total_params = 0
        trainable_total = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append(name)
                trainable_total += param.numel()
                
        print(f"Total parameters in diffusion model: {total_params:,}")
        print(f"Trainable parameters in diffusion model: {trainable_total:,}")
        # print(trainable_params)


    def inject_lora_after_loading(self):
        """
        Inject LoRA adapters after pretrained weights have been loaded.
        This should be called when defer_lora_injection=True.
        """
        if self.train_architecture == "lora":
            print("Injecting LoRA after loading pretrained weights")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
            self._enable_multi_agent_aux_trainable()
            
            self.text_encoder.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.print_trainable_params()
        else:
            print("LoRA injection not needed (train_architecture != 'lora')")

    def _enable_multi_agent_aux_trainable(self) -> None:
        """Train and save multi-agent parameters that are not part of DROID."""
        enabled: list[str] = []
        hub_tokens = getattr(self.model, "hub_tokens", None)
        if hub_tokens is not None:
            hub_tokens.requires_grad_(True)
            enabled.append("hub_tokens")

        role_embedding = getattr(self.model, "role_embedding", None)
        if role_embedding is not None:
            role_embedding.requires_grad_(True)
            enabled.append("role_embedding")

        if enabled:
            print(
                "Trainable multi-agent auxiliary parameters: "
                + ", ".join(enabled)
            )

    def _apply_action_loss_weights(
        self,
        action_loss: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply optional action/gripper/prefix weights before reduction."""
        action_weight = self._config_float("action_loss_weight", 1.0)
        gripper_weight = self._config_float("gripper_action_loss_weight", 1.0)
        gripper_close_weight = self._config_float(
            "gripper_close_action_loss_weight",
            1.0,
        )
        gripper_close_threshold = self._config_float("gripper_close_threshold", 0.0)
        prefix_weight = self._config_float("action_prefix_loss_weight", 1.0)
        prefix_len = self._config_int("action_prefix_loss_len", 0)
        first_close_joint_weight = self._config_float(
            "first_close_joint_loss_weight",
            1.0,
        )
        first_close_window_before = self._config_int(
            "first_close_joint_loss_window_before",
            0,
        )
        first_close_window_after = self._config_int(
            "first_close_joint_loss_window_after",
            0,
        )
        gripper_dims = [int(dim) for dim in getattr(self.config, "gripper_action_dims", [7])]

        weighted = action_loss
        if gripper_weight != 1.0 and weighted.shape[-1] > 0:
            dim_weights = torch.ones(
                weighted.shape[-1], device=weighted.device, dtype=weighted.dtype
            )
            for dim in gripper_dims:
                if -weighted.shape[-1] <= dim < weighted.shape[-1]:
                    dim_weights[dim % weighted.shape[-1]] = gripper_weight
            weighted = weighted * dim_weights.view(*([1] * (weighted.ndim - 1)), -1)

        if (
            gripper_close_weight != 1.0
            and actions is not None
            and weighted.shape == actions.shape
            and weighted.shape[-1] > 0
        ):
            close_weights = torch.ones_like(weighted)
            for dim in gripper_dims:
                if -weighted.shape[-1] <= dim < weighted.shape[-1]:
                    dim_idx = dim % weighted.shape[-1]
                    close_mask = actions[..., dim_idx] < gripper_close_threshold
                    close_weights[..., dim_idx] = torch.where(
                        close_mask,
                        torch.as_tensor(
                            gripper_close_weight,
                            device=weighted.device,
                            dtype=weighted.dtype,
                        ),
                        close_weights[..., dim_idx],
                    )
            weighted = weighted * close_weights

        time_dim = weighted.ndim - 2
        if prefix_weight != 1.0 and prefix_len > 0 and time_dim >= 0:
            steps = min(prefix_len, weighted.shape[time_dim])
            if steps > 0:
                time_weights = torch.ones(
                    weighted.shape[time_dim],
                    device=weighted.device,
                    dtype=weighted.dtype,
                )
                time_weights[:steps] = prefix_weight
                view_shape = [1] * weighted.ndim
                view_shape[time_dim] = weighted.shape[time_dim]
                weighted = weighted * time_weights.view(*view_shape)

        if (
            first_close_joint_weight != 1.0
            and actions is not None
            and weighted.shape == actions.shape
            and weighted.ndim >= 2
            and weighted.shape[-1] > 0
            and first_close_window_before >= 0
            and first_close_window_after >= 0
        ):
            close_phase_mask = self._first_close_phase_mask(
                actions=actions,
                gripper_dims=gripper_dims,
                close_threshold=gripper_close_threshold,
                window_before=first_close_window_before,
                window_after=first_close_window_after,
            )
            if close_phase_mask is not None:
                joint_dim_mask = torch.ones(
                    weighted.shape[-1],
                    device=weighted.device,
                    dtype=torch.bool,
                )
                for dim in gripper_dims:
                    if -weighted.shape[-1] <= dim < weighted.shape[-1]:
                        joint_dim_mask[dim % weighted.shape[-1]] = False
                if joint_dim_mask.any():
                    view_shape = [1] * weighted.ndim
                    view_shape[-1] = weighted.shape[-1]
                    phase_joint_mask = close_phase_mask.unsqueeze(-1) & joint_dim_mask.view(
                        *view_shape
                    )
                    phase_weight = torch.as_tensor(
                        first_close_joint_weight,
                        device=weighted.device,
                        dtype=weighted.dtype,
                    )
                    weighted = torch.where(
                        phase_joint_mask,
                        weighted * phase_weight,
                        weighted,
                    )

        if action_weight != 1.0:
            weighted = weighted * action_weight
        return weighted

    def _first_close_phase_mask(
        self,
        actions: torch.Tensor,
        gripper_dims: list[int],
        close_threshold: float,
        window_before: int,
        window_after: int,
    ) -> torch.Tensor | None:
        """Return a mask over action time steps around each trajectory's first close."""
        if actions.ndim < 2 or actions.shape[-1] <= 0:
            return None

        close_mask = torch.zeros(
            actions.shape[:-1],
            device=actions.device,
            dtype=torch.bool,
        )
        has_gripper_dim = False
        for dim in gripper_dims:
            if -actions.shape[-1] <= dim < actions.shape[-1]:
                has_gripper_dim = True
                close_mask |= actions[..., dim % actions.shape[-1]] < close_threshold
        if not has_gripper_dim:
            return None

        time_steps = close_mask.shape[-1]
        if time_steps <= 0:
            return None

        flat_close = close_mask.reshape(-1, time_steps)
        has_close = flat_close.any(dim=1)
        first_close = flat_close.to(torch.int64).argmax(dim=1)
        time_idx = torch.arange(time_steps, device=actions.device).unsqueeze(0)
        start = (first_close - window_before).unsqueeze(1)
        end = (first_close + window_after).unsqueeze(1)
        flat_phase = has_close.unsqueeze(1) & (time_idx >= start) & (time_idx <= end)
        return flat_phase.reshape(close_mask.shape)

    def _config_float(self, name: str, default: float) -> float:
        value = getattr(self.config, name, default)
        if value is None:
            value = default
        return float(value)

    def _config_int(self, name: str, default: int) -> int:
        value = getattr(self.config, name, default)
        if value is None:
            value = default
        return int(value)

    def _sigma_for_timestep(
        self,
        timestep: torch.Tensor,
        like: torch.Tensor,
    ) -> torch.Tensor:
        timestep_ref = timestep.detach().to(self.scheduler.timesteps.device)
        timestep_id = torch.argmin(
            (
                self.scheduler.timesteps.unsqueeze(1)
                - timestep_ref.flatten().unsqueeze(0)
            ).abs(),
            dim=0,
        )
        sigma = self.scheduler.sigmas[timestep_id].to(
            device=like.device,
            dtype=like.dtype,
        )
        sigma = sigma.reshape(timestep.shape)
        while sigma.ndim < like.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _compute_gripper_clean_action_loss(
        self,
        clean_action_pred: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        has_real_action: torch.Tensor,
    ) -> torch.Tensor:
        loss_weight = self._config_float("gripper_clean_action_loss_weight", 0.0)
        if loss_weight == 0.0:
            return torch.tensor(0.0, device=clean_action_pred.device)
        if clean_action_pred.shape != actions.shape:
            raise ValueError(
                "clean_action_pred and actions must have the same shape: "
                f"{tuple(clean_action_pred.shape)} != {tuple(actions.shape)}"
            )

        close_weight = self._config_float(
            "gripper_clean_close_action_loss_weight",
            1.0,
        )
        close_threshold = self._config_float("gripper_close_threshold", 0.0)
        gripper_dims = [
            int(dim) for dim in getattr(self.config, "gripper_action_dims", [7])
        ]

        dim_mask = torch.zeros(
            actions.shape[-1],
            device=actions.device,
            dtype=torch.bool,
        )
        for dim in gripper_dims:
            if -actions.shape[-1] <= dim < actions.shape[-1]:
                dim_mask[dim % actions.shape[-1]] = True
        if not dim_mask.any():
            return torch.tensor(0.0, device=clean_action_pred.device)

        view_shape = [1] * actions.ndim
        view_shape[-1] = actions.shape[-1]
        valid = dim_mask.view(*view_shape).expand_as(actions) & action_mask.bool()
        real_action_mask = has_real_action.bool()
        while real_action_mask.ndim < valid.ndim:
            real_action_mask = real_action_mask.unsqueeze(-1)
        valid = valid & real_action_mask

        weights = torch.ones_like(actions, dtype=clean_action_pred.dtype)
        if close_weight != 1.0:
            close_weight_tensor = torch.as_tensor(
                close_weight,
                device=actions.device,
                dtype=clean_action_pred.dtype,
            )
            for dim in gripper_dims:
                if -actions.shape[-1] <= dim < actions.shape[-1]:
                    dim_idx = dim % actions.shape[-1]
                    close_mask = actions[..., dim_idx] < close_threshold
                    weights[..., dim_idx] = torch.where(
                        close_mask,
                        close_weight_tensor,
                        weights[..., dim_idx],
                    )

        clean_loss = torch.nn.functional.mse_loss(
            clean_action_pred.float(),
            actions.float(),
            reduction="none",
        )
        valid_f = valid.to(dtype=clean_loss.dtype)
        weighted = clean_loss * weights.float() * valid_f
        denom = valid_f.sum().clamp_min(1.0)
        return weighted.sum() / denom * loss_weight

    def _compute_gripper_binary_action_loss(
        self,
        clean_action_pred: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        has_real_action: torch.Tensor,
    ) -> torch.Tensor:
        loss_weight = self._config_float("gripper_binary_action_loss_weight", 0.0)
        if loss_weight == 0.0:
            return torch.tensor(0.0, device=clean_action_pred.device)
        if clean_action_pred.shape != actions.shape:
            raise ValueError(
                "clean_action_pred and actions must have the same shape: "
                f"{tuple(clean_action_pred.shape)} != {tuple(actions.shape)}"
            )

        close_weight = self._config_float(
            "gripper_binary_close_action_loss_weight",
            1.0,
        )
        close_threshold = self._config_float("gripper_close_threshold", 0.0)
        logit_scale = self._config_float("gripper_binary_logit_scale", 4.0)
        gripper_dims = [
            int(dim) for dim in getattr(self.config, "gripper_action_dims", [7])
        ]

        dim_mask = torch.zeros(
            actions.shape[-1],
            device=actions.device,
            dtype=torch.bool,
        )
        for dim in gripper_dims:
            if -actions.shape[-1] <= dim < actions.shape[-1]:
                dim_mask[dim % actions.shape[-1]] = True
        if not dim_mask.any():
            return torch.tensor(0.0, device=clean_action_pred.device)

        view_shape = [1] * actions.ndim
        view_shape[-1] = actions.shape[-1]
        valid = dim_mask.view(*view_shape).expand_as(actions) & action_mask.bool()
        real_action_mask = has_real_action.bool()
        while real_action_mask.ndim < valid.ndim:
            real_action_mask = real_action_mask.unsqueeze(-1)
        valid = valid & real_action_mask

        close_targets = actions.float() < close_threshold
        logits = -clean_action_pred.float() * float(logit_scale)
        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            close_targets.to(dtype=logits.dtype),
            reduction="none",
        )

        weights = torch.ones_like(binary_loss)
        if close_weight != 1.0:
            close_weight_tensor = torch.as_tensor(
                close_weight,
                device=actions.device,
                dtype=binary_loss.dtype,
            )
            weights = torch.where(close_targets, close_weight_tensor, weights)

        valid_f = valid.to(dtype=binary_loss.dtype)
        weighted = binary_loss * weights * valid_f
        denom = valid_f.sum().clamp_min(1.0)
        return weighted.sum() / denom * loss_weight

    def _gripper_clean_sigma_mask(self, sigma: torch.Tensor) -> torch.Tensor:
        max_sigma = self._config_float("gripper_clean_max_sigma", 1.0)
        if not (0.0 <= max_sigma <= 1.0):
            raise ValueError(f"gripper_clean_max_sigma must be in [0, 1], got {max_sigma}")
        sigma_f = sigma.float()
        return ((1.0 - sigma_f) > 1e-4) & (sigma_f <= max_sigma)

    def _gripper_binary_sigma_mask(self, sigma: torch.Tensor) -> torch.Tensor:
        max_sigma = self._config_float("gripper_binary_max_sigma", 1.0)
        if not (0.0 <= max_sigma <= 1.0):
            raise ValueError(f"gripper_binary_max_sigma must be in [0, 1], got {max_sigma}")
        sigma_f = sigma.float()
        return ((1.0 - sigma_f) > 1e-4) & (sigma_f <= max_sigma)

    def _reconstruct_clean_sample_from_flow_target(
        self,
        noisy_sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Invert this scheduler's training target back to the clean sample.

        FlowMatchScheduler.add_noise uses ``z_t = (1-sigma) * x0 + sigma * noise``
        and its training target is ``noise - x0``. Since
        ``z_t = x0 + sigma * target``, the clean sample is
        ``x0 = z_t - sigma * target``.
        """
        return noisy_sample.float() - sigma.float() * model_output.float()

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_diffusion_model:
                self.model.eval()
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()
    
    
    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.dtype,
                computation_device='cuda',
            ),
        )

        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        print("offloadd")
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                # print("offload", module)
                                module.offload()
                    else:
                        print("tocpu")
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    print("onload")
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            # print("onload", module)
                            module.onload()
                else:
                    print("togpu")
                    model.to(self._device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def _create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seqlen: int,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU KV cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        kv_cache1: KVCacheType = []
        kv_cache_neg: KVCacheType = []
        for _ in range(self.model.num_layers):
            kv_cache1.append(
                torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
            )
            kv_cache_neg.append(
                torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
            )

        return kv_cache1, kv_cache_neg

    def _create_crossattn_caches(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        crossattn_cache: KVCacheType = []
        crossattn_cache_neg: KVCacheType = []

        for _ in range(self.model.num_layers):
            crossattn_cache.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )
            crossattn_cache_neg.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )

        return crossattn_cache, crossattn_cache_neg
        
    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _reset_cached_video_state(self) -> None:
        self.kv_cache1 = None
        self.kv_cache_neg = None
        self.crossattn_cache = None
        self.crossattn_cache_neg = None
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self._ma_cached_token_agent_id = None
        self._ma_cached_token_agent_id_neg = None
        if hasattr(self.model, "_cached_token_agent_id"):
            self.model._cached_token_agent_id = None

    def reset_causal_state(self) -> None:
        """Reset stateful causal video/action inference between episodes."""
        self._reset_cached_video_state()
        self.language = None

    def preprocess_image(self, image):
        image = (image * (2 / 255) - 1).permute(0, 1, 4, 2, 3)
        return image

    def encode_prompt(self, input_ids, attention_mask):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def _ensure_vae_on_device(self, ref_tensor):
        """Lazily move the VAE to the correct device/dtype on first use."""
        if not getattr(self, '_vae_device_ready', False):
            self.vae.to(device=ref_tensor.device, dtype=torch.bfloat16)
            self.vae.eval()
            self._vae_device_ready = True

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        self._ensure_vae_on_device(input_video)
        with torch.no_grad():
            latents = self.vae.encode(input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            batch_size = image.shape[0]
            clip_context = self.image_encoder.encode_image(image)
            image_input = image.transpose(1, 2)
            image_zeros = torch.zeros(batch_size, 3, num_frames-1, height, width, dtype=torch.bfloat16, device=self._device)
            self._ensure_vae_on_device(image_input)
            with torch.no_grad():
                y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))
            # Build mask to match VAE output shape (VAE may use different spatial downsampling, e.g. WanVideoVAE38 uses patch_size=2 -> height/16)
            # y shape is B * 16 * (1+(T-1)/4) * H_latent * W_latent
            num_t = y.shape[2]
            h_latent, w_latent = y.shape[3], y.shape[4]
            msk = torch.zeros(batch_size, 4, num_t, h_latent, w_latent, dtype=y.dtype, device=self._device)
            msk[:, :, 0:1, :, :] = 1
            new_image = y[:, :, 0:1]
            # concat: B * (4+16) * (1+(T-1)/4) * H_latent * W_latent
            y = torch.concat([msk, y], dim=1)
        return clip_context, y, new_image

    def _prepare_multi_agent_i2v_conditioning(
        self,
        videos: torch.Tensor,
        latents: torch.Tensor,
        condition_frame_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Build per-agent I2V conditioning for the multi-agent path.

        ``videos`` is normalized ``[B, P, C, T, H, W]`` and ``latents`` is
        clean VAE latent ``[B, P, C_lat, F_lat, H_lat, W_lat]``. The
        returned ``clean_x`` repeats the selected current-observation
        latent across the latent time axis so training and closed-loop
        inference condition on observed frames only, not future video.
        """
        model_type = getattr(self.model, "model_type", "t2v")
        if model_type not in ("i2v", "ti2v"):
            return None, None, None
        if not hasattr(self, "image_encoder") or not hasattr(self, "vae"):
            return None, None, None

        assert videos.dim() == 6, (
            f"videos must be [B, P, C, T, H, W]; got {tuple(videos.shape)}"
        )
        assert latents.dim() == 6, (
            f"latents must be [B, P, C, F, H, W]; got {tuple(latents.shape)}"
        )
        b, p, c, t, h, w = videos.shape
        assert latents.shape[0] == b and latents.shape[1] == p

        frame = videos[:, :, :, condition_frame_index:condition_frame_index + 1]
        if condition_frame_index < 0:
            frame = videos[:, :, :, condition_frame_index:]
        assert frame.shape[3] == 1, (
            f"conditioning frame slice must contain one frame; got {tuple(frame.shape)}"
        )

        image = frame.permute(0, 1, 3, 2, 4, 5).reshape(b * p, 1, c, h, w)
        clip_bp, y_bp, _ = self.encode_image(image, t, h, w)
        clip_feature = clip_bp.reshape(b, p, *clip_bp.shape[1:]).to(self._device)
        y = y_bp.reshape(b, p, *y_bp.shape[1:]).to(self._device)

        latent_frame_index = condition_frame_index
        clean_frame = latents[:, :, :, latent_frame_index:latent_frame_index + 1]
        if latent_frame_index < 0:
            clean_frame = latents[:, :, :, latent_frame_index:]
        clean_x = clean_frame.expand(
            -1, -1, -1, latents.shape[3], -1, -1
        ).contiguous()

        return clip_feature, y, clean_x
    
    def prepare_extra_input(self, latents=None):
        return {}

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming") -> nn.Module:
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_config)
        for param in model.parameters():
            param.data = param.to(torch.float32)
        return model

    def _encode_global_video(self, video_global: "torch.Tensor") -> "torch.Tensor":
        """VAE-encode a shared scene video stream that has no agent axis.

        Mirrors the per-agent encode block in
        :meth:`_forward_multi_agent` -- normalize uint8 -> [-1, 1],
        optionally resize to (``target_video_height``, ``target_video_width``)
        if the action-head config pins those, then VAE encode with the
        head's own tile config. Returns a clean latent tensor (no
        noise added) of shape ``[B, C_lat, F_lat, H_lat, W_lat]``.

        Called from both training (:meth:`_forward_multi_agent`) and
        inference (:meth:`_get_action_multi_agent`) when the data
        transform emitted ``video_global`` under the shared-global
        layout (see :class:`BimanualDreamTransform.global_views`).
        """
        # Accept either [B, T, H, W, C] (C-last from the transform) or
        # [B, C, T, H, W] (already model-orientation).
        if video_global.dim() == 5 and video_global.shape[-1] in (1, 3):
            video_global = rearrange(video_global, "b t h w c -> b c t h w")
        assert video_global.dim() == 5, (
            f"video_global must be [B, T, H, W, C] or [B, C, T, H, W]; "
            f"got {tuple(video_global.shape)}"
        )

        if video_global.dtype == torch.uint8:
            video_global = video_global.float() / 255.0
            b, c, t, h, w = video_global.shape
            video_global = video_global.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            video_global = video_global.reshape(b * t, c, h, w)
            video_global = self.normalize_video(video_global)
            video_global = video_global.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            assert video_global.min() >= -1.0 and video_global.max() <= 1.0, (
                "video_global must normalize into [-1, 1]"
            )
            video_global = video_global.to(dtype=self.dtype)

        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            b, c, t, h, w = video_global.shape
            if (h, w) != (target_h, target_w):
                video_global = torch.nn.functional.interpolate(
                    video_global.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        latents = self.encode_video(
            video_global,
            self.tiled,
            (self.tile_size_height, self.tile_size_width),
            (self.tile_stride_height, self.tile_stride_width),
        )                                                    # [B, C_lat, F_lat, H_lat, W_lat]
        return latents.to(self._device)

    def _detect_multi_agent(self, action_input: BatchFeature) -> int | None:
        """Return ``P`` if ``action_input`` carries an explicit agent axis,
        else ``None``.

        BimanualDreamTransform stacks state / action / images with a
        leading ``P`` axis. After the trainer's per-sample collate the
        shapes become ``[B, P, ...]``; we sniff that here by looking at
        ``state`` and ``action`` ndim. Falls back to the single-agent
        path when no P axis is present.
        """
        state = getattr(action_input, "state", None)
        actions = getattr(action_input, "action", None)
        # Single-agent state shape: [B, T_s, D]. Multi-agent: [B, P, T_s, D].
        if state is not None and state.ndim == 4:
            return int(state.shape[1])
        if actions is not None and actions.ndim == 4:
            return int(actions.shape[1])
        return None

    def _forward_multi_agent(
        self, backbone_output: BatchFeature, action_input: BatchFeature, num_agents: int,
    ) -> BatchFeature:
        """Multi-agent training forward (PR 9d).

        Mirrors the single-agent ``forward`` but threads an explicit
        agent axis ``P`` through VAE encode, noise sampling, the model
        call, and the dynamics + action loss reduction. The VAE / image
        / text encoders themselves are shared (single set of weights);
        the per-agent split is done by collapsing ``B*P`` into the VAE
        batch dim, then restoring the ``P`` axis before the diffusion
        forward.

        Expected ``action_input`` (per :class:`BimanualDreamTransform` +
        trainer collation):

          * ``state``:        ``[B, P, T_s, D_s]``
          * ``action``:       ``[B, P, T_a, D_a]``
          * ``action_mask``:  ``[B, P, T_a, D_a]``
          * ``images``:       ``[B, P, T, H, W, C]`` (C-last, per-agent
            already V-tiled by the data transform) -- matches the
            single-agent ``[B, T, H, W, C]`` convention with a leading P.
          * ``embodiment_id``, ``has_real_action``: ``[B]`` (shared)
          * ``text``, ``text_attention_mask``:     shared across agents

        Image-to-video conditioning is preserved for Wan I2V/TI2V models:
        each agent gets CLIP/y conditioning from its observed frame and a
        clean current-observation latent prefix. Training uses frame 0
        from the sampled window; closed-loop inference uses the latest
        frame from the rolling history.
        """
        self.set_frozen_modules_to_eval_mode()

        data = action_input
        embodiment_id = action_input.embodiment_id
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask

        state_features = action_input.state  # [B, P, T_s, D_s]
        actions = action_input.action        # [B, P, T_a, D_a]
        assert actions.dim() == 4 and actions.shape[1] == num_agents, (
            f"multi-agent action must be [B, P, T_a, D_a]; got {tuple(actions.shape)}"
        )
        assert state_features.dim() == 4 and state_features.shape[1] == num_agents, (
            f"multi-agent state must be [B, P, T_s, D_s]; "
            f"got {tuple(state_features.shape)}"
        )
        B, P = actions.shape[0], actions.shape[1]

        if actions.numel() > 0:
            assert actions.min() >= -1.0 and actions.max() <= 1.0, (
                "actions must be in [-1,1] range"
            )

        videos = data["images"]
        # Expected shape: [B, P, T, H, W, C]. The single-agent path uses
        # [B, T, H, W, C]; we extend with a leading P. Per-agent view
        # tiling (mapping V_per_agent->1) is the data transform's job.
        assert videos.dim() == 6, (
            f"multi-agent images must be [B, P, T, H, W, C]; "
            f"got {tuple(videos.shape)}. Per-agent view tiling must be "
            f"done in the data transform."
        )
        assert videos.shape[1] == P
        videos = rearrange(videos, "b p t h w c -> b p c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, p, c, t, h, w = videos.shape
            videos = videos.permute(0, 1, 3, 2, 4, 5)  # [B, P, T, C, H, W]
            videos = videos.reshape(b * p * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, p, t, c, h, w).permute(0, 1, 3, 2, 4, 5)
            assert videos.min() >= -1.0 and videos.max() <= 1.0, (
                "videos must be in [-1,1] range"
            )
            videos = videos.to(dtype=self.dtype)

        prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])

        # Wan 5B-style resize (same policy as single-agent).
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, p, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * p * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, p, c, t, target_h, target_w)

        # VAE encode per agent: collapse B*P into the VAE batch dim.
        b, p, c, t, h, w = videos.shape
        videos_bp = videos.reshape(b * p, c, t, h, w)
        latents_bp = self.encode_video(
            videos_bp,
            self.tiled,
            (self.tile_size_height, self.tile_size_width),
            (self.tile_stride_height, self.tile_stride_width),
        )
        _, c_lat, F_lat, h_lat, w_lat = latents_bp.shape
        # [B, P, C_lat, F_lat, H_lat, W_lat]
        latents = latents_bp.reshape(b, p, c_lat, F_lat, h_lat, w_lat)
        latents = latents.to(self._device)
        prompt_embs = prompt_embs.to(self._device)
        clip_features, ys, clean_latents = self._prepare_multi_agent_i2v_conditioning(
            videos=videos,
            latents=latents,
            condition_frame_index=0,
        )
        if ys is not None:
            ys = ys.to(dtype=latents.dtype)
        if clean_latents is not None:
            clean_latents = clean_latents.to(dtype=latents.dtype)

        # Noise + transpose to put frame axis second (matches the
        # single-agent layout, just with an extra leading P).
        # latents / noise after transpose: [B, P, F, C_lat, H_lat, W_lat].
        noise = torch.randn_like(latents).transpose(2, 3)
        latents = latents.transpose(2, 3)

        # ============ VIDEO TIMESTEP SAMPLING (shared across agents) ============
        if self.config.decouple_video_action_noise:
            video_noise_ratio = self.video_beta_dist.sample([B, F_lat])
            timestep_id = (
                (1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps
            ).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "DECOUPLED"
        elif self.config.use_high_noise_emphasis:
            noise_ratio = self.high_noise_beta_dist.sample([B, F_lat])
            timestep_id = (
                (1.0 - noise_ratio) * self.scheduler.num_train_timesteps
            ).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "HIGH_NOISE_EMPHASIS"
        else:
            timestep_id = torch.randint(
                0, self.scheduler.num_train_timesteps, (B, F_lat)
            )
            noise_mode = "STANDARD"

        timestep_id_block = timestep_id[:, 1:].reshape(
            B, -1, self.num_frame_per_block
        )
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)  # [B, P, T_a, D_a]
            T_a = actions.shape[2]
            assert T_a / (F_lat - 1) == (
                self.model.num_action_per_block // self.num_frame_per_block
            ), (
                f"actions.shape={tuple(actions.shape)}, "
                f"noise.shape={tuple(noise.shape)}, video.shape={tuple(videos.shape)}, "
                f"latents.shape={tuple(latents.shape)}"
            )
            assert (F_lat - 1) / state_features.shape[2] == (
                self.num_frame_per_block // self.model.num_state_per_block
            ), (
                f"state_features.shape={tuple(state_features.shape)}, "
                f"noise.shape={tuple(noise.shape)}, video.shape={tuple(videos.shape)}, "
                f"latents.shape={tuple(latents.shape)}"
            )

            # ============ ACTION TIMESTEP SAMPLING (shared across agents) ============
            if self.config.decouple_video_action_noise:
                timestep_action_id = torch.randint(
                    0, self.scheduler.num_train_timesteps, (B, T_a)
                )
                action_mode = "INDEPENDENT"
            else:
                timestep_action_id = timestep_id_block.repeat(
                    1, 1, T_a // (F_lat - 1)
                )
                timestep_action_id = timestep_action_id.reshape(B, -1)
                action_mode = "COUPLED"

            if not self._noise_logged:
                video_mean = timestep_id.float().mean().item()
                action_mean = timestep_action_id.float().mean().item()
                print(
                    f"[NOISE][multi-agent P={P}] Mode={noise_mode} | "
                    f"Video mean_t={video_mean:.0f} | "
                    f"Action mean_t={action_mean:.0f} ({action_mode})"
                )
                self._noise_logged = True
        else:
            noise_action = None
            timestep_action_id = None
            T_a = 0

        timestep_id_block = timestep_id_block.reshape(B, -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        timestep = self.scheduler.timesteps[timestep_id].to(self._device)  # [B, F]

        # Expand timestep along P so add_noise sees a flat [B*P*F] tensor
        # whose sigma broadcasts back to [B, P, F, C, H, W].
        timestep_BPF = timestep.unsqueeze(1).expand(B, P, F_lat).contiguous()
        noisy_latents = self.scheduler.add_noise(
            latents.flatten(0, 2),
            noise.flatten(0, 2),
            timestep_BPF.flatten(0, 2),
        ).unflatten(0, (B, P, F_lat))
        # training_target = noise - sample (shape-agnostic).
        # Transpose to put channel axis where the model emits it.
        training_target = self.scheduler.training_target(
            latents, noise, timestep_BPF
        ).transpose(2, 3)  # [B, P, C, F, H, W]

        if actions.numel() > 0:
            timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)  # [B, T_a]
            timestep_action_BPT = (
                timestep_action.unsqueeze(1).expand(B, P, T_a).contiguous()
            )
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 2),
                noise_action.flatten(0, 2),
                timestep_action_BPT.flatten(0, 2),
            ).unflatten(0, (B, P, T_a))
            training_target_action = self.scheduler.training_target(
                actions, noise_action, timestep_action_BPT
            )
        else:
            timestep_action = None
            noisy_actions = None
            training_target_action = None

        # Shared-global stream (PR 23): when the data transform emitted
        # ``video_global`` (BimanualDreamTransform.global_views set), VAE-
        # encode it once -- no agent axis, no noise -- and pass the latent
        # to the model as clean conditioning. The DiT body splices it into
        # the token sequence as a P-less block (see
        # ``_forward_multi_agent_body``'s ``global_video`` kwarg).
        # Training stays in "predict wrist only" mode: ``noisy_latents`` is
        # still per-agent wrist, ``training_target`` still per-agent wrist.
        if isinstance(data, dict):
            video_global_raw = data.get("video_global", None)
        else:
            video_global_raw = getattr(data, "video_global", None)
        if video_global_raw is not None:
            global_latents = self._encode_global_video(video_global_raw)
            global_latents = global_latents.to(dtype=self.dtype)
            # The DiT body expects channel-first model orientation matching
            # what we feed for noisy_latents.transpose(2, 3) below:
            # [B, C, F, H_lat, W_lat]. encode_video already returns this.
        else:
            global_latents = None

        # Sequence length: P * F * H_grid * W_grid where the grid is the
        # post-patch_embedding shape (stride (1,2,2) -> H_lat//2, W_lat//2).
        # When global_latents is present its tokens (F * H_g_global * W_g_global)
        # extend the sequence; the DiT body handles that internally and the
        # outer ``seq_len`` argument still refers to the per-agent video
        # block length (the body adds its own offsets for global/register/hub).
        H_g = h_lat // 2
        W_g = w_lat // 2
        seq_len = P * F_lat * H_g * W_g

        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self._device).type
        ):
            if actions.numel() > 0:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(2, 3),  # [B, P, C, F, H, W]
                    timestep=timestep,
                    context=prompt_embs,
                    seq_len=seq_len,
                    state=state_features,
                    embodiment_id=embodiment_id,
                    action=noisy_actions,
                    timestep_action=timestep_action,
                    clip_feature=clip_features,
                    y=ys,
                    clean_x=clean_latents,
                    global_video=global_latents,
                )
            else:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(2, 3),
                    timestep=timestep,
                    timestep_action=timestep_action,
                    context=prompt_embs,
                    seq_len=seq_len,
                    state=state_features,
                    embodiment_id=embodiment_id,
                    clip_feature=clip_features,
                    y=ys,
                    clean_x=clean_latents,
                    global_video=global_latents,
                )

            # Per-sample dynamics loss. Crop target to model output spatial
            # size if patch_embedding stride 2 truncates an odd dim.
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[4], : video_noise_pred.shape[5]
                ]
            # Mean over (C, H, W) -> [B, P, F]
            dynamics_loss_per_sample = torch.nn.functional.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction="none"
            ).mean(dim=(2, 4, 5))
            train_w = (
                self.scheduler.training_weight(timestep.flatten(0, 1))
                .unflatten(0, (B, F_lat))
                .to(self._device)
            )  # [B, F]
            weight_dynamics = dynamics_loss_per_sample * train_w.unsqueeze(1)
            weighted_dynamics_loss = weight_dynamics.mean()

            if actions.numel() > 0:
                # action_noise_pred / target: [B, P, T_a, D_a]; mask same shape.
                action_loss_per_sample = torch.nn.functional.mse_loss(
                    action_noise_pred.float(),
                    training_target_action.float(),
                    reduction="none",
                ) * action_mask
                # has_real_action [B] -> [B, 1, 1, 1] for broadcast.
                action_loss_per_sample = (
                    has_real_action[:, None, None, None].float()
                    * action_loss_per_sample
                )
                action_loss_per_sample = self._apply_action_loss_weights(
                    action_loss_per_sample,
                    actions=actions,
                )
                train_w_action = (
                    self.scheduler.training_weight(timestep_action.flatten(0, 1))
                    .unflatten(0, (B, T_a))
                    .to(self._device)
                )  # [B, T_a]
                weight_action = action_loss_per_sample.mean(dim=3) * train_w_action.unsqueeze(1)
                weighted_action_loss = weight_action.mean()
                gripper_clean_action_loss = torch.tensor(0.0, device=self._device)
                gripper_binary_action_loss = torch.tensor(0.0, device=self._device)
                needs_gripper_clean_pred = (
                    float(
                        getattr(
                            self.config,
                            "gripper_clean_action_loss_weight",
                            0.0,
                        )
                        or 0.0
                    )
                    != 0.0
                    or float(
                        getattr(
                            self.config,
                            "gripper_binary_action_loss_weight",
                            0.0,
                        )
                        or 0.0
                    )
                    != 0.0
                )
                if needs_gripper_clean_pred:
                    sigma_action = self._sigma_for_timestep(
                        timestep_action_BPT,
                        noisy_actions,
                    )
                    clean_action_pred = self._reconstruct_clean_sample_from_flow_target(
                        noisy_sample=noisy_actions,
                        model_output=action_noise_pred,
                        sigma=sigma_action,
                    )
                if float(
                    getattr(
                        self.config,
                        "gripper_clean_action_loss_weight",
                        0.0,
                    )
                    or 0.0
                ) != 0.0:
                    clean_action_mask = action_mask.bool() & self._gripper_clean_sigma_mask(
                        sigma_action
                    )
                    gripper_clean_action_loss = self._compute_gripper_clean_action_loss(
                        clean_action_pred=clean_action_pred,
                        actions=actions,
                        action_mask=clean_action_mask,
                        has_real_action=has_real_action,
                    )
                if float(
                    getattr(
                        self.config,
                        "gripper_binary_action_loss_weight",
                        0.0,
                    )
                    or 0.0
                ) != 0.0:
                    binary_action_mask = action_mask.bool() & self._gripper_binary_sigma_mask(
                        sigma_action
                    )
                    gripper_binary_action_loss = self._compute_gripper_binary_action_loss(
                        clean_action_pred=clean_action_pred,
                        actions=actions,
                        action_mask=binary_action_mask,
                        has_real_action=has_real_action,
                    )
                loss = (
                    weighted_dynamics_loss
                    + weighted_action_loss
                    + gripper_clean_action_loss
                    + gripper_binary_action_loss
                )
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)
                gripper_clean_action_loss = torch.tensor(0.0, device=self._device)
                gripper_binary_action_loss = torch.tensor(0.0, device=self._device)
                loss = weighted_dynamics_loss

        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
            "gripper_clean_action_loss": gripper_clean_action_loss,
            "gripper_binary_action_loss": gripper_binary_action_loss,
        }
        return BatchFeature(data=output_dict)

    def _get_action_multi_agent_causal(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        num_agents: int,
    ) -> BatchFeature:
        """Multi-agent DreamZero causal inference.

        This mirrors the original single-agent ``lazy_joint_video_action``
        sampling semantics for P-agent shared-global checkpoints: prime a
        persistent KV cache with clean observed video, denoise future video
        and action jointly with CFG + UniPC, and keep latent video state via
        ``current_start_frame``.
        """
        del backbone_output
        self.set_frozen_modules_to_eval_mode()
        data = action_input

        embodiment_id = action_input.embodiment_id
        state_features = action_input.state             # [B, P, T_s, D_s]
        actions = action_input.action                   # [B, P, T_a, D_a]
        assert actions.dim() == 4 and actions.shape[1] == num_agents
        B, P = actions.shape[0], actions.shape[1]
        T_a, D_a = actions.shape[2], actions.shape[3]

        videos = data["images"]                         # [B, P, T, H, W, C]
        assert videos.dim() == 6 and videos.shape[1] == P
        raw_num_video_frames = videos.shape[2]
        videos = rearrange(videos, "b p t h w c -> b p c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, p, c, t, h, w = videos.shape
            videos = videos.permute(0, 1, 3, 2, 4, 5)
            videos = videos.reshape(b * p * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, p, t, c, h, w).permute(0, 1, 3, 2, 4, 5)
            assert videos.min() >= -1.0 and videos.max() <= 1.0
            videos = videos.to(dtype=self.dtype)

        reset_needed = False
        if self.language is None:
            reset_needed = True
        elif not torch.equal(self.language, data["text"]):
            reset_needed = True
        elif raw_num_video_frames == 1:
            reset_needed = True
        elif self.current_start_frame >= getattr(self.model, "local_attn_size", 10**9):
            reset_needed = True
        if reset_needed:
            self._reset_cached_video_state()
            self.language = data["text"].detach().clone()

        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [
            self.encode_prompt(text, attention_mask).to(self._device)
            for text, attention_mask in text_inputs
        ]

        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, p, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * p * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, p, c, t, target_h, target_w)

        # VAE encode per-agent current/repeated observation windows.
        b, p, c, t, h, w = videos.shape
        videos_bp = videos.reshape(b * p, c, t, h, w)
        latents_bp = self.encode_video(
            videos_bp,
            self.tiled,
            (self.tile_size_height, self.tile_size_width),
            (self.tile_stride_height, self.tile_stride_width),
        )
        _, c_lat, F_lat, h_lat, w_lat = latents_bp.shape
        latents = latents_bp.reshape(b, p, c_lat, F_lat, h_lat, w_lat).to(
            self._device
        )

        if isinstance(data, dict):
            video_global_raw = data.get("video_global", None)
        else:
            video_global_raw = getattr(data, "video_global", None)
        condition_frame_index = 0 if video_global_raw is not None else -1
        clip_features, ys, clean_latents = self._prepare_multi_agent_i2v_conditioning(
            videos=videos,
            latents=latents,
            condition_frame_index=condition_frame_index,
        )
        if ys is not None:
            ys = ys.to(dtype=latents.dtype)
        if clean_latents is not None:
            clean_latents = clean_latents.to(dtype=latents.dtype)
        if self.current_start_frame == 0:
            self.clip_feas = (
                clip_features.to(dtype=latents.dtype) if clip_features is not None else None
            )
            self.ys = ys
        assert self.clip_feas is not None and self.ys is not None, (
            "multi-agent causal I2V inference requires clip/y conditioning"
        )
        self._last_clean_video_cond = (
            clean_latents.detach() if clean_latents is not None else None
        )
        self._last_y_video_cond = ys.detach() if ys is not None else None

        if video_global_raw is not None:
            global_latents = self._encode_global_video(video_global_raw).to(
                dtype=latents.dtype
            )
        else:
            global_latents = None

        block = self.num_frame_per_block
        assert block >= 1
        H_g = h_lat // 2
        W_g = w_lat // 2
        seq_len = P * block * H_g * W_g
        frame_seqlen = P * H_g * W_g
        current_image = (
            clean_latents[:, :, :, :1] if clean_latents is not None else latents[:, :, :, :1]
        )
        current_image = current_image.to(dtype=latents.dtype)

        def _slice_latent_frames(
            tensor: torch.Tensor | None,
            start: int,
            length: int,
        ) -> torch.Tensor | None:
            if tensor is None:
                return None
            frame_dim = 3 if tensor.dim() == 6 else 2
            total = tensor.shape[frame_dim]
            if total == length:
                return tensor
            start = max(min(start, max(total - length, 0)), 0)
            return tensor.narrow(frame_dim, start, length)

        def _repeat_current_to_block(image: torch.Tensor) -> torch.Tensor:
            if block == 1:
                return image
            return image.expand(-1, -1, -1, block, -1, -1).contiguous()

        if self.current_start_frame == 0:
            self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
                batch_size=B,
                dtype=latents.dtype,
                device=latents.device,
                frame_seqlen=frame_seqlen,
            )
            self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
                batch_size=B,
                dtype=latents.dtype,
                device=latents.device,
            )
            self._ma_cached_token_agent_id = None
            self._ma_cached_token_agent_id_neg = None

        assert self.kv_cache1 is not None and self.kv_cache_neg is not None
        assert self.crossattn_cache is not None and self.crossattn_cache_neg is not None
        kv_caches = self._get_caches([self.kv_cache1, self.kv_cache_neg])
        crossattn_caches = self._get_caches(
            [self.crossattn_cache, self.crossattn_cache_neg]
        )

        zero_step = torch.zeros([B, 1], device=latents.device, dtype=torch.int64)
        if self.current_start_frame == 0:
            self._run_multi_agent_diffusion_steps(
                noisy_input=current_image,
                timestep=zero_step,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=P * H_g * W_g,
                y=_slice_latent_frames(self.ys, 0, 1),
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(start_frame=0, update_kv_cache=True),
                clean_x=_slice_latent_frames(clean_latents, 0, 1),
                global_video=_slice_latent_frames(global_latents, 0, 1),
            )
            self.current_start_frame += 1

        if self.current_start_frame != 1:
            ref_block = _repeat_current_to_block(current_image)
            ref_start = self.current_start_frame - block
            self._run_multi_agent_diffusion_steps(
                noisy_input=ref_block,
                timestep=torch.zeros([B, block], device=latents.device, dtype=torch.int64),
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=seq_len,
                y=_slice_latent_frames(self.ys, ref_start, block),
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=ref_start,
                    update_kv_cache=True,
                ),
                clean_x=_slice_latent_frames(clean_latents, ref_start, block),
                global_video=_slice_latent_frames(global_latents, ref_start, block),
            )

        noisy_video = self.generate_noise(
            (B, P, c_lat, block, h_lat, w_lat),
            seed=self.seed,
            device=self._device,
            dtype=latents.dtype,
        )
        noisy_action = self.generate_noise(
            (B, P, T_a, D_a),
            seed=self.seed,
            device=self._device,
            dtype=latents.dtype,
        )

        causal_scheduler = os.environ.get("MAI_CAUSAL_SCHEDULER", "unipc").lower()
        if causal_scheduler == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler_action = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
        elif causal_scheduler == "flowmatch":
            sample_scheduler = FlowMatchScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                shift=self.sigma_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
            sample_scheduler_action = FlowMatchScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                shift=self.sigma_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
        else:
            raise ValueError(
                "MAI_CAUSAL_SCHEDULER must be 'unipc' or 'flowmatch', "
                f"got {causal_scheduler!r}"
            )
        num_inference_steps = int(
            os.environ.get("MAI_NUM_INFERENCE_STEPS", self.num_inference_steps)
        )
        if causal_scheduler == "unipc":
            sample_scheduler.set_timesteps(
                num_inference_steps, device=noisy_video.device, shift=self.sigma_shift
            )
            sample_scheduler_action.set_timesteps(
                num_inference_steps, device=noisy_action.device, shift=self.sigma_shift
            )
        else:
            sample_scheduler.set_timesteps(num_inference_steps, training=False)
            sample_scheduler_action.set_timesteps(num_inference_steps, training=False)
        self._mai_num_inference_steps = num_inference_steps
        self._mai_causal_scheduler = causal_scheduler

        if self.config.decouple_inference_noise:
            video_final_noise = self.config.video_inference_final_noise
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = (
                sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max
                + video_final_noise
            )
            sample_scheduler.timesteps = (
                sample_scheduler.sigmas[:-1] * 1000
            ).to(torch.int64)

        prev_predictions = []
        self.skip_countdown = 0
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self._device).type
        ):
            for index, current_timestep in enumerate(sample_scheduler.timesteps):
                action_timestep = sample_scheduler_action.timesteps[index]
                video_timestep = sample_scheduler.timesteps[index]
                timestep = (
                    torch.ones([B, block], device=latents.device, dtype=torch.int64)
                    * video_timestep
                )
                timestep_action = (
                    torch.ones([B, T_a], device=latents.device, dtype=torch.int64)
                    * action_timestep
                )
                should_run_model = self.should_run_model(
                    index, current_timestep, prev_predictions
                )
                if should_run_model:
                    predictions = self._run_multi_agent_diffusion_steps(
                        noisy_input=noisy_video,
                        timestep=timestep,
                        action=noisy_action,
                        timestep_action=timestep_action,
                        state=state_features,
                        embodiment_id=embodiment_id,
                        context=prompt_embs,
                        seq_len=seq_len,
                        y=_slice_latent_frames(self.ys, self.current_start_frame, block),
                        clip_feature=self.clip_feas,
                        kv_caches=kv_caches,
                        crossattn_caches=crossattn_caches,
                        kv_cache_metadata=dict(
                            start_frame=self.current_start_frame,
                            update_kv_cache=False,
                        ),
                        clean_x=_slice_latent_frames(
                            clean_latents, self.current_start_frame, block
                        ),
                        global_video=_slice_latent_frames(
                            global_latents, self.current_start_frame, block
                        ),
                    )
                    flow_pred_cond, flow_pred_cond_action = predictions[0]
                    if len(predictions) > 1:
                        flow_pred_uncond, _ = predictions[1]
                        flow_pred = flow_pred_uncond + self.cfg_scale * (
                            flow_pred_cond - flow_pred_uncond
                        )
                    else:
                        flow_pred = flow_pred_cond
                    prev_predictions.append(
                        (current_timestep, flow_pred, flow_pred_cond_action)
                    )
                    if len(prev_predictions) > 2:
                        prev_predictions.pop(0)
                else:
                    assert prev_predictions, (
                        "prev_predictions must be set when skipping"
                    )
                    _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

                if flow_pred.shape != noisy_video.shape:
                    noisy_video = noisy_video[
                        ..., : flow_pred.shape[-2], : flow_pred.shape[-1]
                    ]
                if causal_scheduler == "unipc":
                    noisy_video = sample_scheduler.step(
                        model_output=flow_pred,
                        timestep=video_timestep,
                        sample=noisy_video,
                        step_index=index,
                        return_dict=False,
                    )[0]
                    noisy_action = sample_scheduler_action.step(
                        model_output=flow_pred_cond_action,
                        timestep=action_timestep,
                        sample=noisy_action,
                        step_index=index,
                        return_dict=False,
                    )[0]
                else:
                    noisy_video = sample_scheduler.step(
                        model_output=flow_pred,
                        timestep=video_timestep,
                        sample=noisy_video,
                        to_final=(index == self._mai_num_inference_steps - 1),
                    )
                    noisy_action = sample_scheduler_action.step(
                        model_output=flow_pred_cond_action,
                        timestep=action_timestep,
                        sample=noisy_action,
                        to_final=(index == self._mai_num_inference_steps - 1),
                    )

        video_output = noisy_video
        if self.current_start_frame == 1:
            first_frame = current_image[
                ..., : video_output.shape[-2], : video_output.shape[-1]
            ]
            video_output = torch.cat([first_frame, video_output], dim=3)
        self.current_start_frame += block

        self._last_video_pred = video_output.detach()
        self._mai_anchor_i2v_first_frame = self.current_start_frame <= (1 + block)
        return BatchFeature(data={"action_pred": noisy_action})

    def _get_action_multi_agent(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        num_agents: int,
    ) -> BatchFeature:
        """Multi-agent inference (PR 6, minimal): joint flow-matching
        denoising rollout that produces ``action_pred[B, P, T_a, D_a]``.

        Mirrors the input prep of :meth:`_forward_multi_agent` (text +
        VAE encode + reshape) and then loops the
        :class:`FlowUniPCMultistepScheduler` over
        ``self.num_inference_steps`` to fully denoise both video and
        action streams jointly. CFG / KV cache / decoupled inference
        are intentionally omitted to keep this path correct first;
        speed optimisations land later.
        """
        self.set_frozen_modules_to_eval_mode()
        data = action_input

        embodiment_id = action_input.embodiment_id
        state_features = action_input.state             # [B, P, T_s, D_s]
        actions = action_input.action                   # [B, P, T_a, D_a]
        assert actions.dim() == 4 and actions.shape[1] == num_agents
        B, P = actions.shape[0], actions.shape[1]
        T_a, D_a = actions.shape[2], actions.shape[3]

        videos = data["images"]                         # [B, P, T, H, W, C]
        assert videos.dim() == 6 and videos.shape[1] == P
        videos = rearrange(videos, "b p t h w c -> b p c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, p, c, t, h, w = videos.shape
            videos = videos.permute(0, 1, 3, 2, 4, 5)
            videos = videos.reshape(b * p * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, p, t, c, h, w).permute(0, 1, 3, 2, 4, 5)
            assert videos.min() >= -1.0 and videos.max() <= 1.0
            videos = videos.to(dtype=self.dtype)

        prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])

        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, p, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * p * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, p, c, t, target_h, target_w)

        # VAE encode per agent.
        b, p, c, t, h, w = videos.shape
        videos_bp = videos.reshape(b * p, c, t, h, w)
        latents_bp = self.encode_video(
            videos_bp,
            self.tiled,
            (self.tile_size_height, self.tile_size_width),
            (self.tile_stride_height, self.tile_stride_width),
        )
        _, c_lat, F_lat, h_lat, w_lat = latents_bp.shape
        # [B, P, C_lat, F_lat, H_lat, W_lat] -- model orientation matches
        # the training call to ``self.model`` after the ``.transpose(2, 3)``
        # there, so we just stay in this layout throughout the rollout.
        latents = latents_bp.reshape(b, p, c_lat, F_lat, h_lat, w_lat).to(self._device)
        prompt_embs = prompt_embs.to(self._device)
        # Shared-global inference feeds current-repeat camera streams, so
        # condition on frame 0 exactly as training does. Legacy rolling-
        # history inference still conditions on the latest frame.
        if isinstance(data, dict):
            video_global_raw = data.get("video_global", None)
        else:
            video_global_raw = getattr(data, "video_global", None)
        condition_frame_index = 0 if video_global_raw is not None else -1
        clip_features, ys, clean_latents = self._prepare_multi_agent_i2v_conditioning(
            videos=videos,
            latents=latents,
            condition_frame_index=condition_frame_index,
        )
        if ys is not None:
            ys = ys.to(dtype=latents.dtype)
        if clean_latents is not None:
            clean_latents = clean_latents.to(dtype=latents.dtype)
        self._last_clean_video_cond = (
            clean_latents.detach() if clean_latents is not None else None
        )
        self._last_y_video_cond = ys.detach() if ys is not None else None

        H_g = h_lat // 2
        W_g = w_lat // 2
        seq_len = P * F_lat * H_g * W_g

        # Use the *training* scheduler's Euler-style step instead of the
        # UniPC multistep solver: UniPC's torch.compile cache + per-step
        # model_outputs history was misbehaving across the dual (video
        # 5D / action 3D) streams; a plain flow-matching Euler update
        # `sample += pred * (sigma_next - sigma_curr)` matches the
        # training scheduler's convention exactly and has no internal
        # state to corrupt.
        sample_scheduler = FlowMatchScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=self.sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        sample_scheduler_action = FlowMatchScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=self.sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        # Match the single-agent default (16 steps). The earlier 50-step
        # override gave fractionally better offline action MAE but tripled
        # wall-time per ``get_action`` call (~50s vs ~16s on H100), which
        # makes closed-loop rollout (~37 infers / episode at replan=8 /
        # max_steps=300) too slow to fit a 2h server slot. Override via
        # ``MAI_NUM_INFERENCE_STEPS`` env var when accuracy > speed.
        import os as _os
        num_inference_steps = int(
            _os.environ.get("MAI_NUM_INFERENCE_STEPS", self.num_inference_steps)
        )
        sample_scheduler.set_timesteps(num_inference_steps, training=False)
        sample_scheduler_action.set_timesteps(num_inference_steps, training=False)
        self._mai_num_inference_steps = num_inference_steps

        # Shared-global stream (PR 23, inference): when ``video_global``
        # is on the input, VAE-encode it once outside the denoising loop
        # and feed it to every denoising step as clean conditioning. The
        # latents are reused across all 16 steps so the per-step cost is
        # unchanged. The denoising loop still only updates wrist + action.
        if video_global_raw is not None:
            global_latents = self._encode_global_video(video_global_raw).to(dtype=latents.dtype)
        else:
            global_latents = None

        # Keep the I2V condition frame fixed by default in multi-agent
        # inference. Without this, the diagnostic predicted video starts
        # from pure noise even though the conditioning VAE path is valid.
        anchor_i2v_first_frame = (
            _os.environ.get("MAI_ANCHOR_I2V_FIRST_FRAME", "1").lower()
            in ("1", "true", "yes", "on")
        )
        if anchor_i2v_first_frame and clean_latents is not None:
            anchor_video_latent = clean_latents[:, :, :, :1].to(dtype=latents.dtype)
        else:
            anchor_video_latent = None
        self._mai_anchor_i2v_first_frame = anchor_video_latent is not None

        def _anchor_i2v_sample(sample: torch.Tensor) -> torch.Tensor:
            if anchor_video_latent is None:
                return sample
            h = min(sample.shape[-2], anchor_video_latent.shape[-2])
            w = min(sample.shape[-1], anchor_video_latent.shape[-1])
            sample[:, :, :, :1, :h, :w] = anchor_video_latent[
                :, :, :, :, :h, :w
            ].to(device=sample.device, dtype=sample.dtype)
            return sample

        noisy_video = torch.randn_like(latents)
        noisy_video = _anchor_i2v_sample(noisy_video)
        noisy_action = torch.randn(
            B, P, T_a, D_a, device=self._device, dtype=latents.dtype
        )

        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self._device).type
        ):
            for index, _ in enumerate(sample_scheduler.timesteps):
                video_timestep = sample_scheduler.timesteps[index]
                action_timestep = sample_scheduler_action.timesteps[index]

                timestep = torch.ones(
                    [B, F_lat], device=self._device, dtype=torch.int64
                ) * video_timestep
                timestep_action = torch.ones(
                    [B, T_a], device=self._device, dtype=torch.int64
                ) * action_timestep

                video_noise_pred, action_noise_pred = self.model(
                    noisy_video,
                    timestep=timestep,
                    context=prompt_embs,
                    seq_len=seq_len,
                    state=state_features,
                    embodiment_id=embodiment_id,
                    action=noisy_action,
                    timestep_action=timestep_action,
                    clip_feature=clip_features,
                    y=ys,
                    clean_x=clean_latents,
                    global_video=global_latents,
                )

                # Euler step for video. Spatial truncation in training
                # (line 882) can also bite at inference if patch_embedding
                # drops an odd pixel — match shapes by cropping the sample.
                if video_noise_pred.shape != noisy_video.shape:
                    noisy_video = noisy_video[
                        ..., : video_noise_pred.shape[-2], : video_noise_pred.shape[-1]
                    ]
                noisy_video = sample_scheduler.step(
                    model_output=video_noise_pred,
                    timestep=video_timestep,
                    sample=noisy_video,
                    to_final=(index == self._mai_num_inference_steps - 1),
                )
                noisy_video = _anchor_i2v_sample(noisy_video)

                # Euler step for action.
                noisy_action = sample_scheduler_action.step(
                    model_output=action_noise_pred,
                    timestep=action_timestep,
                    sample=noisy_action,
                    to_final=(index == self._mai_num_inference_steps - 1),
                )

        # Stash final denoised video latents so callers (e.g. the bimanual
        # policy server's --save-video-pred path) can VAE-decode them
        # without re-running the rollout. Shape: [B, P, C_lat, F_lat, H_lat, W_lat].
        self._last_video_pred = noisy_video.detach()
        return BatchFeature(data={"action_pred": noisy_action})

    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        num_action_samples: int = 1,
        inference_batch_size: int = 32,
    ) -> BatchFeature:
        # PR 6 (multi-agent inference): route bimanual batches through the
        # joint denoising rollout that returns ``action_pred[B, P, T_a, D_a]``.
        # Single-agent batches keep the base behaviour (delegate to forward).
        num_agents = self._detect_multi_agent(action_input)
        if num_agents is not None and num_agents > 1:
            use_causal = (
                os.environ.get("MAI_USE_CAUSAL_INFERENCE", "1").lower()
                in ("1", "true", "yes", "on")
            )
            if use_causal:
                return self._get_action_multi_agent_causal(
                    backbone_output, action_input, num_agents
                )
            return self._get_action_multi_agent(
                backbone_output, action_input, num_agents
            )
        return self.forward(backbone_output, action_input)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Multi-agent dispatch: if BimanualDreamTransform stacked a P axis
        # onto state / action / images, route to the multi-agent branch.
        num_agents = self._detect_multi_agent(action_input)
        if num_agents is not None and num_agents > 1:
            return self._forward_multi_agent(backbone_output, action_input, num_agents)

        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        data = action_input
        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id
        # print("embodiment_id", embodiment_id)
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask

        state_features = action_input.state

        actions = action_input.action
        # assert the values of action is in between -1 and 1
        if actions.numel() > 0:
            assert actions.min() >= -1.0 and actions.max() <= 1.0, "actions must be in [-1,1] range"
        videos = data["images"]

        videos = rearrange(videos, "b t h w c -> b c t h w")
        print("videos", videos.shape)
        

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)
        
        # shape of B * max_length * dim
        prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])

        # Wan 5B: resize to target resolution so latent tokens/frame matches DiT. Use config target when set
        # (e.g. 160x320 so latent is 10x20 with VAE38 16x → even H,W, no crop in dynamics loss); else 176x320.
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        latents = self.encode_video(videos, self.tiled, (self.tile_size_height, self.tile_size_width), (self.tile_stride_height, self.tile_stride_width))

        # print("latents shape", latents.shape, self.dtype)
        _, _, num_frames, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)

        clip_feas, ys, _ = self.encode_image(image, num_frames, height, width)

        latents = latents.to(self._device)
        clip_feas = clip_feas.to(self._device)
        ys = ys.to(self._device)
        prompt_embs = prompt_embs.to(self._device)
       
        # Loss
        noise = torch.randn_like(latents)

        # specific to autoregressive 
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)
        
        # ============ VIDEO TIMESTEP SAMPLING ============
        if self.config.decouple_video_action_noise:
            # Decoupled mode: sample video from Beta distribution biased towards HIGH noise
            video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "DECOUPLED"
        elif self.config.use_high_noise_emphasis:
            # High noise emphasis mode (coupled): BOTH video and action use Beta distribution
            noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "HIGH_NOISE_EMPHASIS"
        else:
            # Original: uniform sampling over full range
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]))
            noise_mode = "STANDARD"
        
        timestep_id_block = timestep_id[:, 1:].reshape(
                    timestep_id.shape[0], -1, self.num_frame_per_block)
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]
        
        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            assert actions.shape[1] / (noise.shape[1]-1) == (self.model.num_action_per_block // self.num_frame_per_block), f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            assert (noise.shape[1]-1) / state_features.shape[1] == (self.num_frame_per_block // self.model.num_state_per_block), f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            
            # ============ ACTION TIMESTEP SAMPLING ============
            if self.config.decouple_video_action_noise:
                # Decoupled: sample action timestep independently with full range
                timestep_action_id = torch.randint(
                    0, 
                    self.scheduler.num_train_timesteps, 
                    (actions.shape[0], actions.shape[1])
                )
                action_mode = "INDEPENDENT"
            else:
                # Original coupled: action timestep derived from video timestep
                timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1))
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
                action_mode = "COUPLED"
            
            # Log noise mode once
            if not self._noise_logged:
                video_mean = timestep_id.float().mean().item()
                action_mean = timestep_action_id.float().mean().item()
                if noise_mode == "DECOUPLED":
                    print(f"[NOISE] Mode={noise_mode} | Video: Beta({self.config.video_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode} Uniform mean_t={action_mean:.0f}")
                elif noise_mode == "HIGH_NOISE_EMPHASIS":
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Beta({self.config.high_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode}")
                else:
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Uniform mean_t={video_mean:.0f} | Action: {action_mode}")
                self._noise_logged = True
        else:
            noise_action = None
            timestep_action_id = None
            
        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        _, num_frames, num_channels, height, width = noise.shape
        # DiT patch_embedding uses stride (1,2,2), so sequence length is num_frames * (H//2) * (W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        seq_len = num_frames * tokens_per_frame

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)
        
        if actions.numel() > 0:
            timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = self.scheduler.training_target(actions, noise_action, timestep_action)
        else:
            timestep_action = None
            noisy_actions = None
            training_target_action = None

        # Compute loss
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            if actions.numel() > 0:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    action=noisy_actions, timestep_action=timestep_action, 
                    clean_x=latents.transpose(1, 2),
                )
            else:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, timestep_action=timestep_action, 
                    clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    clean_x=latents.transpose(1, 2),
                )

            # Per-sample dynamics loss
            # DiT patch_embedding uses stride (1,2,2), so output spatial size can be smaller than
            # latent when H or W is odd (e.g. latent 11x20 -> model output 10x20). Crop target to match.
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                ]
            dynamics_loss_per_sample = torch.nn.functional.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction='none'
            ).mean(dim=(1,3,4))  # shape: [B, ...]

            weight_dynamics = dynamics_loss_per_sample * self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)
            weighted_dynamics_loss = weight_dynamics.mean()
            
            if actions.numel() > 0:
                action_loss_per_sample = torch.nn.functional.mse_loss(
                    action_noise_pred.float(), training_target_action.float(), reduction='none'
                ) * action_mask  # shape: [B, ...]
                has_real_view = has_real_action.view(
                    -1, *([1] * (action_loss_per_sample.ndim - 1))
                ).float()
                action_loss_per_sample = has_real_view * action_loss_per_sample
                action_loss_per_sample = self._apply_action_loss_weights(
                    action_loss_per_sample,
                    actions=actions,
                )
                weight_action = action_loss_per_sample.mean(dim=2) * self.scheduler.training_weight(
                    timestep_action.flatten(0, 1),
                ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)
                weighted_action_loss = weight_action.mean()
                gripper_clean_action_loss = torch.tensor(0.0, device=self._device)
                gripper_binary_action_loss = torch.tensor(0.0, device=self._device)
                needs_gripper_clean_pred = (
                    float(
                        getattr(
                            self.config,
                            "gripper_clean_action_loss_weight",
                            0.0,
                        )
                        or 0.0
                    )
                    != 0.0
                    or float(
                        getattr(
                            self.config,
                            "gripper_binary_action_loss_weight",
                            0.0,
                        )
                        or 0.0
                    )
                    != 0.0
                )
                if needs_gripper_clean_pred:
                    sigma_action = self._sigma_for_timestep(
                        timestep_action,
                        noisy_actions,
                    )
                    clean_action_pred = self._reconstruct_clean_sample_from_flow_target(
                        noisy_sample=noisy_actions,
                        model_output=action_noise_pred,
                        sigma=sigma_action,
                    )
                if float(
                    getattr(
                        self.config,
                        "gripper_clean_action_loss_weight",
                        0.0,
                    )
                    or 0.0
                ) != 0.0:
                    clean_action_mask = action_mask.bool() & self._gripper_clean_sigma_mask(
                        sigma_action
                    )
                    gripper_clean_action_loss = self._compute_gripper_clean_action_loss(
                        clean_action_pred=clean_action_pred,
                        actions=actions,
                        action_mask=clean_action_mask,
                        has_real_action=has_real_action,
                    )
                if float(
                    getattr(
                        self.config,
                        "gripper_binary_action_loss_weight",
                        0.0,
                    )
                    or 0.0
                ) != 0.0:
                    binary_action_mask = action_mask.bool() & self._gripper_binary_sigma_mask(
                        sigma_action
                    )
                    gripper_binary_action_loss = self._compute_gripper_binary_action_loss(
                        clean_action_pred=clean_action_pred,
                        actions=actions,
                        action_mask=binary_action_mask,
                        has_real_action=has_real_action,
                    )
                loss = (
                    weighted_dynamics_loss
                    + weighted_action_loss
                    + gripper_clean_action_loss
                    + gripper_binary_action_loss
                )
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)
                gripper_clean_action_loss = torch.tensor(0.0, device=self._device)
                gripper_binary_action_loss = torch.tensor(0.0, device=self._device)
                loss = weighted_dynamics_loss
            # loss = dynamics_loss_per_sample.mean()

        # Record log
        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
            "gripper_clean_action_loss": gripper_clean_action_loss,
            "gripper_binary_action_loss": gripper_binary_action_loss,
        }

        return BatchFeature(data=output_dict)

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
    
    def _get_caches(
        self, kv_caches_input: list[KVCacheType],
    ) -> list[KVCacheType]:
        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            assert len(kv_caches_input) == 2
            if self.ip_rank == 0:
                kv_caches = [kv_caches_input[0]]
            else:
                kv_caches = [kv_caches_input[1]]
        else:
            assert len(kv_caches_input) <= 2
            kv_caches = [kv_caches_input[0]]
            if self.cfg_scale != 1.0:
                kv_caches.append(kv_caches_input[1])
        return kv_caches

    def _prepare_text_inputs(self, data: BatchFeature) -> list[tuple[torch.Tensor, torch.Tensor]]:

        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            if self.ip_rank == 0:
                text_inputs = [(data["text"], data["text_attention_mask"])]
            else:
                text_inputs = [(data["text_negative"], data["text_attention_mask_negative"])]
        else:
            text_inputs = [(data["text"], data["text_attention_mask"])]
            if self.cfg_scale != 1.0:
                text_inputs.append((data["text_negative"], data["text_attention_mask_negative"]))
        return text_inputs

    def _get_cache_state_model(self):
        """Return the module that owns causal inference cache metadata."""
        model = self.model
        get_base_model = getattr(model, "get_base_model", None)
        if callable(get_base_model):
            try:
                return get_base_model()
            except Exception:
                pass
        return model

    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches: list[KVCacheType],
        crossattn_caches: list[KVCacheType],
        kv_cache_metadata: dict[str, bool | int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        predictions = []
        for index, prompt_emb in enumerate(context):
            kv_cache = kv_caches[index]
            crossattn_cache = crossattn_caches[index]
            if not kv_cache_metadata["update_kv_cache"] and self.trt_engine is not None:
                obs_noise_pred, action_noise_pred = self.trt_engine(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    context=prompt_emb,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                )
            else:
                obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    embodiment_id=embodiment_id,
                    context=prompt_emb,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start_frame=kv_cache_metadata["start_frame"],
                )
                if kv_cache_metadata["update_kv_cache"]:
                    for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                        kv_cache[block_index] = updated_kv_cache.clone()
            obs_noise_pred = obs_noise_pred.clone()
            if action_noise_pred is not None:
                action_noise_pred = action_noise_pred.clone()
            else:
                action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device) # dummy action noise prediction
            predictions.append((obs_noise_pred, action_noise_pred))
        return self._exchange_predictions(predictions)

    def _run_multi_agent_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        context: list[torch.Tensor],
        seq_len: int,
        y: torch.Tensor | None,
        clip_feature: torch.Tensor | None,
        kv_caches: list[KVCacheType],
        crossattn_caches: list[KVCacheType],
        kv_cache_metadata: dict[str, bool | int],
        clean_x: torch.Tensor | None = None,
        global_video: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run multi-agent cached inference without mixing CFG cache metadata.

        ``CausalWanModel._forward_inference_multi_agent`` stores its
        cached token-agent ids on the model instance. CFG calls the model
        twice (cond/uncond) with separate K/V caches, so the token-agent id
        tracker must also be kept separately per branch.
        """
        cache_state_model = self._get_cache_state_model()
        saved_model_cached_ids = getattr(
            cache_state_model, "_cached_token_agent_id", None
        )
        branch_cached_ids = [
            self._ma_cached_token_agent_id,
            self._ma_cached_token_agent_id_neg,
        ]
        if self.ip_size > 1:
            branch_indices = [0 if self.ip_rank == 0 else 1]
        else:
            branch_indices = list(range(len(context)))

        predictions = []
        update_kv_cache = bool(kv_cache_metadata["update_kv_cache"])
        start_frame = int(kv_cache_metadata["start_frame"])
        try:
            for local_index, prompt_emb in enumerate(context):
                branch_index = branch_indices[local_index]
                kv_cache = kv_caches[local_index]
                crossattn_cache = crossattn_caches[local_index]
                if hasattr(cache_state_model, "_cached_token_agent_id"):
                    cache_state_model._cached_token_agent_id = (
                        None if start_frame == 0 else branch_cached_ids[branch_index]
                    )
                obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    embodiment_id=embodiment_id,
                    context=prompt_emb,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start_frame=start_frame,
                    clean_x=clean_x,
                    global_video=global_video,
                )
                if update_kv_cache:
                    for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                        kv_cache[block_index] = updated_kv_cache.clone()
                    new_cached_ids = getattr(
                        cache_state_model, "_cached_token_agent_id", None
                    )
                    branch_cached_ids[branch_index] = (
                        new_cached_ids.detach().cpu().clone()
                        if new_cached_ids is not None
                        else None
                    )
                elif hasattr(cache_state_model, "_cached_token_agent_id"):
                    cache_state_model._cached_token_agent_id = (
                        branch_cached_ids[branch_index]
                    )

                obs_noise_pred = obs_noise_pred.clone()
                if action_noise_pred is not None:
                    action_noise_pred = action_noise_pred.clone()
                else:
                    action_noise_pred = torch.tensor(
                        0.0, device=obs_noise_pred.device
                    )
                predictions.append((obs_noise_pred, action_noise_pred))
        finally:
            if hasattr(cache_state_model, "_cached_token_agent_id"):
                cache_state_model._cached_token_agent_id = saved_model_cached_ids

        if update_kv_cache:
            self._ma_cached_token_agent_id = branch_cached_ids[0]
            self._ma_cached_token_agent_id_neg = branch_cached_ids[1]
        return self._exchange_predictions(predictions)

    def _exchange_predictions(
        self,
        predictions: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.ip_size == 1:
            return predictions

        assert len(predictions) == 1
        my_predictions = list(predictions[0])

        other_predictions = [torch.empty_like(pred) for pred in my_predictions]

        send_ops = [
            dist.P2POp(op=dist.isend, tensor=pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for pred in my_predictions
        ]
        recv_ops = [
            dist.P2POp(op=dist.irecv, tensor=other_pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for other_pred in other_predictions
        ]
        ops = send_ops + recv_ops

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        output_predictions: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in range(self.ip_size)]
        output_predictions[self.ip_rank] = tuple(my_predictions)
        output_predictions[(self.ip_rank + 1) % self.ip_size] = tuple(other_predictions)
        assert all(isinstance(pred, tuple) for pred in output_predictions)
        return cast(list[tuple[torch.Tensor, torch.Tensor]], output_predictions)
    
    def should_run_model(self, index, current_timestep, prev_predictions):

        if not self.dynamic_cache_schedule:
            return self.dit_step_mask[index]

        # Always run first 2 steps to establish history
        if len(prev_predictions) < 2:
            return True

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        elif self.skip_countdown == 1:
            self.skip_countdown = 0 
            return True

        v_last = prev_predictions[-1][1].flatten(1).float()
        v_prev = prev_predictions[-2][1].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        thresholds = [0.95, 0.93]
        countdowns = [4, 2]

        for threshold, countdown in zip(thresholds, countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False

        return True

    def lazy_joint_video_action(self, backbone_output: BatchFeature, action_input: BatchFeature, latent_video: torch.Tensor | None = None) -> BatchFeature:
        start_time = time.perf_counter()

        # Tracking time taken on GPU for various operations.
        start_text_encoder_event = torch.cuda.Event(enable_timing=True)
        end_text_encoder_event = torch.cuda.Event(enable_timing=True)
        start_image_encoder_event = torch.cuda.Event(enable_timing=True)
        end_image_encoder_event = torch.cuda.Event(enable_timing=True)
        start_vae_event = torch.cuda.Event(enable_timing=True)
        end_vae_event = torch.cuda.Event(enable_timing=True)
        start_kv_event = torch.cuda.Event(enable_timing=True)
        end_kv_event = torch.cuda.Event(enable_timing=True)
        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]

        self.set_frozen_modules_to_eval_mode()
        data = action_input 
        
        videos = data["images"]

        embodiment_id = action_input.embodiment_id
        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=self.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        state_features = state_features.to(dtype=torch.bfloat16)
        videos = videos.to(dtype=torch.bfloat16)

        # Wan 5B: same as training — resize to target resolution so latent matches DiT
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        if self.language is None:
            print("language is None, reset current_start_frame to 0")
            self.language = data["text"]
            self.current_start_frame = 0
        elif not torch.equal(self.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            self.current_start_frame = 0
            self.language = data["text"]
        elif videos.shape[2] == 1:
            print("videos.shape[2] == 1, reset current_start_frame to 0")
            self.current_start_frame = 0
        elif self.current_start_frame >= self.model.local_attn_size:
            print("current_start_frame >= local_attn_size, reset current_start_frame to 0")
            self.current_start_frame = 0

        if self.ip_rank == 0:
            print("videos shape", videos.shape, self.num_frames)

        start_text_encoder_event.record()

        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [self.encode_prompt(text, attention_mask) for text, attention_mask in text_inputs]

        end_text_encoder_event.record()
        
        start_image_encoder_event.record()

        _, _, num_frames, height, width = videos.shape
        if videos.shape[2] == 4 or videos.shape[2] == 9:
            # special case for real-world eval where language is updated
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if self.current_start_frame == 0:
            clip_feas, ys, image = self.encode_image(image, self.num_frames, height, width)
            self.clip_feas = clip_feas.to(dtype=image.dtype)
            self.ys = ys.to(dtype=image.dtype)
        
        assert self.clip_feas is not None and self.ys is not None, "clip_feas and ys must be set"

        end_image_encoder_event.record()

        start_vae_event.record()

        if latent_video is not None and self.current_start_frame != 0:
            image = latent_video
            if self.ip_rank == 0:
                print("image shape@@", image.shape)
        elif self.current_start_frame != 0:
            # this is for real world execution
            if (videos.shape[2] - 1) // 4 == self.num_frame_per_block:
                print("no further action")
            elif videos.shape[2] // 4 != self.num_frame_per_block:
                # Repeating videos along dim 2.
                repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
            
                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)
            else: 
                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)
           
            image = self.vae.encode(
                videos,
                tiled=self.tiled,
                tile_size=(self.tile_size_height, self.tile_size_width),
                tile_stride=(self.tile_stride_height, self.tile_stride_width),
            )

        end_vae_event.record()

        noise_obs = self.generate_noise((image.shape[0], image.shape[1], self.num_frame_per_block, image.shape[3], image.shape[4]), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        noise_action = self.generate_noise((image.shape[0], self.action_horizon, self.model.action_dim), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        batch_size, num_channels, num_frames, height, width = noise_obs.shape
        ######### Generate video #########
        # DiT patch_embedding uses stride (1,2,2), so tokens per frame = (H//2)*(W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        frame_seqlen = tokens_per_frame
        seq_len = num_frames * frame_seqlen

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        if self.current_start_frame == 0:
            # Reinitialize KV cache and crossattn cache for each new sequence.
            self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
                frame_seqlen=frame_seqlen,
            )
            self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
            )

        assert self.kv_cache1 is not None
        assert self.kv_cache_neg is not None
        assert self.crossattn_cache is not None
        assert self.crossattn_cache_neg is not None
        kv_caches = self._get_caches(
            [self.kv_cache1, self.kv_cache_neg],
        )
        crossattn_caches = self._get_caches(
            [self.crossattn_cache, self.crossattn_cache_neg],
        )

        start_kv_event.record()

        if self.current_start_frame == 0:
            timestep = torch.ones([batch_size, 1], device=noise_obs.device, dtype=torch.int64) * 0
            self._run_diffusion_steps(
                noisy_input=image.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=frame_seqlen,
                y=self.ys[:, :, 0:1],
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=0,
                    update_kv_cache=True,
                ),
            )
            self.current_start_frame += 1
            
        timestep = torch.ones([batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64) * 0

        if self.current_start_frame != 1:
            current_ref_latents = image[:, -self.num_frame_per_block:]
            if self.current_start_frame <= self.ys.shape[2]:
                y = self.ys[:, :, self.current_start_frame - self.num_frame_per_block : self.current_start_frame]
            else:
                y = self.ys[:, :, -self.num_frame_per_block:]
            self._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=seq_len,
                y=y,
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=self.current_start_frame - self.num_frame_per_block,
                    update_kv_cache=True,
                ),
            )

        end_kv_event.record()

        noisy_input = noise_obs
        noisy_input_action = noise_action

        # Step 3.1: Spatial denoising loop

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)

        # Decoupled inference: video sigmas end at video_final_noise instead of 0
        # This rescales the schedule so video still takes all denoising steps, 
        # but ends at a higher noise level (e.g., 1.0 → 0.9 → 0.8 instead of 1.0 → 0.5 → 0.0)
        if self.config.decouple_inference_noise:
            video_final_noise = self.config.video_inference_final_noise
            # Rescale video sigmas: map [sigma_max, 0] -> [sigma_max, video_final_noise]
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)
            if self.ip_rank == 0:
                print(f"Decoupled inference: video sigmas {sigma_max:.3f} -> {sample_scheduler.sigmas[-1].item():.3f}")

        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        prev_predictions = [] 
        self.skip_countdown = 0
        dit_compute_steps = 0
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            start_diffusion_events[index].record()

            # Get timesteps from respective schedulers
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[index]  # Already rescaled if decoupled

            # set current timestep
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [batch_size, self.action_horizon],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * action_timestep

            # check if we need to run the DIT step
            should_run_model = self.should_run_model(index, current_timestep, prev_predictions)
            if should_run_model:
                dit_compute_steps += 1
                if self.current_start_frame + self.num_frame_per_block <= self.ys.shape[2]:
                    y = self.ys[:, :, self.current_start_frame : self.current_start_frame + self.num_frame_per_block]
                else:
                    y = self.ys[:, :, -self.num_frame_per_block:]
                predictions = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=state_features,
                    embodiment_id=embodiment_id,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=self.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame,
                        update_kv_cache=False,
                    ),
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                flow_pred_uncond, flow_pred_uncond_action = predictions[1]

                flow_pred = flow_pred_uncond + self.cfg_scale * (flow_pred_cond - flow_pred_uncond)
                prev_predictions.append((current_timestep, flow_pred, flow_pred_cond_action))
                max_cache_size = 2
                if len(prev_predictions) > max_cache_size:
                    prev_predictions.pop(0)

            else:
                assert len(prev_predictions) > 0, "prev_predictions must be set when skipping"
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            end_diffusion_events[index].record()

            # Video: denoising step (uses rescaled schedule if decoupled)
            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]
            
            # Action: always fully denoises with standard schedule (1000->0)
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_cond_action,
                timestep=action_timestep,
                sample=noisy_input_action,
                step_index=index,
                return_dict=False,
            )[0]

        latents = noisy_input
        latents_action = noisy_input_action
        output = latents

        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        # Do torch.cuda.synchronize() to ensure all operations are completed before timing.
        # This isn't expected to affect inference performance since it's at the end of an inference step.
        torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time
        text_encoder_time = start_text_encoder_event.elapsed_time(end_text_encoder_event) / 1000
        image_encoder_time = start_image_encoder_event.elapsed_time(end_image_encoder_event) / 1000
        vae_time = start_vae_event.elapsed_time(end_vae_event) / 1000
        kv_creation_time = start_kv_event.elapsed_time(end_kv_event) / 1000
        diffusion_times = [s.elapsed_time(e) for s, e in zip(start_diffusion_events, end_diffusion_events)]
        diffusion_time = sum(diffusion_times) / 1000
        scheduler_time = total_time - kv_creation_time - diffusion_time - text_encoder_time - image_encoder_time - vae_time

        if self.ip_rank == 0:
            print(f"Time taken: Total {total_time:.2f} seconds, "
                  f"Text Encoder {text_encoder_time:.2f} seconds, "
                  f"Image Encoder {image_encoder_time:.2f} seconds, "
                  f"VAE {vae_time:.2f} seconds, "
                  f"KV Cache Creation {kv_creation_time:.2f} seconds, "
                  f"Diffusion {diffusion_time:.2f} seconds, "
                  f"DIT Compute Steps {dit_compute_steps} steps, "
                  f"Scheduler {scheduler_time:.2f} seconds")

        return BatchFeature(data={"action_pred": latents_action, "video_pred": output.transpose(1, 2)})
    
    def cache_predict_order1(self, current_timestep, timestep_1, f1, timestep_2, f2):
        h_curr = current_timestep - timestep_1
        h_past = timestep_1 - timestep_2

        v_prime = (f1 - f2) / h_past

        # Prediction 
        damping_factor = 0.25
        flow_pred = f1 + (v_prime * h_curr) * damping_factor
        return flow_pred

    def post_initialize(self):
        # Move models to the cuda device and set the dtype to bfloat16.
        print("Moving models to the cuda device and setting the dtype to bfloat16.")
        self.model.to(device=self._device, dtype=torch.bfloat16)
        self.text_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.image_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.vae.to(device=self._device, dtype=torch.bfloat16)
        import os
        ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"
        LOAD_TRT_ENGINE = os.getenv("LOAD_TRT_ENGINE", None)

        # Torch compile the modules. Skip _forward_blocks: Dynamo with fullgraph can fail on
        # shape variation (e.g. x [1,50,C] vs e [1,200,C]); the block aligns e to x at runtime.
        if not ENABLE_TENSORRT:
            print("Torch compiling the TextEncoder, ImageEncoder, and VAE modules (Wan _forward_blocks not compiled).")

            self.text_encoder.forward = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.text_encoder.forward)

            self.image_encoder.model.visual.forward = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.image_encoder.model.visual.forward)

            self.vae.model.encode = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.vae.model.encode)
        
        self.trt_engine = None
        if LOAD_TRT_ENGINE is not None:
            print(f"Loading TRT engine from {LOAD_TRT_ENGINE}")
            import groot.control.tensorrt_utils as trt_utils
            model_path = LOAD_TRT_ENGINE
            self.trt_engine = trt_utils.load_tensorrt_engine(model_path, model_type="ar_14B")

    def parallelize(self, device_mesh: DeviceMesh) -> None:
        ip_mesh = device_mesh["ip"]
        self.ip_rank = ip_mesh.get_local_rank()
        self.ip_size = ip_mesh.size()
        self.ip_group = ip_mesh.get_group()

        assert self.ip_size == 1 or self.ip_size == 2, "ip_size must be 1 or 2"
        assert self.ip_rank >= 0 and self.ip_rank < self.ip_size, "ip_rank must be in [0, ip_size)"

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
