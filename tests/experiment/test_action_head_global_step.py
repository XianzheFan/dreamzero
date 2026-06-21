import ast
import contextlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _Timer:
    def with_label(self, _label):
        return contextlib.nullcontext()


class _Accelerator:
    def unwrap_model(self, model):
        return model.module


def _install_h5py_stub(monkeypatch):
    fake_h5py = types.SimpleNamespace(
        Dataset=type("Dataset", (), {}),
        Datatype=type("Datatype", (), {}),
        Group=type("Group", (), {}),
    )
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)


def _install_base_import_stubs(monkeypatch):
    _install_h5py_stub(monkeypatch)

    def module(name):
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    groot_pkg = module("groot")
    groot_pkg.__path__ = []
    vla_pkg = module("groot.vla")
    vla_pkg.__path__ = []
    groot_pkg.vla = vla_pkg

    common_pkg = module("groot.vla.common")
    common_pkg.__path__ = []
    common_utils = module("groot.vla.common.utils")
    common_utils.json_dump = lambda *args, **kwargs: None
    common_pkg.utils = common_utils
    vla_pkg.common = common_pkg

    data_pkg = module("groot.vla.data")
    data_pkg.__path__ = []
    dataset_pkg = module("groot.vla.data.dataset")
    dataset_pkg.__path__ = []
    lerobot_sharded = module("groot.vla.data.dataset.lerobot_sharded")
    lerobot_sharded.ShardedLeRobotMixtureDataset = type(
        "ShardedLeRobotMixtureDataset",
        (),
        {},
    )
    schema = module("groot.vla.data.schema")
    schema.EmbodimentTag = type("EmbodimentTag", (), {})
    transform = module("groot.vla.data.transform")
    transform.ComposedModalityTransform = type("ComposedModalityTransform", (), {})

    experiment_utils = module("groot.vla.experiment.utils")
    experiment_utils.compute_grad_accum_to_match_global_bs = lambda global_bs, bs: 1
    experiment_utils.dtype_from_string = lambda dtype: dtype
    experiment_utils.get_checkpoint_path = lambda output_dir: (None, True)
    experiment_utils.mprint = lambda *args, **kwargs: None
    experiment_utils.safe_save_model_for_hf_trainer = lambda *args, **kwargs: None

    utils_pkg = module("groot.vla.utils")
    utils_pkg.__path__ = []
    timer_mod = module("groot.vla.utils.timer")
    timer_mod.ContextTimer = _Timer


def _load_base_module(monkeypatch):
    _install_base_import_stubs(monkeypatch)
    path = _REPO_ROOT / "groot/vla/experiment/base.py"
    spec = importlib.util.spec_from_file_location("_test_dreamzero_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_trainer(cls, *, global_step):
    trainer = cls.__new__(cls)
    trainer.state = types.SimpleNamespace(global_step=global_step, save_steps=100)
    trainer.accelerator = _Accelerator()
    trainer.enable_profiling = False
    trainer.current_step = 0
    trainer.profiling_steps = 1
    trainer.timer = _Timer()
    trainer.global_rank = 0
    trainer.benchmark_time = False
    trainer.restart_max_seconds = 0
    trainer.start_time = 0
    trainer.micro_global_step = 0
    return trainer


def test_base_trainer_sets_unwrapped_action_head_global_step_before_forward(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    base_mod = _load_base_module(monkeypatch)
    BaseTrainer = base_mod.BaseTrainer

    action_head = types.SimpleNamespace(global_step=-1)
    engine = types.SimpleNamespace(
        module=types.SimpleNamespace(action_head=action_head),
    )
    trainer = _make_trainer(BaseTrainer, global_step=17)

    def fake_training_step(self, model, inputs):
        assert model is engine
        assert inputs == {"batch": 1}
        assert action_head.global_step == 17
        return torch.tensor(0.0)

    monkeypatch.setattr(transformers.Trainer, "training_step", fake_training_step)

    loss = BaseTrainer.training_step(trainer, engine, {"batch": 1})

    assert float(loss) == 0.0
    assert trainer.current_step == 1


def test_runtime_provenance_reads_code_marker_and_gamma_knobs(monkeypatch, tmp_path):
    base_mod = _load_base_module(monkeypatch)

    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "OSMO_CODE_COMMIT").write_text("abc123\n", encoding="utf-8")
    monkeypatch.setenv("CODE_ROOT", str(code_root))
    monkeypatch.setenv("EXPECTED_CODE_COMMIT", "abc123")
    monkeypatch.setenv("STAGE_LABEL", "droidwidth-teacher-style")
    monkeypatch.setenv("MODEL_ACTION_DIM", "32")
    monkeypatch.setenv("GLOBAL_VIDEO_ATTENTION_MODE", "bidirectional")

    provenance = base_mod.collect_runtime_provenance()

    assert provenance["osmo_code_commit"] == "abc123"
    assert provenance["expected_code_commit"] == "abc123"
    assert provenance["stage_label"] == "droidwidth-teacher-style"
    assert provenance["model_action_dim"] == "32"
    assert provenance["global_video_attention_mode"] == "bidirectional"


def test_vla_trainer_training_step_delegates_step_propagation_to_base_trainer():
    path = _REPO_ROOT / "groot/vla/experiment/experiment.py"
    tree = ast.parse(path.read_text())
    class_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VLATrainer"
    )
    training_step = next(
        node
        for node in class_def.body
        if isinstance(node, ast.FunctionDef) and node.name == "training_step"
    )

    source = ast.get_source_segment(path.read_text(), training_step)
    assert "self.model.action_head" not in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "training_step"
        for node in ast.walk(training_step)
    )
