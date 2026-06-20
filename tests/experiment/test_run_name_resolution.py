from omegaconf import OmegaConf

from groot.vla.utils.training_args import resolve_training_run_name


def test_resolve_training_run_name_preserves_explicit_name():
    cfg = OmegaConf.create(
        {
            "output_dir": "/workspace/outputs/robofactory_liftbarrier_shared_global_binary",
            "run_name": "dz-rf2-lb500-lossmask32d-50k-fresh-xz-20260620",
        }
    )

    assert (
        resolve_training_run_name(cfg)
        == "dz-rf2-lb500-lossmask32d-50k-fresh-xz-20260620"
    )


def test_resolve_training_run_name_falls_back_for_missing_name():
    cfg = OmegaConf.create(
        {
            "output_dir": "/workspace/outputs/robofactory_liftbarrier_shared_global_binary",
            "run_name": "???",
        }
    )

    assert (
        resolve_training_run_name(cfg)
        == "robofactory_liftbarrier_shared_global_binary"
    )
