from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CFG = REPO_ROOT / "groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml"
RELATIVE_CFG = REPO_ROOT / "groot/vla/configs/data/dreamzero/robotwin_franka_bimanual_relative.yaml"
SHARED_GLOBAL_CFG = REPO_ROOT / "groot/vla/configs/data/dreamzero/robotwin_franka_bimanual_shared_global.yaml"
ROBOFACTORY_SHARED_GLOBAL_CFG = REPO_ROOT / "groot/vla/configs/data/dreamzero/robofactory_bimanual_shared_global.yaml"
CONVERTER = REPO_ROOT / "scripts/data/robotwin_to_lerobot_v2.py"


def _load_yaml(path: Path):
    with path.open() as f:
        return yaml.safe_load(f)


def test_robotwin_has_dedicated_embodiment_namespace():
    base = _load_yaml(BASE_CFG)

    assert "robotwin" in base["modality_configs"]
    assert "robotwin" in base["transforms"]
    assert "robotwin" in base["metadata_versions"]
    assert base["fps"]["robotwin"] == 20

    assert "modality_config_robotwin" in base
    assert "transform_robotwin" in base
    assert base["modality_config_robotwin"] is not base["modality_config_robofactory"]
    assert base["transform_robotwin"] is not base["transform_robofactory"]


def test_robotwin_training_configs_do_not_mount_data_as_robofactory():
    for path in (RELATIVE_CFG, SHARED_GLOBAL_CFG):
        cfg = _load_yaml(path)
        dataset_path = cfg["train_dataset"]["mixture_spec"][0]["dataset_path"]
        assert list(dataset_path) == ["robotwin"]
        assert "robofactory" not in dataset_path


def test_shared_global_configs_keep_gripper_absolute_with_eight_dim_arms():
    for path in (SHARED_GLOBAL_CFG, ROBOFACTORY_SHARED_GLOBAL_CFG):
        cfg = _load_yaml(path)

        assert cfg["relative_action"] is True
        assert cfg["relative_action_per_horizon"] is False
        assert cfg["relative_action_keys"] == [
            "panda0_joint_pos",
            "panda1_joint_pos",
        ]
        assert all("gripper" not in key for key in cfg["relative_action_keys"])
        assert cfg["agent_state_dims"] == [[0, 8], [8, 16]]
        assert cfg["agent_action_dims"] == [[0, 8], [8, 16]]


def test_robotwin_converter_writes_robotwin_embodiment_tag():
    text = CONVERTER.read_text()

    assert '"embodiment_tag": "robotwin"' in text
    assert '"embodiment_tag": "robofactory"' not in text
    assert "dedicated robotwin embodiment tag/config" in text
