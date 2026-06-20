from omegaconf import DictConfig, OmegaConf


def resolve_training_run_name(training_args_cfg: DictConfig) -> str:
    output_dir = str(training_args_cfg.output_dir).rstrip("/")
    fallback_run_name = output_dir.split("/")[-1]
    if OmegaConf.is_missing(training_args_cfg, "run_name"):
        return fallback_run_name
    run_name = training_args_cfg.get("run_name")
    if run_name is None or str(run_name).strip() in ("", "???"):
        return fallback_run_name
    return str(run_name)
