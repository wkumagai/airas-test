import os
import subprocess
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> Any:
    assert cfg.mode in ("trial", "full"), "mode must be trial or full"

    # mode-based auto config
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.max_steps = 2
        cfg.training.eval_every_n_steps = 1
        cfg.training.log_every_n_steps = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"

    # Ensure results_dir exists
    os.makedirs(str(cfg.results_dir), exist_ok=True)

    # Launch train.py as subprocess with Hydra overrides
    run_id = str(cfg.run.run_id)
    cmd = [
        "python",
        "-u",
        "-m",
        "src.train",
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"

    print("Running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env, check=True)
    return p.returncode


if __name__ == "__main__":
    main()
