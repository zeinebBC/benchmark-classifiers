"""MLflow run persistence — save/restore run_id alongside checkpoints.

When a training run is interrupted and resumed with --resume, this module
lets the trainer continue logging into the same MLflow run so that the
training and validation curves stay continuous.

Usage in a training script:
    from utils.mlflow_utils import load_run_id, save_run_info

    existing_run_id = load_run_id(args.resume)
    logger = MLFlowLogger(..., run_id=existing_run_id)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    save_run_info(run_dir, run_name, logger.run_id)
"""
from __future__ import annotations

import json
from pathlib import Path

_INFO_FILE = "run_info.json"


def save_run_info(run_dir: str | Path, run_name: str, run_id: str) -> None:
    """Persist run metadata next to checkpoints for future resumes."""
    path = Path(run_dir) / _INFO_FILE
    path.write_text(json.dumps({"run_name": run_name, "mlflow_run_id": run_id}, indent=2))
    print(f"[mlflow] run_info saved → {path}  (run_id={run_id})")


def load_run_id(checkpoint_path: str | Path | None) -> str | None:
    """
    Return the MLflow run_id saved alongside a checkpoint, or None.
    Looks for run_info.json in the same directory as the checkpoint file.
    """
    if checkpoint_path is None:
        return None
    info_file = Path(checkpoint_path).parent / _INFO_FILE
    if not info_file.exists():
        return None
    try:
        run_id = json.loads(info_file.read_text()).get("mlflow_run_id")
        if run_id:
            print(f"[mlflow] Resuming MLflow run_id={run_id}")
        return run_id
    except Exception as e:
        print(f"[mlflow] WARNING: could not read {info_file}: {e}")
        return None
