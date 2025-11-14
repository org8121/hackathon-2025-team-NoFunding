"""Helper functions for W&B logging, metrics, and server checkpoints."""

import json
import os
import warnings
from logging import INFO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb
from flwr.common import log
from sklearn.metrics import roc_auc_score

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

PARTITION_HOSPITAL_MAP = {
    0: "A",
    1: "B",
    2: "C",
}

CHECKPOINT_DIR = Path("models/checkpoints")


def _sanitize_run_name(run_name: Optional[str]) -> str:
    sanitized = run_name or "default_run"
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sanitized)
    return sanitized or "default_run"


def _checkpoint_metadata_path(run_name: str) -> Path:
    sanitized = _sanitize_run_name(run_name)
    return CHECKPOINT_DIR / f"{sanitized}_checkpoint.json"


def _write_checkpoint_metadata(run_name: str, server_round: int, filename: Optional[str]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "run_name": run_name,
        "last_completed_round": int(server_round),
    }
    if filename is not None:
        data["checkpoint_filename"] = filename
    with open(_checkpoint_metadata_path(run_name), "w", encoding="utf-8") as meta_file:
        json.dump(data, meta_file)


def get_last_completed_round(run_name: str) -> int:
    """Return the last completed global round stored in checkpoint metadata."""
    path = _checkpoint_metadata_path(run_name)
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as meta_file:
            data = json.load(meta_file)
    except (json.JSONDecodeError, OSError):
        return 0
    try:
        return int(data.get("last_completed_round", 0))
    except (TypeError, ValueError):
        return 0


def save_training_checkpoint(arrays: Any, server_round: int, run_name: str, best_auroc: Optional[float]) -> Optional[Path]:
    """Persist the current global model and metadata for later resumption."""
    if arrays is None:
        return None
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_run_name(run_name)
    filename = f"{sanitized}_round{server_round:04d}.pt"
    checkpoint_path = CHECKPOINT_DIR / filename
    state = {
        "state_dict": arrays.to_torch_state_dict(),
        "server_round": int(server_round),
        "best_auroc": best_auroc,
        "run_name": run_name,
    }
    torch.save(state, checkpoint_path)
    _write_checkpoint_metadata(run_name, server_round, filename)
    log(INFO, f"  Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_latest_checkpoint(run_name: str) -> Optional[Dict[str, Any]]:
    """Load the latest checkpoint state for the given run, if it exists."""
    sanitized = _sanitize_run_name(run_name)
    metadata_path = _checkpoint_metadata_path(run_name)
    checkpoint_path = None
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                data = json.load(meta_file)
                filename = data.get("checkpoint_filename")
                if filename:
                    candidate = CHECKPOINT_DIR / filename
                    if candidate.exists():
                        checkpoint_path = candidate
        except (json.JSONDecodeError, OSError):
            checkpoint_path = None
    if checkpoint_path is None:
        pattern = f"{sanitized}_round*.pt"
        candidates = sorted(CHECKPOINT_DIR.glob(pattern))
        if not candidates:
            return None
        checkpoint_path = candidates[-1]
    state = torch.load(checkpoint_path, map_location="cpu")
    return {
        "checkpoint_path": checkpoint_path,
        **state,
    }


def compute_metrics(reply_metrics):
    """Compute AUROC and confusion matrix metrics."""
    probs = np.array(reply_metrics["probs"])
    labels = np.array(reply_metrics["labels"])
    auroc = roc_auc_score(labels, probs)
    cm_metrics = compute_metrics_from_confusion_matrix(
        reply_metrics["tp"], reply_metrics["tn"], reply_metrics["fp"], reply_metrics["fn"]
    )
    return {"auroc": auroc, "eval_loss": reply_metrics["eval_loss"], **cm_metrics}


def compute_aggregated_metrics(replies):
    """Compute aggregated metrics across all hospitals."""
    all_probs = [p for r in replies for p in r.content["metrics"]["probs"]]
    all_labels = [l for r in replies for l in r.content["metrics"]["labels"]]
    auroc = roc_auc_score(np.array(all_labels), np.array(all_probs))

    tp = sum(r.content["metrics"]["tp"] for r in replies)
    tn = sum(r.content["metrics"]["tn"] for r in replies)
    fp = sum(r.content["metrics"]["fp"] for r in replies)
    fn = sum(r.content["metrics"]["fn"] for r in replies)
    cm_metrics = compute_metrics_from_confusion_matrix(tp, tn, fp, fn)

    return {"auroc": auroc, **cm_metrics}


def compute_metrics_from_confusion_matrix(tp, tn, fp, fn):
    """Compute classification metrics from confusion matrix components."""
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def log_training_metrics(replies, server_round):
    """Log training metrics to W&B."""
    if wandb.run is None:
        return
    log_dict = {}
    for reply in replies:
        hospital = f"Hospital{PARTITION_HOSPITAL_MAP[reply.content['metrics']['partition-id']]}"
        log_dict[f"{hospital}/train_loss"] = reply.content["metrics"]["train_loss"]
    wandb.log(log_dict, step=server_round)


def log_eval_metrics(replies, agg_metrics, server_round, weighted_by_key, log_fn):
    """Log evaluation metrics to console and W&B."""
    log_fn("METRICS BY HOSPITAL")
    log_dict = {}

    for reply in replies:
        hospital = f"Hospital{PARTITION_HOSPITAL_MAP[reply.content['metrics']['partition-id']]}"
        metrics = compute_metrics(reply.content["metrics"])
        n = reply.content["metrics"].get(weighted_by_key, 0)

        log_fn(f"  {hospital} (n={n}):")
        for k, v in metrics.items():
            log_fn(f"    {k:12s}: {v:.4f}")
            log_dict[f"{hospital}/{k}"] = v

    log_fn("AGGREGATED METRICS:")
    for k, v in agg_metrics.items():
        log_fn(f"  {k:12s}: {v:.4f}")
        log_dict[f"Global/{k}"] = v

    if wandb.run is not None:
        wandb.log(log_dict, step=server_round)


def save_best_model(arrays, agg_metrics, server_round, run_name, best_auroc):
    """Save model to disk if its AUROC is higher than <best_auroc>.

    Models are saved to ./models/ directory (in scratch during SLURM jobs)
    with filename encoding: {run_name}_round{N}_auroc{XXXX}.pt

    After training, the best model will be automatically copied to
    /home/${USER}/models/ by submit-job.sh.

    Feel free to save models based on whatever metric you want.

    Returns updated best_auroc.
    """
    current_auroc = agg_metrics["auroc"]
    if best_auroc is None or current_auroc > best_auroc:
        log(INFO, f"✓ New best model! Round {server_round}, AUROC: {current_auroc:.4f}")

        # Create models directory (relative to working directory, in scratch during SLURM jobs)
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        # Save model with run_name, round, and AUROC encoded in filename
        auroc_str = f"{int(current_auroc * 10000):04d}"
        model_filename = f"{run_name}_round{server_round}_auroc{auroc_str}.pt"
        model_path = os.path.join(models_dir, model_filename)
        torch.save(arrays.to_torch_state_dict(), model_path)

        log(INFO, f"  Model saved to {model_path}")

        # Also log to W&B if active
        if wandb.run is not None:
            metadata = {**agg_metrics, "round": server_round, "run_name": run_name}
            artifact = wandb.Artifact(model_filename.replace('.pt', ''), type="model", metadata=metadata)
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

        return current_auroc
    else:
        log(INFO, f"  Model not saved (AUROC {current_auroc:.4f} ≤ best {best_auroc:.4f})")
        return best_auroc
    

def save_local_model(arrays, local_metric, server_round, run_name, hospital_id):
    """
    Models are saved to ./models/local_models/ directory (in scratch during SLURM jobs)
    with filename encoding: {run_name}_round{N}_auroc{XXXX}.pt

    After training, the best model will be automatically copied to
    /home/${USER}/models/ by submit-job.sh.

    Feel free to save models based on whatever metric you want.

    Returns updated best_auroc.
    """
    current_auroc = local_metric["auroc"]
    log(INFO, f"✓ New local model! Round {server_round}, AUROC: {current_auroc:.4f}")

    # Create models directory (relative to working directory, in scratch during SLURM jobs)
    models_dir = "models/local_models"
    os.makedirs(models_dir, exist_ok=True)

    # Save model with run_name, round, and AUROC encoded in filename
    auroc_str = f"{int(current_auroc * 10000):04d}"
    model_filename = f"{run_name}_{hospital_id}_round{server_round}_auroc{auroc_str}.pt"
    model_path = os.path.join(models_dir, model_filename)
    torch.save(arrays.to_torch_state_dict(), model_path)

    log(INFO, f"  Model saved to {model_path}")

    # Also log to W&B if active
    if wandb.run is not None:
        metadata = {**local_metric, "round": server_round, "run_name": run_name}
        artifact = wandb.Artifact(model_filename.replace('.pt', ''), type="model", metadata=metadata)
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    return current_auroc


# For reference: These are all labels in the original dataset.
# In the challenge we only consider a binary classification: (no) finding.
LABELS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
