from logging import INFO
import os

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from cold_start_hackathon.task import Net, load_data, test
from sklearn.metrics import roc_auc_score

from cold_start_hackathon.task import Net
from cold_start_hackathon.util import (
    compute_aggregated_metrics,
    log_training_metrics,
    log_eval_metrics,
    save_best_model,
    save_training_checkpoint,
    load_latest_checkpoint,
)

# ============================================================================
# W&B Configuration - Fill in your credentials or set via environment variables
# ============================================================================
# Option 1: Set these constants directly (e.g., WANDB_API_KEY = "your_api_key_here")
# Option 2: Leave as None and set environment variables (WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT)
# If all W&B config is None/unset, W&B logging will be disabled
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)  # Your W&B API key
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", None)  # Your W&B project name
# ============================================================================

app = ServerApp()

datasets_to_test = [
        ("Hospital A", "HospitalA", "eval"),
        ("Hospital B", "HospitalB", "eval"),
        ("Hospital C", "HospitalC", "eval"),
        ("Test D (OOD)", "Test", "test_D"),
    ]


def evaluate_split(model, dataset_name, split_name, device):
    """Evaluate on any dataset split and return predictions."""
    loader = load_data(dataset_name, split_name, batch_size=32)
    _, _, _, _, _, probs, labels = test(model, loader, device)
    return probs, labels


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Log GPU device
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log(INFO, f"Device: {device}")

    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config["local-epochs"]

    # Get run name from environment variable (set by submit-job.sh). Feel free to change this.
    run_name = os.environ.get("JOB_NAME", "your_custom_run_name")

    # Initialize W&B if credentials are provided
    use_wandb = WANDB_API_KEY and WANDB_PROJECT
    if use_wandb:
        wandb.login(key=WANDB_API_KEY)
        log(INFO, "Wandb login successful")
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "num_rounds": num_rounds,
                "learning_rate": lr,
                "local_epochs": local_epochs,
            }
        )
        log(INFO, "Wandb initialized with run_id: %s", wandb.run.id)
    else:
        log(INFO, "W&B disabled (credentials not provided). Set WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT to enable.")


    checkpoint_state = load_latest_checkpoint(run_name)
    best_auroc = None
    start_round = 0
    if checkpoint_state is not None:
        arrays = ArrayRecord(checkpoint_state["state_dict"])
        best_auroc = checkpoint_state.get("best_auroc")
        start_round = int(checkpoint_state.get("server_round", 0))
        ckpt_path = checkpoint_state.get("checkpoint_path")
        log(INFO, f"Resuming from checkpoint round {start_round} ({ckpt_path})")
    else:
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())

    remaining_rounds = max(0, num_rounds - start_round)
    if remaining_rounds == 0:
        log(INFO, "No remaining rounds to run; checkpoint already reached configured total.")
        if use_wandb:
            wandb.finish()
            log(INFO, "Wandb run finished")
        return

    #Evaluate starting global model here
    '''for display_name, dataset_name, split_name in datasets_to_test:
        try:
            probs, labels = evaluate_split(global_model, dataset_name, split_name, device)
            n = len(labels)

            # Compute per-dataset AUROC for display
            auroc = roc_auc_score(labels, probs)
            print(f"  {display_name:<15} AUROC: {auroc:.4f} (n={n})")

        except FileNotFoundError:
            # Test dataset doesn't exist for participants - skip silently
            pass
'''
    strategy = HackathonFedAvg(
        fraction_train=1,
        run_name=run_name,
        start_round=start_round,
        best_auroc=best_auroc,
        fraction_evaluate=1.0,
        min_available_nodes=3,
        min_train_nodes=3,
        min_evaluate_nodes=3,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=remaining_rounds,
    )

    log(INFO, "Training complete")
    if use_wandb:
        wandb.finish()
        log(INFO, "Wandb run finished")


class HackathonFedAvg(FedAvg):
    """FedAvg strategy that logs metrics, saves models, and handles checkpoints."""

    def __init__(self, *args, run_name=None, start_round: int = 0, best_auroc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_auroc = best_auroc
        self._run_name = run_name or "your_run"
        self._round_offset = int(start_round)
        self._arrays = None

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, replies)
        self._arrays = arrays
        global_round = server_round + self._round_offset
        log_training_metrics(replies, global_round)
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        global_round = server_round + self._round_offset
        agg_metrics = compute_aggregated_metrics(replies)
        log_eval_metrics(replies, agg_metrics, global_round, self.weighted_by_key, lambda msg: log(INFO, msg))
        self._best_auroc = save_best_model(self._arrays, agg_metrics, global_round, self._run_name, self._best_auroc)
        save_training_checkpoint(self._arrays, global_round, self._run_name, self._best_auroc)
        return agg_metrics
