import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from cold_start_hackathon.task import Net, load_data
from cold_start_hackathon.util import (
    PARTITION_HOSPITAL_MAP,
    compute_metrics,
    save_local_model,
    get_last_completed_round,
    LOCAL_MODELS_DIR,
)
from cold_start_hackathon.task import test as test_fn
from cold_start_hackathon.task import train as train_fn
from flwr.common import log

app = ClientApp()

RUN_NAME = os.environ.get("JOB_NAME", "your_custom_run_name")
CLIENT_ROUND_TRACKER = defaultdict(int)
GLOBAL_ROUND_OFFSET = get_last_completed_round(RUN_NAME)


def _get_config_value(config, key):
    if isinstance(config, dict):
        return config.get(key)
    try:
        return config[key]
    except Exception:
        getter = getattr(config, "get", None)
        if getter is not None:
            return getter(key)
    return None


def _get_server_round(msg: Message, partition_id: int) -> int:
    config = msg.content["config"]
    round_value = _get_config_value(config, "server_round")
    base_round = CLIENT_ROUND_TRACKER[partition_id]
    if round_value is not None:
        try:
            base_round = int(round_value)
            CLIENT_ROUND_TRACKER[partition_id] = base_round
        except (TypeError, ValueError):
            CLIENT_ROUND_TRACKER[partition_id] += 1
            base_round = CLIENT_ROUND_TRACKER[partition_id]
    else:
        CLIENT_ROUND_TRACKER[partition_id] += 1
        base_round = CLIENT_ROUND_TRACKER[partition_id]
    return GLOBAL_ROUND_OFFSET + base_round


def _personalization_start_round(run_config: dict) -> float:
    pct = run_config.get("personalization-start-percentage", 101)
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        return math.inf
    pct = max(0.0, pct)
    num_rounds = run_config.get("num-server-rounds")
    if num_rounds is None:
        return math.inf
    try:
        num_rounds = int(num_rounds)
    except (TypeError, ValueError):
        return math.inf
    if num_rounds <= 0:
        return math.inf
    return math.ceil((pct / 100.0) * num_rounds)


def _find_latest_local_model_path(hospital_id: str, current_round: Optional[int]) -> Optional[Path]:
    if not LOCAL_MODELS_DIR.exists():
        return None
    latest_round = -1
    latest_path = None
    pattern = f"{RUN_NAME}_{hospital_id}_round"
    for path in LOCAL_MODELS_DIR.glob(f"{RUN_NAME}_{hospital_id}_round*_auroc*.pt"):
        name = path.stem
        if pattern not in name:
            continue
        try:
            round_part = name.split("_round", 1)[1].split("_auroc", 1)[0]
            round_idx = int(round_part)
        except (IndexError, ValueError):
            continue
        if current_round is not None and round_idx >= current_round:
            continue
        if round_idx > latest_round:
            latest_round = round_idx
            latest_path = path
    return latest_path


def _load_personalized_weights(model: Net, hospital_id: str, server_round: int, run_config: dict) -> bool:
    start_round = _personalization_start_round(run_config)
    if server_round <= start_round:
        return False
    checkpoint_path = _find_latest_local_model_path(hospital_id, server_round)
    if checkpoint_path is None:
        return False
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    log(f"Loaded personalized weights for {hospital_id} from {checkpoint_path.name}")
    return True


def _evaluate_local_model(model: Net, dataset_name: str, image_size: int, device: torch.device):
    valloader = load_data(dataset_name, "eval", image_size=image_size)
    eval_loss, tp, tn, fp, fn, probs, labels = test_fn(model, valloader, device)
    metrics = compute_metrics({
        "eval_loss": eval_loss,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "probs": probs,
        "labels": labels,
    })
    return metrics


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    partition_id = context.node_config["partition-id"]
    hospital_suffix = PARTITION_HOSPITAL_MAP[partition_id]
    hospital_id = f"Hospital{hospital_suffix}"
    server_round = _get_server_round(msg, partition_id)

    # Load the model and initialize it with the received or personalized weights
    model = Net()
    personalized_loaded = _load_personalized_weights(model, hospital_id, server_round, context.run_config)
    if not personalized_loaded:
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    # Load the data
    dataset_name = hospital_id
    print(f"Running training on dataset: {dataset_name}")
    image_size = context.run_config["image-size"]
    trainloader = load_data(dataset_name, "train", image_size=image_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Evaluate locally to compute metrics for saving personalized checkpoints
    local_metric = _evaluate_local_model(model, dataset_name, image_size, device)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    save_local_model(model_record, local_metric, server_round, RUN_NAME, hospital_id)
    metrics = {
        "partition-id": context.node_config["partition-id"],
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
    image_size = context.run_config["image-size"]
    valloader = load_data(dataset_name, "eval", image_size=image_size)

    eval_loss, tp, tn, fp, fn, probs, labels = test_fn(model, valloader, device)

    metric_record = MetricRecord({
        "partition-id": context.node_config["partition-id"],
        "eval_loss": eval_loss,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num-examples": len(valloader.dataset),
        "probs": probs.tolist(),  # Convert numpy array to list for MetricRecord
        "labels": labels.tolist(),  # Convert numpy array to list for MetricRecord
    })
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
