import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import models

try:
    from torchvision.models import EfficientNet_B0_Weights
except (ImportError, AttributeError):
    EfficientNet_B0_Weights = None

try:
    import torchxrayvision as xrv
except ImportError:
    xrv = None
from tqdm import tqdm

hospital_datasets = {}  # Cache loaded hospital datasets


def _env_flag(name: str, default: bool = False) -> bool:
    """Return True if the environment variable is set to a truthy value."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class Net(nn.Module):
    """Configurable Chest X-ray classifier with lightweight pretrained backbones."""

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        train_backbone: Optional[bool] = None,
        architecture: Optional[str] = None,
    ):
        super().__init__()

        self.architecture = (architecture or os.environ.get("MODEL_NAME") or "resnet50-res224-nih").lower()
        self.backbone, backbone_features = self._build_backbone(self.architecture, pretrained)
        self.classifier = nn.Linear(backbone_features, 1)

        checkpoint_path = checkpoint_path or os.environ.get("CHESTXRAY_PRETRAINED")
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        if train_backbone is None:
            train_backbone = _env_flag("TRAIN_BACKBONE", default=False)

        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _build_backbone(self, architecture: str, pretrained: bool):
        if architecture in {"resnet50", "resnet50-res224-nih", "resnet50_nih"}:
            return self._build_xrv_resnet50(pretrained)
        if architecture in {"efficientnet_b0", "efficientnet-b0"}:
            return self._build_efficientnet_b0(pretrained)
        raise ValueError(
            "Unknown architecture '{architecture}'. Supported: resnet50-res224-nih, efficientnet_b0.".format(
                architecture=architecture
            )
        )

    def _build_xrv_resnet50(self, pretrained: bool):
        if xrv is None:
            raise ImportError(
                "torchxrayvision is required for resnet50-res224-nih. Install it via `pip install torchxrayvision`."
            )
        weights = "resnet50-res224-nih" if pretrained else None
        base_model = xrv.models.ResNet(weights=weights)
        backbone = base_model.model
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    def _build_efficientnet_b0(self, pretrained: bool):
        if pretrained and EfficientNet_B0_Weights is None:
            raise ImportError(
                "torchvision>=0.13 is required for EfficientNet_B0 pretrained weights. Upgrade torchvision or set pretrained=False."
            )
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained and EfficientNet_B0_Weights is not None else None
        model = models.efficientnet_b0(weights=weights)
        orig_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        model.features[0][0] = new_conv
        if weights is not None:
            with torch.no_grad():
                new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, in_features

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        normalized_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            normalized_state_dict[new_key] = value

        current_state = self.state_dict()
        for key, tensor in list(normalized_state_dict.items()):
            if (
                key in current_state
                and isinstance(tensor, torch.Tensor)
                and isinstance(current_state[key], torch.Tensor)
                and tensor.ndim >= 2
                and current_state[key].ndim >= 2
                and current_state[key].shape[1] == 1
                and tensor.shape[1] != 1
            ):
                normalized_state_dict[key] = tensor.mean(dim=1, keepdim=True)

        incompatible = self.load_state_dict(normalized_state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"Checkpoint missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Checkpoint unexpected keys: {incompatible.unexpected_keys}")

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def collate_preprocessed(batch):
    """Collate function for preprocessed data: Convert list of dicts to dict of batched tensors."""
    result = {}
    for key in batch[0].keys():
        if key in ["x", "y"]:
            # Convert lists to tensors and stack
            result[key] = torch.stack([torch.tensor(item[key]) for item in batch])
        else:
            # Keep other fields as lists
            result[key] = [item[key] for item in batch]
    return result


def load_data(
    dataset_name: str,
    split_name: str,
    image_size: int = 128,
    batch_size: int = 128,
):
    """Load hospital X-ray data.

    Args:
        dataset_name: Dataset name ("HospitalA", "HospitalB", "HospitalC")
        split_name: Split name ("train", "eval")
        image_size: Image size (128 or 224)
        batch_size: Number of samples per batch
    """
    dataset_dir = os.environ["DATASET_DIR"]

    # Use preprocessed dataset based on image_size
    cache_key = f"{dataset_name}_{split_name}_{image_size}"
    dataset_path = f"{dataset_dir}/xray_fl_datasets_preprocessed_{image_size}/{dataset_name}"

    # Load and cache dataset
    global hospital_datasets
    if cache_key not in hospital_datasets:
        full_dataset = load_from_disk(dataset_path)
        hospital_datasets[cache_key] = full_dataset[split_name]
        print(f"Loaded {dataset_path}/{split_name}")

    data = hospital_datasets[cache_key]
    shuffle = (split_name == "train")  # shuffle only for training splits
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_preprocessed)
    return dataloader


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_loss = running_loss / (len(trainloader) * epochs)
    return avg_loss


def test(net, testloader, device):
    """Evaluate the model on the test set (binary classification).

    Returns:
        avg_loss: Average BCE loss
        tp: True Positives
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        all_probs: Array of prediction probabilities (for AUROC)
        all_labels: Array of true labels (for AUROC)
    """
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    net.eval()
    total_loss = 0.0

    all_probs = []
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in testloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            # Store for metric calculation
            all_probs.append(probs.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / len(testloader)

    # Flatten arrays
    all_probs = np.concatenate(all_probs).flatten()
    all_predictions = np.concatenate(all_predictions).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # Calculate confusion matrix components
    tp = int(np.sum((all_predictions == 1) & (all_labels == 1)))
    tn = int(np.sum((all_predictions == 0) & (all_labels == 0)))
    fp = int(np.sum((all_predictions == 1) & (all_labels == 0)))
    fn = int(np.sum((all_predictions == 0) & (all_labels == 1)))

    return avg_loss, tp, tn, fp, fn, all_probs, all_labels
