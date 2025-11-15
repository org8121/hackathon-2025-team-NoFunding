import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import models

try:
    from torchvision.models import EfficientNet_B0_Weights, ResNet34_Weights
except (ImportError, AttributeError):
    EfficientNet_B0_Weights = None
    ResNet34_Weights = None
from tqdm import tqdm

hospital_datasets = {}  # Cache loaded hospital datasets


def _env_flag(name: str, default: bool = False) -> bool:
    """Return True if the environment variable is set to a truthy value."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class Net(nn.Module):
    """Chest X-ray classifier backed by a frozen torchvision backbone."""

    def __init__(self, train_backbone: Optional[bool] = None, architecture: Optional[str] = None):
        super().__init__()
        self.architecture = (architecture or os.environ.get("MODEL_NAME") or "efficientnet_b0").lower()
        self.backbone, backbone_features = self._build_backbone(self.architecture)
        self.classifier = nn.Linear(backbone_features, 1)

        if train_backbone is None:
            train_backbone = _env_flag("TRAIN_BACKBONE", default=False)

        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _build_backbone(self, architecture: str):
        """Return a backbone model and its feature dimensionality."""
        if architecture in {"resnet34", "resnet34-imagenet"}:
            return self._build_resnet34()
        if architecture in {"efficientnet_b0", "efficientnet-b0"}:
            return self._build_efficientnet_b0()
        raise ValueError(
            f"Unknown architecture '{architecture}'. Supported: resnet34, efficientnet_b0."
        )

    def _build_resnet34(self):
        """Create a 1-channel ResNet34 initialized with ImageNet1K v1 weights."""
        if ResNet34_Weights is None:
            raise ImportError(
                "torchvision>=0.13 is required for ResNet34 pretrained weights. "
                "Install a compatible torchvision version to continue."
            )
        weights = ResNet34_Weights.IMAGENET1K_V1
        try:
            model = models.resnet34(weights=weights)
        except TypeError:
            # Fallback for older torchvision versions
            model = models.resnet34(pretrained=True)

        orig_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            model.conv1.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))

        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_features

    def _build_efficientnet_b0(self):
        """Create a 1-channel EfficientNet-B0 initialized with ImageNet1K v1 weights."""
        if EfficientNet_B0_Weights is None:
            raise ImportError(
                "torchvision>=0.13 is required for EfficientNet_B0 pretrained weights. "
                "Install a compatible torchvision version to continue."
            )
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        try:
            model = models.efficientnet_b0(weights=weights)
        except TypeError:
            model = models.efficientnet_b0(pretrained=True)

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
        with torch.no_grad():
            new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))

        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, in_features

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
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Prefetch next batches
        persistent_workers=True,  # Keep workers alive between epochs
        collate_fn=collate_preprocessed
    )
    return dataloader


def calculate_pos_weight(trainloader):
    """Calculate positive class weight for handling class imbalance."""
    num_positive = 0
    num_negative = 0
    for batch in trainloader:
        labels = batch["y"]
        num_positive += labels.sum().item()
        num_negative += (labels == 0).sum().item()

    if num_positive == 0:
        return torch.tensor([1.0])

    pos_weight = num_negative / num_positive
    print(f"Class imbalance - Positive weight: {pos_weight:.2f}")
    return torch.tensor([pos_weight])


def train(
    net, 
    trainloader, 
    epochs, 
    lr, 
    device,
    use_amp=True,
    gradient_clip_norm=10.0
):
    """Train the model with mixed precision and gradient clipping.
    
    Args:
        net: Model to train
        trainloader: Training data loader
        epochs: Number of local epochs
        lr: Learning rate
        device: Device to train on
        use_amp: Enable automatic mixed precision (CRITICAL for memory efficiency)
        gradient_clip_norm: Max gradient norm for clipping (stability)
    """
    net.to(device)
    
    # Calculate class weights for imbalanced data
    pos_weight = calculate_pos_weight(trainloader).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    
    # Optimizer with L2 regularization
    optimizer = torch.optim.Adam(
        (p for p in net.parameters() if p.requires_grad), 
        lr=lr,
        weight_decay=1e-4
    )
    
    # Mixed precision scaler for memory efficiency
    scaler = GradScaler() if use_amp else None
    
    net.train()
    running_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            
            if use_amp:
                # Mixed precision forward pass
                with autocast():
                    outputs = net(x)
                    loss = criterion(outputs, y)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping for stability
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=gradient_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(x)
                loss = criterion(outputs, y)
                loss.backward()
                
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=gradient_clip_norm)
                
                optimizer.step()
            
            epoch_loss += loss.item()
        
        running_loss += epoch_loss / len(trainloader)
    
    avg_loss = running_loss / epochs
    return avg_loss


def test(net, testloader, device, use_amp=True):
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
            
            if use_amp:
                with autocast():
                    outputs = net(x)
                    loss = criterion(outputs, y)
            else:
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
