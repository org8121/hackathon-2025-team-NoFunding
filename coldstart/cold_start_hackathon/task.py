import os

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

hospital_datasets = {}  # Cache loaded hospital datasets


class Net(nn.Module):
    """Starting point: ResNet18-based model for binary chest X-ray classification."""

    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(weights=None)
        # Adapt to grayscale input
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        # Binary classification head (single logit)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)  # No sigmoid, using BCEWithLogitsLoss


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
    batch_size: int = 16,
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
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
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
