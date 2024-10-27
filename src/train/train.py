import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
from tqdm import tqdm
import torch.nn.functional as F

import wandb

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(module_path)


from dataset.dataset import SegDataset
from dataset.utils import get_transform
from models.unet.model import UNet
from utils import get_device, set_seed, configure_deterministic_behavior

DATA_MEANS = (0.30451819, 0.33103726, 0.30483206)
DATA_STD = (0.09850518, 0.10860541, 0.12041442)

train_dataset = SegDataset(
    "data/patches/train",
    transform=get_transform(is_train=True, mean=DATA_MEANS, std=DATA_STD),
    resize=None,
)


def get_dataset_subset(dataset, num_samples):
    indices = get_subset_indices(dataset, num_samples)
    return get_subset(dataset, indices)


def get_subset_indices(dataset, num_samples):
    return np.random.choice(len(dataset), num_samples, replace=False)


def get_subset(dataset, indices):
    return Subset(dataset, indices)


val_dataset = SegDataset(
    "data/patches/val",
    transform=get_transform(is_train=False, mean=DATA_MEANS, std=DATA_STD),
    resize=None,
)

test_dataset = SegDataset(
    "data/patches/test",
    transform=get_transform(is_train=False, mean=DATA_MEANS, std=DATA_STD),
    resize=None,
)

train_dataset = get_dataset_subset(train_dataset, 3200)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

val_dataset = get_dataset_subset(val_dataset, 1600)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)


def criterion(logits, mask, mode="binary"):
    return DiceLoss(mode=mode)(logits, mask) + nn.BCEWithLogitsLoss()(logits, mask)


def train(model, train_loader, optimizer, criterion, epochs, grad_scaler, device):

    # experiment = wandb.init(
    #     project="Uncertainty Estimation Ex 1",
    #     job_type="train",
    #     resume="allow",
    #     anonymous="must",
    # )

    # experiment.config.update(
    #     {
    #         "epochs": epochs,
    #         "batch_size": 32,
    #         "learning_rate": 1e-3,
    #         "weight_decay": 1e-3,
    #         "optimizer": optimizer,
    #         "amp": grad_scaler,
    #         "allow_val_change": True,
    #     }
    # )
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        with tqdm(
            total=len(train_loader),
            desc=f"Train Epoch {epoch + 1}/{epochs}",
            unit="batch",
        ) as pbar:
            for img, mask in train_loader:

                # Check the shape of the input
                assert (
                    img.shape[1] == model.in_channels
                ), f"Expected {model.in_channels} channels, got {img.shape[1]}"

                # Move data to device
                img = img.to(device, dtype=torch.float32)
                mask = mask.to(device, dtype=torch.float32)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                # with torch.amp.autocast(device, enabled=True):
                logits = model(img)
                mask = mask.permute(0, 3, 1, 2)

                # Check the shape of the output
                assert (
                    logits.shape == mask.shape
                ), f"Expected shape {mask.shape}, got {logits.shape}"

                # Calculate loss
                loss = criterion(logits, mask)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_train_loss += loss.item()
                pbar.set_postfix({"loss (batch)": loss.item()})
                pbar.update()

            epoch_train_loss /= len(train_loader)
            print(f"Train Epoch {epoch + 1}/{epochs} Loss: {epoch_train_loss:.4f}")

        with tqdm(
            total=len(val_dataloader),
            desc=f"Val Epoch {epoch + 1}/{epochs}",
            unit="batch",
        ) as pbar:
            for img, mask in val_dataloader:
                model.eval()
                with torch.no_grad():
                    img = img.to(device, dtype=torch.float32)
                    mask = mask.to(device, dtype=torch.float32)
                    logits = model(img)
                    mask = mask.permute(0, 3, 1, 2)
                    val_loss = criterion(logits, mask)
                    epoch_val_loss += val_loss.item()
                    pbar.set_postfix({"val_loss (batch)": val_loss.item()})
                    pbar.update()

            epoch_val_loss /= len(val_dataloader)
            print(
                f"Val Epoch {epoch + 1}/{epochs} Validation Loss: {epoch_val_loss:.4f}"
            )


if __name__ == "__main__":
    set_seed(42)
    configure_deterministic_behavior()
    device = get_device()
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
    grad_scaler = torch.amp.GradScaler(device, enabled=True)

    train(model, train_dataloader, optimizer, criterion, 10, grad_scaler, device)
