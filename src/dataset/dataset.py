import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from glob import glob
from utils import get_transform


class SegDataset(Dataset):
    def __init__(
        self, path, resize=None, transform=None, img_ext="jpg", mask_ext="png"
    ):
        self.img_paths = glob(path + f"/img/*.{img_ext}")
        self.mask_paths = glob(path + f"/label/*.{mask_ext}")
        self.img_paths.sort()
        self.mask_paths.sort()
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv.imread(self.img_paths[idx], cv.IMREAD_COLOR)
        mask = cv.imread(self.mask_paths[idx], cv.IMREAD_GRAYSCALE)

        if self.resize is not None:
            img = cv.resize(img, self.resize)
            mask = cv.resize(mask, self.resize)

        mask = np.expand_dims(mask, axis=2)
        if self.transform is not None:
            data = self.transform(image=img, mask=mask)
            img = data["image"]
            mask = data["mask"]

        mask = mask.permute(2, 0, 1)

        return img, mask

    @property
    def get_mean_std(self):
        mean = np.zeros(3)
        std = np.zeros(3)
        num_images = len(self.img_paths)

        for i, img_path in enumerate(self.img_paths):
            img = cv.imread(img_path, cv.IMREAD_COLOR)

            if self.resize is not None:
                img = cv.resize(img, self.resize)

            img = img.astype(np.float32) / 255.0

            mean += img.mean(axis=(0, 1))
            std += img.std(axis=(0, 1))

        mean /= num_images
        std /= num_images

        return mean, std


if __name__ == "__main__":
    DATA_MEANS = (0.30451819, 0.33103726, 0.30483206)
    DATA_STD = (0.09850518, 0.10860541, 0.12041442)
    train_dataset = SegDataset(
        "data/patches/train",
        transform=get_transform(is_train=True, mean=DATA_MEANS, std=DATA_STD),
    )
    test_dataset = SegDataset(
        "data/patches/test",
        transform=get_transform(is_train=False, mean=DATA_MEANS, std=DATA_STD),
    )
    val_dataset = SegDataset(
        "data/patches/val",
        transform=get_transform(is_train=False, mean=DATA_MEANS, std=DATA_STD),
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False
    )

    print(f"Train dataloader: {len(train_dataloader)} batches")
    print(f"Test dataloader: {len(test_dataloader)} batches")
    print(f"Validation dataloader: {len(val_dataloader)} batches")

    img, mask = next(iter(test_dataloader))
    print(f"Image shape: {img.shape}, mask shape: {mask.shape}")
