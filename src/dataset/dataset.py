import os
import cv2 as cv
import numpy as np
import torch
from glob import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms


class SegDataset(Dataset):
    def __init__(
        self, path, resize=None, transform=None, img_ext="jpg", mask_ext="png"
    ):
        self.img_paths = glob(path + f"/img/*.{img_ext}")
        self.mask_paths = glob(path + f"/label/*.{mask_ext}")
        self.img_paths.sort()
        self.mask_paths.sort()
        # self.ids = [os.path.basename(p).split(".")[0] for p in self.img_paths]
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # id = self.ids[idx]
        img = cv.imread(self.img_paths[idx], cv.IMREAD_COLOR)
        mask = cv.imread(self.mask_paths[idx], cv.IMREAD_GRAYSCALE)

        if self.resize is not None:
            img = cv.resize(img, self.resize)
            mask = cv.resize(mask, self.resize)

        mask = np.expand_dims(mask, axis=2)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
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

            # Convert image to float32 for precise calculations
            img = img.astype(np.float32) / 255.0

            mean += img.mean(axis=(0, 1))
            std += img.std(axis=(0, 1))

        mean /= num_images
        std /= num_images

        return mean, std


if __name__ == "__main__":

    train_path = "data/patches/val"
    test_path = "data/patches/test"

    train_dataset = SegDataset(train_path, resize=None, transform=None)
    test_dataset = SegDataset(test_path, resize=None, transform=None)
    num = np.random.randint(0, len(train_dataset))
    img, mask = train_dataset[num]
    img2, mask2 = test_dataset[num]
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title(f"Train Image")
    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Train Mask")
    plt.subplot(2, 2, 3)
    plt.imshow(img2)
    plt.title(f"Test Image")
    plt.subplot(2, 2, 4)
    plt.imshow(mask2, cmap="gray")
    plt.title(f"Test Mask")
    plt.show()
