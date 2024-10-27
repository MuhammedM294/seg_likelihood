import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from glob import glob


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

        mask = np.where(mask == 255, 1, 0)

        return img, mask

    @property
    def get_mean_std(self):
        mean = np.zeros(3)
        std = np.zeros(3)
        num_images = len(self.img_paths)

        for img_path in self.img_paths:
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
    dataset = SegDataset("data/patches/train")
    ran_int = np.random.randint(0, len(dataset))
    print(ran_int)
    img, mask = dataset[ran_int]
    print(np.unique(mask, return_counts=True))
    print(img.shape, mask.shape)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0], cmap="gray")
    plt.axis("off")
    plt.show()
