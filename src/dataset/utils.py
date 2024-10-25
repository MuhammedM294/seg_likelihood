import os
import cv2 as cv
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def img_to_patches(img, patch_size, overlap):
    """
    Splits an image into patches of size patch_size x patch_size with overlap.
    """
    h, w, _ = img.shape
    step = patch_size - overlap

    for i in range(0, h, step):
        for j in range(0, w, step):

            end_i = min(i + patch_size, h)
            end_j = min(j + patch_size, w)
            start_i = end_i - patch_size
            start_j = end_j - patch_size

            yield img[start_i:end_i, start_j:end_j]


def create_patches(dataset_dir, patch_size=512, overlap=0, patches_dir=None):
    """
    Splits all images in the dataset into patches of size patch_size x patch_size with overlap.
    """

    assert os.path.exists(dataset_dir), f"Path {dataset_dir} does not exist"
    assert (
        patches_dir != dataset_dir
    ), "The save_to path should not be the same as imgs_path"

    if patches_dir is None:
        patches_dir = os.path.join(dataset_dir, "patches")

    if not os.path.exists(patches_dir):
        os.makedirs(os.path.dirname(patches_dir))

    imgs_paths = glob(os.path.join(dataset_dir, "**/*"), recursive=True)
    imgs_paths = [f for f in imgs_paths if f.endswith((".jpg", ".png"))]

    print(f"Found {len(imgs_paths)} images")

    for img_path in tqdm(imgs_paths, desc="Creating patches"):

        patch_name = img_path.replace("data/", patches_dir + "/")
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        patches = img_to_patches(img, patch_size, overlap)

        for i, img in enumerate(patches):
            if patch_name.endswith(".jpg"):
                img_save_name = patch_name.replace(".jpg", f"_{i+1}.jpg")
            elif patch_name.endswith(".png"):
                img_save_name = patch_name.replace(".png", f"_{i+1}.png")
            else:
                raise ValueError("Invalid image extension")

            if not os.path.exists(os.path.dirname(img_save_name)):
                os.makedirs(os.path.dirname(img_save_name))

            cv.imwrite(img_save_name, img)


def get_transform(is_train=True, mean=None, std=None):
    """
    Returns the appropriate transformation based on whether the user wants training or testing transformations.
    """
    if is_train:
        # Training Transformations
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=160),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomCrop(height=128, width=128),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=mean,
                    std=std,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Testing Transformations
        return A.Compose(
            [
                A.Resize(512, 512),
                A.Normalize(
                    mean=mean,
                    std=std,
                ),
                ToTensorV2(),
            ]
        )


if __name__ == "__main__":
    DATASET_PATH = "data/"

    create_patches(DATASET_PATH, patch_size=512, overlap=0, patches_dir="data/patches")
