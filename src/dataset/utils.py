import os
import cv2 as cv
from glob import glob


def img_to_patches(img, patch_size, overlap):
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
    for img_path in imgs_paths:

        patch_name = img_path.replace("data/", patches_dir + "/")
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        patches = img_to_patches(img, patch_size, overlap)
        valid_ext = [".jpg", ".png", "tif", "geotif"]
        file_name, file_ext = os.path.splitext(patch_name)
        assert file_ext in valid_ext, f"Invalid file extension {file_ext}"
        for i, img in enumerate(patches):
            img_save_name = file_name + f"_{i+1}" + file_ext
            print(f"Saving patch {img_save_name}")
            if not os.path.exists(os.path.dirname(img_save_name)):
                os.makedirs(os.path.dirname(img_save_name))

            cv.imwrite(img_save_name, img)


if __name__ == "__main__":
    DATASET_PATH = "data/"

    create_patches(DATASET_PATH, patch_size=512, overlap=0, patches_dir=None)
