import cv2
import os
import albumentations as A
import glob
import shutil
from tqdm import tqdm

DEST_DIR = "dataset-augmented"
ROOT_DIR = "dataset-split"
IMG_EXT = ".jpg"
NUM_AUG = 10

transforms = A.Compose(
    [
        A.augmentations.transforms.CLAHE(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(),
        A.RandomBrightnessContrast(),
        A.ImageCompression(),
        A.ISONoise(),
        A.augmentations.geometric.transforms.Perspective(),
    ]
)

# Get only images from training set in split_dataset
image_paths = glob.glob(f"{ROOT_DIR}/train/*/*.jpg")

# Create destination directory
shutil.rmtree(DEST_DIR, ignore_errors=True)
os.makedirs(DEST_DIR)

# Make directory for training set in destination directory
os.makedirs(os.path.join(DEST_DIR, "train"))
os.makedirs(os.path.join(DEST_DIR, "train", "healthy"))
os.makedirs(os.path.join(DEST_DIR, "train", "wssv"))

# Copy rest of images from split_dataset to destination directory
shutil.copytree(os.path.join(ROOT_DIR, "val"), os.path.join(DEST_DIR, "val"))
shutil.copytree(os.path.join(ROOT_DIR, "test"), os.path.join(DEST_DIR, "test"))

for idx, image_path in enumerate(tqdm(image_paths)):
    image = cv2.imread(image_path)

    # Could be "healthy" or "wssv"
    label = image_path.split(os.path.sep)[-2]

    # Could be "train", "val", or "test"
    split = image_path.split(os.path.sep)[1]

    # Save original image
    filename = os.path.join(DEST_DIR, split, label, str(idx)) + IMG_EXT
    cv2.imwrite(filename, image)

    # Save augmented images
    for i in range(NUM_AUG):
        augmented_image = transforms(image=image)["image"]

        filename = os.path.join(DEST_DIR, split, label, str(idx))
        augmented_filename = filename + ("_augmented_%d" % (i + 1)) + IMG_EXT

        cv2.imwrite(augmented_filename, augmented_image)
