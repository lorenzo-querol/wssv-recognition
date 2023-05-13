import numpy as np
import cv2
import os
import albumentations as A
import glob
import shutil

transforms = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(interpolation=cv2.INTER_CUBIC),
        A.GaussianBlur(),
        A.RandomBrightnessContrast(),
        A.ImageCompression(quality_lower=85, quality_upper=100),
    ]
)

DATA_DIR = "dataset"
DEST_DIR = "augmented_dataset"
CLASSES = ["healthy", "wssv"]
NUM_AUG = 5
IMG_EXT = ".jpg"

# Reset destination directory
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)

for c in CLASSES:
    os.makedirs(os.path.join(DEST_DIR, c))

image_paths = glob.glob(f"{DATA_DIR}/*/*{IMG_EXT}", recursive=True)

for idx, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    label = image_path.split("\\")[-2]

    # Save original image
    filename = os.path.join(DEST_DIR, label, str(idx)) + IMG_EXT
    cv2.imwrite(filename, image)

    # Save augmented images
    for i in range(NUM_AUG):
        augmented_image = transforms(image=image)["image"]

        filename = os.path.join(DEST_DIR, label, str(idx))
        augmented_filename = filename + ("_augmented_%d" % (i + 1)) + IMG_EXT

        cv2.imwrite(augmented_filename, augmented_image)
