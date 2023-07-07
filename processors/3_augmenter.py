import cv2
import os
import albumentations as A
import glob
import shutil
from tqdm import tqdm
import argparse

transforms = A.Compose(
    [
        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5),
                A.Transpose(p=0.5),
                A.Perspective(p=0.5),
            ],
            p=0.8,
        ),
        A.OneOf(
            [
                A.GaussNoise(p=0.3),
                A.MotionBlur(p=0.2),
                A.MedianBlur(p=0.2),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
            ],
            p=0.5,
        ),
    ]
)


def augment(root_dir, dest_dir, num_aug):
    """
    Augments the images in the training set and saves them in the destination directory.
    Copies the rest of the images from the validation and test sets to the destination directory.

    Inputs:
    - root_dir: Path to the root directory of the dataset
    - dest_dir: Path to the destination directory to save the augmented data
    - num_aug: Number of augmented images to create for each image in the training set
    """

    # Get only images from training set in split_dataset
    image_paths = glob.glob(f"{root_dir}/train/*/*.jpg")

    # Create destination directory
    shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(dest_dir)

    # Make directory for training set in destination directory
    os.makedirs(os.path.join(dest_dir, "train"))
    os.makedirs(os.path.join(dest_dir, "train", "healthy"))
    os.makedirs(os.path.join(dest_dir, "train", "wssv"))

    # Copy rest of images from split_dataset to destination directory
    shutil.copytree(os.path.join(root_dir, "valid"), os.path.join(dest_dir, "valid"))
    shutil.copytree(os.path.join(root_dir, "test"), os.path.join(dest_dir, "test"))

    for idx, image_path in enumerate(tqdm(image_paths)):
        image = cv2.imread(image_path)

        # Could be "healthy" or "wssv"
        label = image_path.split(os.path.sep)[-2]

        # Could be "train", "valid", or "test"
        split = image_path.split(os.path.sep)[2]

        # Save original image
        filename = os.path.join(dest_dir, split, label, str(idx)) + ".jpg"
        cv2.imwrite(filename, image)

        # Save augmented images
        for i in range(num_aug):
            augmented_image = transforms(image=image)["image"]

            filename = os.path.join(dest_dir, split, label, str(idx))
            augmented_filename = filename + ("_augmented_%d" % (i + 1)) + ".jpg"

            cv2.imwrite(augmented_filename, augmented_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--num_aug", type=int, default=5)
    args = parser.parse_args()

    augment(args.root_dir, args.dest_dir, args.num_aug)


if __name__ == "__main__":
    main()
