import os
import glob
import shutil
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RANDOM_STATE = 42


def load_images(path):
    images = []
    for filename in glob.glob(f"{path}/*.jpg"):
        images.append(cv2.imread(filename))
    return images


def split(root_dir, dest_dir):
    """
    Splits the dataset into train, validation, and test sets.

    Inputs:
    - root_dir: Path to the root directory of the dataset
    - dest_dir: Path to the destination directory to save the split data

    """
    healthy = load_images(f"{root_dir}/healthy")
    wssv = load_images(f"{root_dir}/wssv")

    healthy_count = len(healthy)
    wssv_count = len(wssv)

    X = np.vstack((healthy, wssv))
    y = np.array([0] * healthy_count + [1] * wssv_count)

    X = list(zip(X, y))

    # First split the data in two sets, 70% for training, 30% for Val/Test
    x_train, x_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Then split the 20/10% into validation and test sets
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp, y_temp, test_size=0.3333, random_state=42
    )

    # Save all sets to local directory called split
    shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(f"{dest_dir}/train/healthy")
    os.makedirs(f"{dest_dir}/train/wssv")
    os.makedirs(f"{dest_dir}/valid/healthy")
    os.makedirs(f"{dest_dir}/valid/wssv")
    os.makedirs(f"{dest_dir}/test/healthy")
    os.makedirs(f"{dest_dir}/test/wssv")

    print("Creating train set...")
    for i, (image, label) in enumerate(tqdm(x_train)):
        if label == 0:
            cv2.imwrite(f"{dest_dir}/train/healthy/{i}.jpg", image)
        else:
            cv2.imwrite(f"{dest_dir}/train/wssv/{i}.jpg", image)

    print("Creating valid set...")
    for i, (image, label) in enumerate(tqdm(x_valid)):
        if label == 0:
            cv2.imwrite(f"{dest_dir}/valid/healthy/{i}.jpg", image)
        else:
            cv2.imwrite(f"{dest_dir}/valid/wssv/{i}.jpg", image)

    print("Creating test set...")
    for i, (image, label) in enumerate(tqdm(x_test)):
        if label == 0:
            cv2.imwrite(f"{dest_dir}/test/healthy/{i}.jpg", image)
        else:
            cv2.imwrite(f"{dest_dir}/test/wssv/{i}.jpg", image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", help="Root directory of images")
    parser.add_argument(
        "--dest_dir", help="Destination directory to save the split data"
    )
    args = parser.parse_args()

    split(args.root_dir, args.dest_dir)


if __name__ == "__main__":
    main()
