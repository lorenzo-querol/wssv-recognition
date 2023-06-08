import cv2
import numpy as np
from tqdm import tqdm
import glob
import shutil
import os

DEST_DIR = "dataset-segmented"
ROOT_DIR = "dataset-split"


def segment(image_paths, split):
    for i, image_path in enumerate(tqdm(image_paths)):
        image = cv2.imread(image_path)

        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Define the criteria and perform k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 2, None, criteria, 2, cv2.KMEANS_RANDOM_CENTERS
        )

        # Calculate the average pixel value for each cluster
        average_colors = np.uint8(centers)

        # Find the index of the cluster with the lower average pixel value
        cluster_index = np.argmin(average_colors)

        # Convert the label image to 8-bit and reshape it to the original image shape
        labels = labels.reshape(image.shape[:2]).astype(np.uint8)

        white_area = np.sum(labels == 1)

        # Convert the label image to 8-bit and reshape it to the original image shape
        labels = labels.reshape(image.shape[:2]).astype(np.uint8)

        # Invert the segmentation mask if the white area is larger than the segmented part
        if white_area > (labels.size - white_area):
            labels = 1 - labels

        # Segment the image based on the labels
        segmented_image = np.where(labels == 1, 255, 0).astype(np.uint8)

        segmented_image = cv2.bitwise_and(image, image, mask=segmented_image)

        filename = f"{DEST_DIR}/{split}/{image_path.split('/')[-2]}/{i}.jpg"
        cv2.imwrite(filename, segmented_image)


# Create destination directory
shutil.rmtree(DEST_DIR, ignore_errors=True)
os.makedirs(DEST_DIR)

# Make directory for training set in destination directory
os.makedirs(os.path.join(DEST_DIR, "train"))
os.makedirs(os.path.join(DEST_DIR, "train", "healthy"))
os.makedirs(os.path.join(DEST_DIR, "train", "wssv"))

os.makedirs(os.path.join(DEST_DIR, "val"))
os.makedirs(os.path.join(DEST_DIR, "val", "healthy"))
os.makedirs(os.path.join(DEST_DIR, "val", "wssv"))

# Copy rest of images from split_dataset to destination directory
shutil.copytree(os.path.join(ROOT_DIR, "test"), os.path.join(DEST_DIR, "test"))

# Get only images from training set in split_dataset
train_paths = glob.glob(f"{ROOT_DIR}/train/*/*.jpg")
valid_paths = glob.glob(f"{ROOT_DIR}/val/*/*.jpg")

segment(train_paths, "train")
segment(valid_paths, "val")
