import os
import glob
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

MAIN_DIR = "dataset"
RANDOM_STATE = 42


def load_images(path):
    images = []
    for filename in glob.glob(f"{path}/*.jpg"):
        images.append(cv2.imread(filename))
    return images


healthy = load_images(f"{MAIN_DIR}/healthy")
wssv = load_images(f"{MAIN_DIR}/wssv")

healthy_count = len(healthy)
wssv_count = len(wssv)

labels = np.array([0] * healthy_count + [1] * wssv_count)

all_images = np.vstack((healthy, wssv))
all_images = list(zip(all_images, labels))

# Split dataset to train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    all_images, labels, test_size=0.4, random_state=RANDOM_STATE, stratify=labels
)

# Split validation set to validation and test sets
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size=0.2, random_state=RANDOM_STATE, stratify=y_test
)

# Save all sets to local directory called split-dataset
shutil.rmtree("split-dataset", ignore_errors=True)
os.makedirs("split-dataset/train/healthy")
os.makedirs("split-dataset/train/wssv")
os.makedirs("split-dataset/val/healthy")
os.makedirs("split-dataset/val/wssv")
os.makedirs("split-dataset/test/healthy")
os.makedirs("split-dataset/test/wssv")

for i, (image, label) in enumerate(x_train):
    if label == 0:
        cv2.imwrite(f"split-dataset/train/healthy/{i}.jpg", image)
    else:
        cv2.imwrite(f"split-dataset/train/wssv/{i}.jpg", image)

for i, (image, label) in enumerate(x_val):
    if label == 0:
        cv2.imwrite(f"split-dataset/val/healthy/{i}.jpg", image)
    else:
        cv2.imwrite(f"split-dataset/val/wssv/{i}.jpg", image)

for i, (image, label) in enumerate(x_test):
    if label == 0:
        cv2.imwrite(f"split-dataset/test/healthy/{i}.jpg", image)
    else:
        cv2.imwrite(f"split-dataset/test/wssv/{i}.jpg", image)
