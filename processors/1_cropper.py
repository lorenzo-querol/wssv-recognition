import glob
from cropimage import Cropper
from tqdm import tqdm
import cv2
import os
import shutil

IMAGE_DIR = "dataset/1-raw-v4"
DEST_DIR = "dataset/2-cropped-v3"
SIZE = 224

healthy_images = glob.glob(f"{IMAGE_DIR}/healthy/*.jpg", recursive=True)
wssv_images = glob.glob(f"{IMAGE_DIR}/wssv/*.jpg", recursive=True)
cropper = Cropper()

# make destination directory
shutil.rmtree(DEST_DIR, ignore_errors=True)
os.makedirs(DEST_DIR)
os.makedirs(f"{DEST_DIR}/healthy")
os.makedirs(f"{DEST_DIR}/wssv")

print("Cropping healthy images...")
for i, path in enumerate(tqdm(healthy_images)):
    result = cropper.crop(path, False, (SIZE, SIZE))
    filename = f"{DEST_DIR}/healthy/{SIZE}_{i}.jpg"
    cv2.imwrite(filename, result)

print("Cropping wssv images...")
for i, path in enumerate(tqdm(wssv_images)):
    result = cropper.crop(path, False, (SIZE, SIZE))
    filename = f"{DEST_DIR}/wssv/{SIZE}_{i}.jpg"
    cv2.imwrite(filename, result)
