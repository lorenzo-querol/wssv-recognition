import os
import glob
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

from pathlib import Path
from rembg import remove, new_session

ROOT_DIR = "dataset/2-cropped-v3"
DEST_DIR = "dataset/SEGMENTED"
RANDOM_STATE = 42
SIZE = 224

# make destination directory
shutil.rmtree(DEST_DIR, ignore_errors=True)
os.makedirs(DEST_DIR)
os.makedirs(f"{DEST_DIR}/healthy")
os.makedirs(f"{DEST_DIR}/wssv")

model_name = "u2net"
session = new_session(model_name)

num = 0
for file in Path(f"{ROOT_DIR}/healthy").glob("*.jpg"):
    input_path = str(file)
    output_path = f"{DEST_DIR}/healthy/{SIZE}_{num}.jpg"

    input = Image.open(input_path)
    output = remove(
        input,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=255,
        alpha_matting_background_threshold=0,
        post_process_mask=True,
        bgcolor=(255, 255, 255, 255),
    )
    output = output.convert("RGB")
    output.save(output_path)

    num += 1

num = 0
for file in Path(f"{ROOT_DIR}/wssv").glob("*.jpg"):
    input_path = str(file)
    output_path = f"{DEST_DIR}/wssv/{SIZE}_{num}.jpg"

    input = Image.open(input_path)
    output = remove(
        input,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=255,
        alpha_matting_background_threshold=0,
        post_process_mask=True,
        bgcolor=(255, 255, 255, 255),
    )
    output = output.convert("RGB")
    output.save(output_path)

    num += 1
