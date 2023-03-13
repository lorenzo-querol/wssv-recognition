#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:56:14 2023

@author: lorenzoquerol
"""

import cv2
import numpy as np
import os
import shutil
from PIL import Image
import glob
import random

import smartcrop
from sklearn.cluster import KMeans
from skimage.exposure import match_histograms

from utils import centroid_histogram, plot_colors

import matplotlib.pyplot as plt

from typing import List

plt.rcParams["figure.figsize"] = (12,50)

element = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]])

def segment_shrimp(file_name: str, show: bool):
    bgr_img = cv2.imread(file_name)
    
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    norm_img = rgb_img / 255.
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].imshow(rgb_img)
        ax[0].set_title("Original image")
        ax[1].imshow(blur_img, 'gray')
        ax[1].set_title("Gray image")
        plt.show()

    '''
    Masking using Otsu thresholding
    '''
    T, mask = cv2.threshold(
        blur_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    mask = cv2.normalize(mask, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mask_img = cv2.bitwise_and(norm_img, norm_img, mask=mask)
    
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].imshow(mask)
        ax[0].set_title("Mask")
        ax[1].imshow(mask_img, 'gray')
        ax[1].set_title("Masked image")
        plt.show()

    return mask_img


def crop_square_image(image_files: List[str], dest_dir: str, size: int):
    '''
    Crops images to 1:1 aspect ratio and saves them to the destination
    folder specified.

    Inputs:
    - image_files: a list of paths to each image
    - dest_dir: path to the destination folder
    - size: size of image

    Returns:
    - None
    '''

    sc = smartcrop.SmartCrop()

    ###########################################################################

    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)

    healthy_folder = f'{dest_dir}/train/healthy'
    wssv_folder = f'{dest_dir}/train/wssv'

    os.makedirs(healthy_folder)
    os.makedirs(wssv_folder)

    ###########################################################################

    for image_file in image_files:
        image = Image.open(image_file)

        localized = sc.crop(image, size, size)

        bbox = (
            localized['top_crop']['x'],
            localized['top_crop']['y'],
            localized['top_crop']['width'] + localized['top_crop']['x'],
            localized['top_crop']['height'] + localized['top_crop']['y']
        )

        cropped = image.crop(bbox)
        resized = cropped.resize((size, size))

    ###########################################################################
    
        file = image_file.replace('/', '\\')
        split = file.split('\\')
        split[0] = dest_dir
        split[-1] = '{label}-{file_name}.jpg'
        split[-1] = split[-1].format(label=split[2],
                                     file_name=len(os.listdir(healthy_folder if split[-2] == 'healthy' else wssv_folder)) + 1)

        final_path = '/'.join(split)
        resized.save(final_path)
    
if __name__ == "__main__":
    data_dir = 'wssv-dataset'
    dest_dir = 'cropped'

    image_files = glob.glob(f'{data_dir}/train/*/*.jpg', recursive=True)

    if not os.path.isdir(dest_dir):
        crop_square_image(image_files, dest_dir, 300)

    cropped_files = glob.glob(f'{dest_dir}/train/*/*.jpg', recursive=True)

    idx = random.randint(0, len(cropped_files)-1)
    
    segment_shrimp(cropped_files[idx], True)
