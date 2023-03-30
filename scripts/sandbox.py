# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 00:06:57 2023

@author: Enzo
"""
import cv2
import glob
import shutil
import os


def to_hsv(images, dest):
    hsv_images = []
    for image in images:
        hsv_img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2HSV)
        hsv_images.append(hsv_img)

    for i, image in enumerate(hsv_images):
        cv2.imwrite(f'{dest}/{i}.jpg', image)


def to_luv(images, dest):
    luv_images = []
    for image in images:
        luv_img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2LUV)
        # Get third channel
        L, U, V = cv2.split(luv_img)

        # Apply median blur
        blurred = cv2.medianBlur(L, 7)

        # Apply Otsu thresholding
        ret, threshold = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        # Get mask for cropped image
        mask = cv2.merge([opened, opened, opened])

        # Get the output
        finalOutput = 255 * (mask * image)

        luv_images.append(finalOutput)

    for i, image in enumerate(luv_images):
        cv2.imwrite(f'{dest}/{i}.jpg', image)


def to_otsu(images, dest):

    otsu_images = []
    for image in images:
        hsv = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2HSV)

        # Get third channel
        L, U, V = cv2.split(hsv)

        # Apply median blur
        blurred = cv2.medianBlur(L, 7)

        # Apply Otsu thresholding
        ret, threshold = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        # Get mask for cropped image
        # mask = cv2.merge([opened, opened, opened])

        otsu_images.append(opened)

    for i, image in enumerate(otsu_images):
        cv2.imwrite(f'{dest}/{i}.jpg', image)


data_dir = 'wssv-dataset'
dest_dir = 'hsv'

healthy = glob.glob(f'{data_dir}/train/healthy/*.jpg')
wssv = glob.glob(f'{data_dir}/train/wssv/*.jpg')

if os.path.isdir(dest_dir):
    shutil.rmtree(dest_dir)

healthy_folder = f'{dest_dir}/healthy'
wssv_folder = f'{dest_dir}/wssv'

os.makedirs(healthy_folder)
os.makedirs(wssv_folder)

to_otsu(healthy, healthy_folder)
to_otsu(wssv, wssv_folder)
