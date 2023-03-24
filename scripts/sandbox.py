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
    
    
    hsv_images= []
    for image in images:
        hsv_img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2HSV)
        hsv_images.append(hsv_img)
    
    for i, image in enumerate(hsv_images):
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

to_hsv(healthy, healthy_folder)
to_hsv(wssv, wssv_folder)