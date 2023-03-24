# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:10:24 2023

@author: Enzo
"""
import glob
import smartcrop 
from alive_progress import alive_bar
from PIL import Image
import os
import shutil

image_dir = '../wssv-dataset/train'
dest_dir = '../cropped' 
size = 200

healthy_images = glob.glob(f'{image_dir}/healthy/*.jpg', recursive=True)
wssv_images = glob.glob(f'{image_dir}/wssv/*.jpg', recursive=True)

sc = smartcrop.SmartCrop()

if os.path.isdir(dest_dir):
    shutil.rmtree(dest_dir)
    os.makedirs(f'{dest_dir}/healthy')
    os.makedirs(f'{dest_dir}/wssv')
    
with alive_bar(len(healthy_images), bar='smooth', spinner=None) as bar:
    for i, path in enumerate(healthy_images):
        image = Image.open(path)
        localized = sc.crop(image, size, size)
    
        bbox = (
            localized['top_crop']['x'],
            localized['top_crop']['y'],
            localized['top_crop']['width'] + localized['top_crop']['x'],
            localized['top_crop']['height'] + localized['top_crop']['y']
        )
        
        cropped = image.crop(bbox)
        resized = cropped.resize((size, size))
        
        cropped_filename = f'{dest_dir}/healthy/{size}_{i}.jpg'
        
        resized.save(cropped_filename)
        bar()

with alive_bar(len(wssv_images), bar='smooth', spinner=None) as bar:
    for i, path in enumerate(wssv_images):
        image = Image.open(path)
        localized = sc.crop(image, size, size)
    
        bbox = (
            localized['top_crop']['x'],
            localized['top_crop']['y'],
            localized['top_crop']['width'] + localized['top_crop']['x'],
            localized['top_crop']['height'] + localized['top_crop']['y']
        )
        
        cropped = image.crop(bbox)
        resized = cropped.resize((size, size))
        
        cropped_filename = f'{dest_dir}/wssv/{size}_{i}.jpg'
        
        resized.save(cropped_filename)
        bar()
                
