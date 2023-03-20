#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 10:44:33 2023

@author: lorenzoquerol
"""

import seaborn as sns
sns.set_theme(style="ticks")

from PIL import Image
import smartcrop
import cv2
import glob
import pandas as pd

from alive_progress import alive_bar
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600

def load_images(path):
    images = []
    filenames = os.listdir(path)

    with alive_bar(len(filenames), bar='smooth', spinner=None) as bar:
        for filename in filenames:
            image = cv2.imread(os.path.join(path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            bar()

    return np.array(images, dtype=object)

def show_raw_images(images, classname, start_index=0):
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(15, 3))
    plt.suptitle(classname)
    axes = axes.ravel()
    index = start_index
    for i in range(4):
        axes[i].imshow(images[index])
        # axes[i].set_title(images[index].shape)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        index += 1
    plt.show()

def show_images_with_labels(images, labels, classnames, start_index=0):
    fig, ax = plt.subplots(ncols=4, nrows=2)

    index = start_index
    for i in range(8):
        ax[i].imshow(images[index], cmap='gray')
        ax[i].set_title(classnames[labels[index]])
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        index += 1
    plt.show()
    
def crop_images(images, size):
    sc = smartcrop.SmartCrop()
    cropped_images = []
    
    with alive_bar(len(images), bar='smooth', spinner=None) as bar:
        for image in images:
            image = np.array(image)
            image = Image.fromarray(image)
            localized = sc.crop(image, size, size)
        
            bbox = (
                localized['top_crop']['x'],
                localized['top_crop']['y'],
                localized['top_crop']['width'] + localized['top_crop']['x'],
                localized['top_crop']['height'] + localized['top_crop']['y']
            )
        
            cropped = image.crop(bbox)
            resized = cropped.resize((size, size))
            cropped_images.append(np.array(resized))
            bar()
    
    return np.array(cropped_images)
    
def preprocess_images(images):
    preprocessed_images = []
    
    with alive_bar(len(images), bar='smooth', spinner=None) as bar:
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, dsize=(150, 150))
            # image = image / 255.
            preprocessed_images.append(image)
            bar()
            
    return preprocessed_images

def find_misclassifications(labels, preds):
    indices = []
    for i, (label, pred) in enumerate(zip(preds, labels)):
        if pred != label:
            indices.append(i)

    return np.array(indices)

def show_misclassifications(images, misclassified, labels, preds, start_index=0):
    fig, axes = plt.subplots(ncols=7, nrows=2, figsize=(18, 6))
    
    classnames = ['healthy', 'wssv']
    index = start_index
    for i in range(2):
        for j in range(7):
            axes[i, j].imshow(images[misclassified[index]], cmap='gray')
            axes[i, j].set_title(f'actual: {classnames[labels[misclassified[index]]]} \n'
                                 f'pred: {classnames[preds[misclassified[index]]]}')
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
            if index == (index+14):
                break
            index += 1
    plt.show()



