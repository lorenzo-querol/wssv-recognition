# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 08:53:55 2023

@author: Enzo
"""
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
import pandas as pd
import seaborn as sns
sns.set_theme(style="ticks")


plt.rcParams['figure.dpi'] = 600


def extract_lbp(images):
    lbps = []
    with alive_bar(len(images), title="Extracting LBP", bar='smooth', spinner=None) as bar:
        for image in images:
            lbp = local_binary_pattern(image, P=8, R=2)
            lbps.append(lbp)
            bar()

    return np.array(lbps)


def split_image(image, sub_images_num, bins_per_sub_images):

    fig, ax = plt.subplots()

    grid = np.arange(
        0, image.shape[1]+1, image.shape[1]//sub_images_num)

    sub_image_histograms = []

    for i in range(1, len(grid)):
        for j in range(1, len(grid)):
            sub_image = image[grid[i-1]:grid[i], grid[j-1]:grid[j]]

            sub_image_histogram = np.histogram(sub_image,
                                               bins=bins_per_sub_images)[0]

            sub_image_histograms.append(sub_image_histogram)

    histogram = np.array(sub_image_histograms).flatten()
    ax.hist(histogram)
    plt.axis('off')
    plt.show()


def create_histograms(images, sub_images_num, bins_per_sub_images):
    all_histograms = []

    # fig, ax = plt.subplots(3, 3)

    with alive_bar(len(images), title="Creating Histograms", bar='smooth', spinner=None) as bar:
        for image in images:
            grid = np.arange(
                0, image.shape[1]+1, image.shape[1]//sub_images_num)

            sub_image_histograms = []

            # temp = {}
            # temp['image'] = [image]
            # temp['class'] = label
            # ctr = 1

            for i in range(1, len(grid)):
                for j in range(1, len(grid)):
                    sub_image = image[grid[i-1]:grid[i], grid[j-1]:grid[j]]

                    # temp[f'sub_image_{ctr}'] = [sub_image]
                    # ctr += 1

                    sub_image_histogram = np.histogram(
                        sub_image, bins=bins_per_sub_images)[0]
                    sub_image_histograms.append(sub_image_histogram)

            histogram = np.array(sub_image_histograms).flatten()

            # temp = pd.DataFrame(temp)

            # sub_images_df = pd.concat([sub_images_df, temp], axis = 1)

            all_histograms.append(histogram)
            bar()

    return np.array(all_histograms)


def extract_glcm(images,
                 sub_images_num,
                 dists=[5],
                 angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                 lvl=256,
                 sym=True,
                 norm=True):

    props = ['dissimilarity', 'correlation',
             'homogeneity', 'contrast', 'ASM', 'energy']
    glcms = []
    with alive_bar(len(images), bar='smooth', spinner=None) as bar:
        for image in images:

            grid = np.arange(
                0, image.shape[1]+1, image.shape[1]//sub_images_num)

            image_features = []
            sub_image_features = []
            for i in range(1, len(grid)):
                for j in range(1, len(grid)):
                    sub_image = image[grid[i-1]: grid[i], grid[j-1]: grid[j]]

                    glcm = graycomatrix(sub_image,
                                        distances=dists,
                                        angles=angles,
                                        levels=lvl,
                                        symmetric=sym,
                                        normed=norm)

                    glcm_props = [prop for name in props for
                                  prop in graycoprops(glcm, name)[0]]

                    for item in glcm_props:
                        sub_image_features.append(item)

                    image_features.append(sub_image_features)

            features = np.array(image_features).flatten()

            glcms.append(features)
            bar()

    return np.array(glcms)


def extract_glcm_noloop(images,
                        dists=[5],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        lvl=256,
                        sym=True,
                        norm=True):

    props = ['dissimilarity', 'correlation',
             'homogeneity', 'contrast', 'ASM', 'energy']

    glcms = []
    with alive_bar(len(images), bar='smooth', spinner=None) as bar:
        for image in images:
            channel_features = []
            for c in range(3):
                features = []
                glcm = graycomatrix(image[:, :, c],
                                    distances=dists,
                                    angles=angles,
                                    levels=lvl,
                                    symmetric=sym,
                                    normed=norm)

                glcm_props = [prop for name in props for
                              prop in graycoprops(glcm, name)[0]]

                for item in glcm_props:
                    features.append(item)

                channel_features.append(features)

            glcms.append(channel_features)
            bar()

    return np.array(glcms)
