#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 10:44:33 2023

@author: lorenzoquerol
"""

from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern

import seaborn as sns

from alive_progress import alive_bar
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 10)


def load_images(path):
    images = []
    filenames = os.listdir(path)

    with alive_bar(len(filenames), bar='smooth', spinner=None) as bar:
        for filename in filenames:
            image = cv2.imread(os.path.join(path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            bar()

    return np.array(images)

def show_raw_images(images, classname, start_index=0):
    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(16, 2.5))
    plt.suptitle(classname)

    index = start_index
    for i in range(10):
        axes[i].imshow(images[index])
        axes[i].set_title(images[index].shape)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        index += 1
    plt.show()
    
def preprocess_images(images):
    preprocessed_images = []
    with alive_bar(len(images), bar='smooth', spinner=None) as bar:
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, dsize=(150, 150))
            image = image / 255.
            preprocessed_images.append(image)
            bar()

    return np.array(preprocessed_images)

def extract_lbp(images):
    lbps = []
    with alive_bar(len(images), bar='smooth', spinner=None) as bar:
        for image in images:
            lbp = local_binary_pattern(image, P=8, R=1)
            lbps.append(lbp)
            bar()

    return np.array(lbps)

def create_histograms(images, sub_images_num, bins_per_sub_images):
    all_histograms = []

    with alive_bar(len(images), bar='smooth', spinner=None) as bar:
        for image in images:
            grid = np.arange(
                0, image.shape[1]+1, image.shape[1]//sub_images_num)

            sub_image_histograms = []

            for i in range(1, len(grid)):
                for j in range(1, len(grid)):
                    sub_image = image[grid[i-1]:grid[i], grid[j-1]:grid[j]]

                    sub_image_histogram = np.histogram(
                        sub_image, bins=bins_per_sub_images)[0]
                    sub_image_histograms.append(sub_image_histogram)

            histogram = np.array(sub_image_histograms).flatten()

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
                    sub_image = image[grid[i-1]:grid[i], grid[j-1]:grid[j]]

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

def find_misclassifications(labels, preds):
    indices = []
    for i, (label, pred) in enumerate(zip(preds, labels)):
        if pred != label:
            indices.append(i)

    return np.array(indices)

def show_misclassifications(images, misclassified, labels, preds, start_index=0):
    fig, axes = plt.subplots(ncols=7, nrows=2, figsize=(18, 6))

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


"""
Main Driver
"""
main_dir = 'wssv-dataset/train'
classnames = ['healthy', 'wssv']

print("\nLoading images...")
class_0 = load_images(f'{main_dir}/healthy')
class_1 = load_images(f'{main_dir}/wssv')

# show_raw_images(class_0, classnames[0])
# show_raw_images(class_1, classnames[1])

"""
Exploratory Data Analysis - Class Distribution
"""

class_0_num_samples = len(class_0)
class_1_num_samples = len(class_1)
num_per_class = {'healthy': class_0_num_samples, 
                 'wssv': class_1_num_samples}

plt.bar(num_per_class.keys(), num_per_class.values());
plt.title("Number of Images by Class");
plt.xlabel('Class Name');
plt.ylabel('# of Images');
plt.show()

"""
Exploratory Data Analysis - Resolution Distribution 
"""

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
y_height = []
x_width = []
for image in class_0:
    h, w, _ = image.shape
    y_height.append(h)
    x_width.append(w)
    
ax[0].scatter(x_width, y_height,)
ax[0].set_ylabel('height (pixels)')
ax[0].set_xlabel('width (pixels)')
ax[0].set_title('Image Sizes of Class 0 | healthy')

y_height = []
x_width = []
for image in class_1:
    h, w, _ = image.shape
    y_height.append(h)
    x_width.append(w)
    
ax[1].scatter(x_width, y_height)
ax[1].set_ylabel('height (pixels)')
ax[1].set_xlabel('width (pixels)')
ax[1].set_title('Image Sizes of Class 1 | wssv')
plt.show()

"""
Image Preprocessing and Feature Extraction
"""
print("\nPreprocessing images...")
class_0_preprocessed = preprocess_images(class_0)
class_1_preprocessed = preprocess_images(class_1)
all_images = np.vstack((class_0_preprocessed, class_1_preprocessed))
labels = np.array([0]*class_0_num_samples + [1]*class_1_num_samples)

X_train, X_test, y_train, y_test = train_test_split(all_images,
                                                    labels,
                                                    test_size=0.2,
                                                    shuffle=True)

print("\nExtracting LBP...")
X_train_lbp = extract_lbp(X_train)
X_test_lbp = extract_lbp(X_test)

# print("\nExtracting GLCM...")
# X_train_glcm = extract_glcm(X_train, 3)
# X_test_glcm = extract_glcm(X_test, 3)

print("\nCreating Histograms...")
X_train_lbp_hist = create_histograms(
    X_train_lbp, sub_images_num=3, bins_per_sub_images=64)
X_test_lbp_hist = create_histograms(
    X_test_lbp, sub_images_num=3, bins_per_sub_images=64)

features_train = X_train_lbp_hist
features_test = X_test_lbp_hist

all_features = np.vstack((features_train, features_test))
all_features = np.column_stack((labels, all_features))
all_features.tofile('features.csv', sep = ',')

scaler = StandardScaler()

"""
Logistic Regression
"""
model_logreg = make_pipeline(StandardScaler(), 
                             LogisticRegression())
model_logreg.fit(features_train, y_train)
y_preds_logreg = model_logreg.predict(features_test)

score_logreg_train = np.round(model_logreg.score(features_train, y_train) * 100, 2)
score_logreg_test = np.round(model_logreg.score(features_test, y_test) * 100, 2)
f1_logreg = np.round(f1_score(y_test, y_preds_logreg) * 100, 2)

print(f'\nLogistic regression train accuracy: {score_logreg_train}%')
print(f'Logistic regression test accuracy: {score_logreg_test}%')
print(f'Logistic Regression F1 score: {f1_logreg}%')    

cm = confusion_matrix(y_test, y_preds_logreg, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('Logistic Regression Confusion Matrix')
plt.show()


"""
Support Vector Machine
"""
model_svm = make_pipeline(StandardScaler(), 
                          SVC())
model_svm.fit(features_train, y_train)
y_preds_svm = model_svm.predict(features_test)

score_svm_train = np.round(model_svm.score(features_train, y_train) * 100, 2)
score_svm_test = np.round(model_svm.score(features_test, y_test) * 100, 2)
f1_svm = np.round(f1_score(y_test, y_preds_svm) * 100, 2)

cm = confusion_matrix(y_test, y_preds_svm, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('SVM Confusion Matrix')
plt.show()

print(f'\nSVM train accuracy: {score_svm_train}%')
print(f'SVM test accuracy: {score_svm_test}%')
print(f'SVM F1 score: {f1_svm}%')


"""
K-Nearest Neighbor
"""
model_knn = make_pipeline(StandardScaler(), 
                          KNeighborsClassifier(n_neighbors=1))
model_knn.fit(features_train, y_train)
y_preds_knn = model_knn.predict(features_test)

score_knn_train = np.round(model_knn.score(features_train, y_train) * 100, 2)
score_knn_test = np.round(model_knn.score(features_test, y_test) * 100, 2)
f1_knn = np.round(f1_score(y_test, y_preds_knn) * 100, 2)

cm = confusion_matrix(y_test, y_preds_knn, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('KNN Confusion Matrix')
plt.show()

print(f'\nKNN train accuracy: {score_knn_train}%')
print(f'KNN test accuracy: {score_knn_test}%')
print(f'KNN F1 score: {f1_knn}%')

"""
Random Forest
"""
model_rf = make_pipeline(StandardScaler(), 
                         RandomForestClassifier())
model_rf.fit(features_train, y_train)
y_preds_rf = model_rf.predict(features_test)

score_rf_train = np.round(model_rf.score(features_train, y_train) * 100, 2)
score_rf_test = np.round(model_rf.score(features_test, y_test) * 100, 2)
f1_rf = np.round(f1_score(y_test, y_preds_rf) * 100, 2)

print(f'\nRandom Forest train accuracy: {score_rf_train}%')
print(f'Random Forest test accuracy: {score_rf_test}%')
print(f'Random Forest F1 score: {f1_rf}%')

cm = confusion_matrix(y_test, y_preds_rf, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('Random Forest Confusion Matrix')
plt.show()


"""
Decision Tree
"""
model_dt = make_pipeline(StandardScaler(), 
                         DecisionTreeClassifier())
model_dt.fit(features_train, y_train)
y_preds_dt = model_dt.predict(features_test)

score_dt_train = np.round(model_dt.score(features_train, y_train) * 100, 2)
score_dt_test = np.round(model_dt.score(features_test, y_test) * 100, 2)
f1_dt = np.round(f1_score(y_test, y_preds_dt) * 100, 2)

print(f'\nDecision Tree train accuracy: {score_dt_train}%')
print(f'Decision Tree test accuracy: {score_dt_test}%')
print(f'Decision Tree F1 score: {f1_dt}%')

cm = confusion_matrix(y_test, y_preds_dt, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('Decision Tree Confusion Matrix')
plt.show()

# misclassifications = find_misclassifications(y_test, predictions)

# show_misclassifications(X_test, misclassifications, y_test, predictions)
