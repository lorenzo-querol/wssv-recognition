#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 21:51:52 2023

@author: lorenzoquerol
"""

from data_augmentation import augment_images
from feature_extraction import extract_lbp, create_histograms, extract_glcm_noloop
from utils import load_images, show_raw_images, show_images_with_labels, crop_images, preprocess_images
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy as cp
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFECV

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline

import seaborn as sns
sns.set_theme(style="ticks")

plt.rcParams['figure.dpi'] = 600


random_state = 16
# %% Data Loading and Cropping

main_dir = 'wssv-dataset/train'
main_dir2 = 'wssv-dataset'
classnames = ['healthy', 'wssv']

print("\nLoading images...")
class_0 = load_images(f'{main_dir}/healthy')
class_1 = load_images(f'{main_dir}/wssv')

class_0_num_samples = len(class_0)
class_1_num_samples = len(class_1)

print("\nCropping images...")
class_0_cropped = crop_images(class_0, 150)
class_1_cropped = crop_images(class_1, 150)

show_raw_images(class_0_cropped, classnames[0])
show_raw_images(class_1_cropped, classnames[1])

print("\nPreprocessing images...")
class_0_preprocessed = preprocess_images(class_0_cropped)
class_1_preprocessed = preprocess_images(class_1_cropped)

labels = np.array([0] * class_0_num_samples + [1] * class_1_num_samples)
all_images = np.vstack((class_0_preprocessed, class_1_preprocessed))
all_images = list(zip(all_images, labels))

x_train, x_test, y_train, y_test = train_test_split(all_images,
                                                    labels,
                                                    test_size = 0.3,
                                                    random_state = random_state)

x_train, y_train = augment_images(x_train, 
                                  y_train, 
                                  num_aug = 5)
# %% Exploratory Data Analysis - Class Distribution

num_per_class = {'healthy': class_0_num_samples,
                 'wssv': class_1_num_samples}
idx_to_class = {0: 'healthy', 1: 'wssv'}

plt.bar(num_per_class.keys(), num_per_class.values())
plt.title("Number of Images by Class")
plt.xlabel('Class Name')
plt.ylabel('# of Images')
plt.show()

# %% Image Preprocessing and Feature Extraction

x_train_images = [i for i, j in x_train]
x_test_images = [i for i, j in x_test]

x_train_lbp = extract_lbp(x_train_images)
x_test_lbp = extract_lbp(x_test_images)

print("\nCreating Histograms...")
x_train_lbp_hist = create_histograms(x_train_lbp,
                                     sub_images_num = 3,
                                     bins_per_sub_images = 64)

x_test_lbp_hist = create_histograms(x_test_lbp,
                                    sub_images_num = 3,
                                    bins_per_sub_images = 64)

# %% Helper Functions

def cross_val_predict(model, kfold: StratifiedKFold, X: np.array, y: np.array):

    model_ = cp.deepcopy(model)
    
    train_actual = np.empty([0], dtype=int)
    train_pred = np.empty([0], dtype=int)
    
    val_actual = np.empty([0], dtype=int)
    val_pred = np.empty([0], dtype=int)
    
    for train_idx, test_idx in kfold.split(X, y):

        x_train, y_train, x_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

        model_.fit(x_train, y_train)
        
        train_actual = np.append(train_actual, y_train)
        train_pred = np.append(train_pred, model_.predict(x_train))
        
        val_actual = np.append(val_actual, y_test)
        val_pred = np.append(val_pred, model_.predict(x_test))

    return train_actual, train_pred, val_actual, val_pred, model_

def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list, classnames:list):
    cm = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, 
                                  display_labels = classnames)
    
    disp.plot()
    plt.show()
    
    return cm
    
def compute_fnr(cm):
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    FNR = FN / (TP + FN)
    
    return FNR

# %% Logistic Regression

pipeline = Pipeline(steps=[('s', StandardScaler()),
                           ('m', SVC(kernel='sigmoid', random_state=random_state))
                           ])

cv = StratifiedKFold(n_splits = 5,
                      shuffle = True,
                      random_state = random_state)

train_actual, train_pred, val_actual, val_pred, model = cross_val_predict(pipeline, 
                                                                   cv, 
                                                                   x_train_lbp_hist, 
                                                                   y_train)

train_cm = plot_confusion_matrix(train_actual, train_pred, [0, 1], classnames)
val_cm = plot_confusion_matrix(val_actual, val_pred, [0, 1], classnames)

train_acc = balanced_accuracy_score(train_actual, train_pred)
train_f1 = f1_score(train_actual, train_pred, average='weighted')
train_fnr = compute_fnr(train_cm)

val_acc = balanced_accuracy_score(train_actual, train_pred)
val_f1 = f1_score(val_actual, val_pred, average='weighted')
val_fnr = compute_fnr(val_cm)

test_pred = model.predict(x_test_lbp_hist)
test_cm = plot_confusion_matrix(y_test, test_pred, [0, 1], classnames)

test_acc = balanced_accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average='weighted')
test_fnr = compute_fnr(test_cm)

print("\nWithout SMOTE and Tomek-Links")
print('\nTraining')
print('Accuracy: %.2f' % train_acc)
print('F1 Score: %.2f' % train_f1)
print('FNR: %.2f' % train_fnr[1])

print('\nValidation')
print('Accuracy: %.2f' % val_acc)
print('F1 Score: %.2f' % val_f1)
print('FNR: %.2f' % val_fnr[1])

print('\nTest')
print('Accuracy: %.2f' % test_acc)
print('F1 Score: %.2f' % test_f1)
print('FNR: %.2f' % test_fnr[1])

pipeline = Pipeline(steps=[('s', StandardScaler()),
                           ('r', SMOTETomek(tomek = TomekLinks(sampling_strategy = 'majority'),
                                            smote = SMOTE(sampling_strategy = 'minority'))),
                           ('m', SVC(kernel='sigmoid', random_state=random_state))
                          ])

train_actual, train_pred, val_actual, val_pred, model = cross_val_predict(pipeline, 
                                                                   cv, 
                                                                   x_train_lbp_hist, 
                                                                   y_train)

train_cm = plot_confusion_matrix(train_actual, train_pred, [0, 1], classnames)
val_cm = plot_confusion_matrix(val_actual, val_pred, [0, 1], classnames)

train_acc = balanced_accuracy_score(train_actual, train_pred)
train_f1 = f1_score(train_actual, train_pred, average='weighted')
train_fnr = compute_fnr(train_cm)

val_acc = balanced_accuracy_score(train_actual, train_pred)
val_f1 = f1_score(val_actual, val_pred, average='weighted')
val_fnr = compute_fnr(val_cm)

test_pred = model.predict(x_test_lbp_hist)
test_cm = plot_confusion_matrix(y_test, test_pred, [0, 1], classnames)

test_acc = balanced_accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average='weighted')
test_fnr = compute_fnr(test_cm)

print("\nWith SMOTE and Tomek-Links")
print('\nTraining')
print('Accuracy: %.2f' % train_acc)
print('F1 Score: %.2f' % train_f1)
print('FNR: %.2f' % train_fnr[1])

print('\nValidation')
print('Accuracy: %.2f' % val_acc)
print('F1 Score: %.2f' % val_f1)
print('FNR: %.2f' % val_fnr[1])

print('\nTest')
print('Accuracy: %.2f' % test_acc)
print('F1 Score: %.2f' % test_f1)
print('FNR: %.2f' % test_fnr[1])