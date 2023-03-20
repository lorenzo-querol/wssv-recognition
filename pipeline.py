# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 08:49:42 2023

@author: Enzo
"""

from data_augmentation import augment_images
from feature_extraction import extract_lbp, create_histograms, extract_glcm_noloop
from utils import load_images, show_raw_images, show_images_with_labels, crop_images, preprocess_images
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
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

x_train, y_train = augment_images(x_train, y_train, num_aug=5)
# %% Exploratory Data Analysis - Class Distribution

num_per_class = {'healthy': class_0_num_samples,
                 'wssv': class_1_num_samples}
idx_to_class = {0: 'healthy', 1: 'wssv'}

plt.bar(num_per_class.keys(), num_per_class.values())
plt.title("Number of Images by Class")
plt.xlabel('Class Name')
plt.ylabel('# of Images')
plt.show()

# %% Exploratory Data Analysis - Resolution Distribution

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
y_height_class0 = []
x_width_class0 = []
labels_class0 = []
for image in class_0:
    h, w, _ = image.shape
    y_height_class0.append(h)
    x_width_class0.append(w)
    labels_class0.append('healthy')
ax[0].scatter(x_width_class0, y_height_class0)
ax[0].set_ylabel('Height (pixels)')
ax[0].set_xlabel('Width (pixels)')
ax[0].set_title('Image Sizes of Class 0 | healthy')

y_height_class1 = []
x_width_class1 = []
labels_class1 = []
for image in class_1:
    h, w, _ = image.shape
    y_height_class1.append(h)
    x_width_class1.append(w)
    labels_class1.append('wssv')

res_arr = np.column_stack((
    np.hstack((labels_class0, labels_class1)),
    np.hstack((x_width_class0, x_width_class1)),
    np.hstack((y_height_class0, y_height_class1)),
))

res_df = pd.DataFrame(res_arr, columns = ['class', 'width', 'height'])

ax[1].scatter(x_width_class1, y_height_class1)
ax[1].set_ylabel('Height (pixels)')
ax[1].set_xlabel('Width (pixels)')
ax[1].set_title('Image Sizes of Class 1 | wssv')
plt.show()

# %% Image Preprocessing and Feature Extraction

x_train_images = [i for i, j in x_train]
x_test_images = [i for i, j in x_test]

x_train_lbp = extract_lbp(x_train_images)
x_test_lbp = extract_lbp(x_test_images)

# show_images_with_labels(x_train_lbp, labels, classnames)

print("\nCreating Histograms...")
x_train_lbp_hist = create_histograms(x_train_lbp,
                                     sub_images_num = 3,
                                     bins_per_sub_images = 64)

x_test_lbp_hist = create_histograms(x_test_lbp,
                                    sub_images_num = 3,
                                    bins_per_sub_images = 64)

# %% Create Dataframe for features

# features_with_labels = np.column_stack((labels, lbp_hist))
# columns = ['class'] + [f'sub_image_{x}' for x in range(features_with_labels.shape[1]-1)]
# features_df = pd.DataFrame(features_with_labels, columns=columns)

# %% Logistic Regression

pipeline = Pipeline(steps=[('s', StandardScaler()),
                           ('m', SVC())
                           ])

cv = StratifiedKFold(n_splits = 5,
                     shuffle = True,
                     random_state = random_state)

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_weighted']

scores = cross_validate(pipeline,
                        x_train_lbp_hist,
                        y_train,
                        scoring = scoring,
                        cv = cv,
                        n_jobs = -1,
                        return_train_score = True)

print("\nWithout SMOTE and Tomek-Links")
print('\nTraining')
print('Accuracy: %.4f' % np.mean(scores['train_accuracy']))
print('Precision: %.4f' % np.mean(scores['train_precision_macro']))
print('Recall: %.4f' % np.mean(scores['train_recall_macro']))
print('F1 Score: %.4f' % np.mean(scores['train_f1_weighted']))

print('\nValidation')
print('\nAccuracy: %.4f' % np.mean(scores['test_accuracy']))
print('Precision: %.4f' % np.mean(scores['test_precision_macro']))
print('Recall: %.4f' % np.mean(scores['test_recall_macro']))
print('F1 Score: %.4f' % np.mean(scores['test_f1_weighted']))

pipeline.fit(x_train_lbp_hist, y_train)
y_preds_svm = pipeline.predict(x_test_lbp_hist)

score_svm_test = np.round(pipeline.score(x_test_lbp_hist, y_test) * 100, 2)
f1_svm = f1_score(y_test, y_preds_svm, average='weighted')

cm = confusion_matrix(y_test, 
                      y_preds_svm, 
                      labels = [0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix = cm, 
                              display_labels = classnames)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FNR = FN/(TP+FN)

print('\nTest')
print('Accuracy: %.4f' % score_svm_test)
print('F1 score: %.4f' % f1_svm)
print('False Negative Rate (Healthy): %.4f' % FNR[0])
print('False Negative Rate (WSSV): %.4f' % FNR[1])

disp.plot()
plt.show()

pipeline = Pipeline(steps=[('s', StandardScaler()),
                           ('r', SMOTETomek(tomek = TomekLinks(sampling_strategy = 'majority'),
                                            smote = SMOTE(sampling_strategy = 'minority'))),
                           ('m', SVC())
                           ])

cv = StratifiedKFold(n_splits = 5,
                     shuffle = True,
                     random_state = random_state)

scores = cross_validate(pipeline,
                        x_train_lbp_hist,
                        y_train,
                        scoring = scoring,
                        cv = cv,
                        n_jobs = -1,
                        return_train_score = True)

print("\nAfter Applying SMOTE and Tomek-Links")
print('\nTraining')
print('Accuracy: %.4f' % np.mean(scores['train_accuracy']))
print('Precision: %.4f' % np.mean(scores['train_precision_macro']))
print('Recall: %.4f' % np.mean(scores['train_recall_macro']))
print('F1 Score: %.4f' % np.mean(scores['train_f1_weighted']))

print('\nValidation')
print('\nAccuracy: %.4f' % np.mean(scores['test_accuracy']))
print('Precision: %.4f' % np.mean(scores['test_precision_macro']))
print('Recall: %.4f' % np.mean(scores['test_recall_macro']))
print('F1 Score: %.4f' % np.mean(scores['test_f1_weighted']))

pipeline.fit(x_train_lbp_hist, y_train)
y_preds_svm = pipeline.predict(x_test_lbp_hist)

score_svm_test = np.round(pipeline.score(x_test_lbp_hist, y_test) * 100, 2)
f1_svm = f1_score(y_test, y_preds_svm, average='weighted')

cm = confusion_matrix(y_test, 
                      y_preds_svm, 
                      labels = [0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix = cm, 
                              display_labels = classnames)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FNR = FN/(TP+FN)

print('\nTest')
print('Accuracy: %.4f' % score_svm_test)
print('F1 score: %.4f' % f1_svm)
print('False Negative Rate (Healthy): %.4f' % FNR[0])
print('False Negative Rate (WSSV): %.4f' % FNR[1])

disp.plot()
plt.show()
