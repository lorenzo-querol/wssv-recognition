# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 08:49:42 2023

@author: Enzo
"""

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
from sklearn.model_selection import LearningCurveDisplay

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline

import seaborn as sns
sns.set_theme(style="ticks")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600

from utils import load_images, show_raw_images, crop_images, preprocess_images
from feature_extraction import extract_lbp, create_histograms, extract_glcm_noloop
from data_augmentation import augment_images

random_state = 42
#%% Data Loading and Cropping

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

x_train, y_train = augment_images(x_train, y_train)
#%% Exploratory Data Analysis - Class Distribution

num_per_class = {'healthy': class_0_num_samples, 
                 'wssv': class_1_num_samples}
idx_to_class = {0: 'healthy', 1: 'wssv'}

plt.bar(num_per_class.keys(), num_per_class.values());
plt.title("Number of Images by Class");
plt.xlabel('Class Name');
plt.ylabel('# of Images');
plt.show()

#%% Exploratory Data Analysis - Resolution Distribution 

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
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

res_df = pd.DataFrame(res_arr, columns=['class', 'width', 'height'])

ax[1].scatter(x_width_class1, y_height_class1)
ax[1].set_ylabel('Height (pixels)')
ax[1].set_xlabel('Width (pixels)')
ax[1].set_title('Image Sizes of Class 1 | wssv')
plt.show()

#%% Image Preprocessing and Feature Extraction

x_test_images = [i for i, j in x_test]

x_train_lbp = extract_lbp(x_train[:, 0])
x_test_lbp = extract_lbp(x_test_images)
    
print("\nCreating Histograms...")
x_train_lbp_hist = create_histograms(x_train_lbp, 
                                     sub_images_num = 3, 
                                     bins_per_sub_images = 64)

x_test_lbp_hist = create_histograms(x_test_lbp, 
                                    sub_images_num = 3, 
                                    bins_per_sub_images = 64)

#%% Create Dataframe for features 

# features_with_labels = np.column_stack((labels, lbp_hist))
# columns = ['class'] + [f'sub_image_{x}' for x in range(features_with_labels.shape[1]-1)]
# features_df = pd.DataFrame(features_with_labels, columns=columns)

#%% Logistic Regression

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
print('Mean Training Accuracy: %.4f' % np.mean(scores['train_accuracy']))
print('Mean Training Precision: %.4f' % np.mean(scores['train_precision_macro']))
print('Mean Training Recall: %.4f' % np.mean(scores['train_recall_macro']))
print('Mean Training F1 Score: %.4f' % np.mean(scores['train_f1_weighted']))

print('\nMean Validation Accuracy: %.4f' % np.mean(scores['test_accuracy']))
print('Mean Validation Precision: %.4f' % np.mean(scores['test_precision_macro']))
print('Mean Validation Recall: %.4f' % np.mean(scores['test_recall_macro']))
print('Mean Validation F1 Score: %.4f' % np.mean(scores['test_f1_weighted']))

pipeline.fit(x_train_lbp_hist, y_train)
y_preds_svm = pipeline.predict(x_test_lbp_hist)
cm = confusion_matrix(y_test, y_preds_svm, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('SVM Confusion Matrix Without SMOTE and Tomek-Links')
plt.show()


pipeline = Pipeline(steps=[('s', StandardScaler()),
                           ('r', SMOTETomek(tomek = TomekLinks(sampling_strategy = 'majority'),
                                            smote = SMOTE(sampling_strategy = 'minority'))), 
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

print("\nAfter Applying SMOTE and Tomek-Links")
print('Mean Training Accuracy: %.4f' % np.mean(scores['train_accuracy']))
print('Mean Training Precision: %.4f' % np.mean(scores['train_precision_macro']))
print('Mean Training Recall: %.4f' % np.mean(scores['train_recall_macro']))
print('Mean Training F1 Score: %.4f' % np.mean(scores['train_f1_weighted']))

print('\nMean Validation Accuracy: %.4f' % np.mean(scores['test_accuracy']))
print('Mean Validation Precision: %.4f' % np.mean(scores['test_precision_macro']))
print('Mean Validation Recall: %.4f' % np.mean(scores['test_recall_macro']))
print('Mean Validation F1 Score: %.4f' % np.mean(scores['test_f1_weighted']))

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)

# common_params = {
#     "X": x_train,
#     "y": y_train,
#     "train_sizes": np.linspace(0.1, 1.0, 5),
#     "cv": StratifiedKFold(n_splits=5),
#     "score_type": "both",
#     "n_jobs": 4,
#     "line_kw": {"marker": "o"},
#     "std_display_style": "fill_between",
#     "score_name": "Accuracy",
# }

# LearningCurveDisplay.from_estimator(pipeline, **common_params)
# handles, label = ax[ax_idx].get_legend_handles_labels()
# ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
# ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")

# score_svm_train = np.round(model_svm.score(x_train, y_train) * 100, 2)
# score_svm_test = np.round(model_svm.score(x_test, y_test) * 100, 2)
# f1_svm = np.round(f1_score(y_test, y_preds_svm) * 100, 2)

# print(f'\nSVM train accuracy: {score_svm_train}%')
# print(f'SVM test accuracy: {score_svm_test}%')
# print(f'SVM F1 score: {f1_svm}%')    

pipeline.fit(x_train_lbp_hist, y_train)
y_preds_svm = pipeline.predict(x_test_lbp_hist)
cm = confusion_matrix(y_test, y_preds_svm, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('SVM Confusion Matrix With SMOTE and Tomek-Links')
plt.show()
