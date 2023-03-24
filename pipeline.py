# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:53:54 2023

@author: Enzo
"""
# %%

from ../functions/data_augmentation import augment_images
from feature_extraction import extract_lbp, create_histograms, extract_glcm_noloop, split_image
from utils import load_images, show_raw_images, crop_images, preprocess_images

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
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

sns.set_theme(style="ticks")
plt.rcParams['figure.dpi'] = 100
random_state = 42

# %% Data Loading

main_dir = 'cropped'
classnames = ['healthy', 'wssv']

class_0 = load_images(f'{main_dir}/healthy')
class_1 = load_images(f'{main_dir}/wssv')

class_0 = preprocess_images(class_0)
class_1 = preprocess_images(class_1)

class_0_num_samples = len(class_0)
class_1_num_samples = len(class_1)

labels = np.array([0] * class_0_num_samples + [1] * class_1_num_samples)

all_images = np.vstack((class_0, class_1))
all_images = list(zip(all_images, labels))

x_train, x_test, y_train, y_test = train_test_split(all_images,
                                                    labels,
                                                    test_size=0.3,
                                                    random_state=random_state)

x_train, y_train = augment_images(x_train, y_train, 5)

# %% Exploratory Data Analysis - Class Distribution

df = pd.DataFrame(all_images, columns=['Image', 'Class'])
df['Class'] = df['Class'].map({0: 'healthy', 1: 'wssv'})
df['Class'].value_counts().plot(kind='bar')

plt.title("Distribution of Classes")
plt.xlabel('Class Name')
plt.ylabel('# of Images')
plt.show()

# %% Image Preprocessing and Feature Extraction

x_train_images = [i for i, j in x_train]
x_test_images = [i for i, j in x_test]

x_train_lbp = extract_lbp(x_train_images)
x_test_lbp = extract_lbp(x_test_images)

# %% Create Histogram from LBP Images

x_train_lbp_hist = create_histograms(x_train_lbp,
                                     sub_images_num=3,
                                     bins_per_sub_images=64)

x_test_lbp_hist = create_histograms(x_test_lbp,
                                    sub_images_num=3,
                                    bins_per_sub_images=64)

# %% Helper function


def evaluate(pipeline):
    cv = StratifiedKFold(n_splits=10,
                         shuffle=True,
                         random_state=random_state)

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_weighted']

    scores = cross_validate(pipeline,
                            x_train_lbp_hist,
                            y_train,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=-1,
                            return_train_score=True,
                            return_estimator=True)

    print('\nTraining')
    print('Accuracy: %.4f' % max(scores['train_accuracy']))
    print('Precision: %.4f' % max(scores['train_precision_macro']))
    print('Recall: %.4f' % max(scores['train_recall_macro']))
    print('F1 Score: %.4f' % max(scores['train_f1_weighted']))

    print('\nValidation')
    print('Accuracy: %.4f' % max(scores['test_accuracy']))
    print('Precision: %.4f' % max(scores['test_precision_macro']))
    print('Recall: %.4f' % max(scores['test_recall_macro']))
    print('F1 Score: %.4f' % max(scores['test_f1_weighted']))

    pipeline.fit(x_train_lbp_hist, y_train)
    y_preds_svm = pipeline.predict(x_test_lbp_hist)

    test_acc = pipeline.score(x_test_lbp_hist, y_test)
    test_f1 = f1_score(y_test, y_preds_svm)

    cm = confusion_matrix(y_test, y_preds_svm, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classnames)

    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    FNR = FN / (TP + FN)

    print('\nTest')
    print('Accuracy: %.4f' % test_acc)
    print('F1 score: %.4f' % test_f1)
    print('False Negative Rate (Healthy): %.4f' % FNR[0])
    print('False Negative Rate (WSSV): %.4f' % FNR[1])

    disp.plot()
    plt.show()

# %% Classification


scaler = StandardScaler()

classifier1 = SVC(random_state=random_state)
pipeline1 = Pipeline(steps=[('scaler', scaler),
                            ('classifier', classifier1)
                            ])

classifier2 = SVC(random_state=random_state)
sampler = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'),
                     smote=SMOTE(sampling_strategy='minority')
                     )

pipeline2 = Pipeline(steps=[('scaler', scaler),
                            ('sampler', sampler),
                            ('classifier', classifier2)
                            ])

print("\nWithout SMOTE and Tomek-Links")
evaluate(pipeline1)

print("\nAfter Applying SMOTE and Tomek-Links")
evaluate(pipeline2)
