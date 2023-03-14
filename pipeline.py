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
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFECV
from sklearn.model_selection import GridSearchCV

import seaborn as sns
sns.set_theme(style="ticks")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_images, show_raw_images, crop_images, preprocess_images
from feature_extraction import extract_lbp, create_histograms, extract_glcm_noloop

#%% Data Loading and Cropping

main_dir = 'wssv-dataset/train'
classnames = ['healthy', 'wssv']

print("\nLoading images...")
class_0 = load_images(f'{main_dir}/healthy')
class_1 = load_images(f'{main_dir}/wssv')

print("\nCropping images...")
class_0_cropped = crop_images(class_0, 150)
class_1_cropped = crop_images(class_1, 150)

show_raw_images(class_0_cropped, classnames[0])
show_raw_images(class_1_cropped, classnames[1])

#%% Exploratory Data Analysis - Class Distribution

class_0_num_samples = len(class_0)
class_1_num_samples = len(class_1)

num_per_class = {'healthy': class_0_num_samples, 
                 'wssv': class_1_num_samples}
idx_to_class = {0: 'healthy', 1: 'wssv'}

plt.bar(num_per_class.keys(), num_per_class.values());
plt.title("Number of Images by Class");
plt.xlabel('Class Name');
plt.ylabel('# of Images');
plt.show()

#%% Exploratory Data Analysis - Resolution Distribution 

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

#%% Image Preprocessing and Feature Extraction

print("\nPreprocessing images...")
class_0_preprocessed, class_0_hsv = preprocess_images(class_0_cropped)
class_1_preprocessed, class_1_hsv = preprocess_images(class_1_cropped)
all_images = np.vstack((class_0_preprocessed, class_1_preprocessed))
all_hsv_images = np.vstack((class_0_hsv, class_1_hsv))

print("\nExtracting LBP...")
lbp_images = extract_lbp(all_images)

print("\nCreating Histograms...")
lbp_hist = create_histograms(lbp_images, 
                             sub_images_num = 3, 
                             bins_per_sub_images = 64)

print("\nExtracting GLCM...")
glcm_features = extract_glcm_noloop(all_hsv_images)

#%% Create Dataframe for features 

labels = np.array([0] * class_0_num_samples + [1] * class_1_num_samples)
all_features = np.column_stack((labels, lbp_hist))
columns = ['class'] + [f'sub_image_{x}' for x in range(all_features.shape[1]-1)]
features_df = pd.DataFrame(all_features, columns=columns)

#%% Model Training

x_train, x_test, y_train, y_test = train_test_split(lbp_hist,
                                                    labels,
                                                    test_size=0.3)

#%% Logistic Regression

model_logreg = make_pipeline(StandardScaler(), 
                              LogisticRegression())
model_logreg.fit(x_train, y_train)
y_preds_logreg = model_logreg.predict(x_test)

score_logreg_train = np.round(model_logreg.score(x_train, y_train) * 100, 2)
score_logreg_test = np.round(model_logreg.score(x_test, y_test) * 100, 2)
f1_logreg = np.round(f1_score(y_test, y_preds_logreg) * 100, 2)

print(f'\nLogistic regression train accuracy: {score_logreg_train}%')
print(f'Logistic regression test accuracy: {score_logreg_test}%')
print(f'Logistic Regression F1 score: {f1_logreg}%')    

cm = confusion_matrix(y_test, y_preds_logreg, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('Logistic Regression Confusion Matrix')
plt.show()

model_logreg = make_pipeline(StandardScaler(), 
                              LogisticRegression())

grid_values = {'logisticregression__penalty': ['l1', 'l2'],
               'logisticregression__C': [0.001, .009, 0.01, .09, 1 ,5, 10, 25],
               'logisticregression__solver': ['liblinear']}

grid_clf_acc = GridSearchCV(model_logreg, 
                            param_grid = grid_values, 
                            scoring = 'f1',
                            refit = True)
grid_clf_acc.fit(x_train, y_train)
y_preds_logreg = grid_clf_acc.predict(x_test)

score_logreg_train = np.round(grid_clf_acc.score(x_train, y_train) * 100, 2)
score_logreg_test = np.round(grid_clf_acc.score(x_test, y_test) * 100, 2)
f1_logreg = np.round(f1_score(y_test, y_preds_logreg) * 100, 2)

print('\nHyperparameter Optimized Logistic Regression')
print(f'Logistic regression train accuracy: {score_logreg_train}%')
print(f'Logistic regression test accuracy: {score_logreg_test}%')
print(f'Logistic Regression F1 score: {f1_logreg}%')    

cm = confusion_matrix(y_test, y_preds_logreg, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('Logistic Regression Confusion Matrix')
plt.show()

#%% SVM

model_svm = make_pipeline(StandardScaler(), 
                          SVC())
model_svm.fit(x_train, y_train)
y_preds_svm = model_svm.predict(x_test)

score_svm_train = np.round(model_svm.score(x_train, y_train) * 100, 2)
score_svm_test = np.round(model_svm.score(x_test, y_test) * 100, 2)
f1_svm = np.round(f1_score(y_test, y_preds_svm) * 100, 2)

print(f'\nSVM train accuracy: {score_svm_train}%')
print(f'SVM test accuracy: {score_svm_test}%')
print(f'SVM F1 score: {f1_svm}%')    

cm = confusion_matrix(y_test, y_preds_svm, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('SVM Confusion Matrix')
plt.show()

model_svm = make_pipeline(StandardScaler(), 
                          SVC())

grid_values = {'svc__C': [0.1, 1, 10, 100, 1000], 
              'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'svc__kernel': ['rbf', 'sigmoid']} 

grid_clf_acc = GridSearchCV(model_svm, 
                            param_grid = grid_values, 
                            scoring = 'f1',
                            refit = True)
grid_clf_acc.fit(x_train, y_train)
y_preds_svm = grid_clf_acc.predict(x_test)

score_svm_train = np.round(grid_clf_acc.score(x_train, y_train) * 100, 2)
score_svm_test = np.round(grid_clf_acc.score(x_test, y_test) * 100, 2)
f1_svm = np.round(f1_score(y_test, y_preds_svm) * 100, 2)

print('\nHyperparameter Optimized SVM')
print(f'SVM train accuracy: {score_svm_train}%')
print(f'SVM test accuracy: {score_svm_test}%')
print(f'SVM F1 score: {f1_svm}%')    

cm = confusion_matrix(y_test, y_preds_svm, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

disp.plot()
plt.title('SVM Confusion Matrix')
plt.show()

# f1_scores_kb = []
# for k in range(1, all_features.shape[1]):
#     best_k = SelectKBest(score_func=f_classif, k=k)
#     features_transformed = best_k.fit_transform(lbp_hist, labels)
    
#     train_x, test_x, train_y, test_y = train_test_split(features_transformed,
#                                                         labels,
#                                                         test_size=0.2,
#                                                         random_state=42)
    
#     model_svm = make_pipeline(StandardScaler(), LogisticRegression())
#     model_svm.fit(train_x, train_y)
#     y_pred = model_svm.predict(test_x)

#     f1 = f1_score(test_y, y_pred)
#     f1_scores_kb.append(f1)

# fig, ax = plt.subplots(dpi = 300) 
# ax.plot(range(1, all_features.shape[1]), f1_scores_kb)
# ax.set_xlabel("Best k features")
# ax.set_ylabel("F1-score")

# max_value = max(f1_scores_kb)
# k_value = f1_scores_kb.index(max_value) + 1

# f1_scores = []
# c_value = 30
# for c in range(1, c_value):
#     train_x, test_x, train_y, test_y = train_test_split(lbp_hist,
#                                                         labels,
#                                                         test_size=0.3,
#                                                         random_state=42)
    
#     model_svm = make_pipeline(StandardScaler(), 
#                               PCA(n_components=c),
#                               LogisticRegression())
    
#     model_svm.fit(train_x, train_y)
#     y_pred = model_svm.predict(test_x)

#     f1 = f1_score(test_y, y_pred)
#     f1_scores.append(f1)

# fig, ax = plt.subplots(dpi = 300) 
# ax.plot(range(1, c_value), f1_scores)
# ax.set_xlabel("Best k features")
# ax.set_ylabel("F1-score")


# cv_estimator = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
# X_train,X_test,Y_train,Y_test = train_test_split(lbp_hist, labels, test_size=0.2, random_state=42)
# cv_estimator.fit(X_train, Y_train)
# cv_selector = RFECV(cv_estimator, cv= 5, step=1, scoring='f1')
# cv_selector = cv_selector.fit(X_train, Y_train)
# rfecv_mask = cv_selector.get_support() #list of booleans
# rfecv_features = [] 
# for bool, feature in zip(rfecv_mask, X_train.columns):
#     if bool:
#         rfecv_features.append(feature)
# print('Optimal number of features :', cv_selector.n_features_)
# print('Best features :', rfecv_features)
# n_features = X_train.shape[1]
# plt.figure(figsize=(8,8))
# plt.barh(range(n_features), cv_estimator.feature_importances_, align='center') 
# plt.yticks(np.arange(n_features), X_train.columns.values) 
# plt.xlabel('Feature importance')
# plt.ylabel('Feature')
# plt.show()


# model_svm = make_pipeline(StandardScaler(), 
#                           SVC())
# model_svm.fit(features_train, y_train)
# y_preds_svm = model_svm.predict(features_test)

# score_svm_train = np.round(model_svm.score(features_train, y_train) * 100, 2)
# score_svm_test = np.round(model_svm.score(features_test, y_test) * 100, 2)
# f1_svm = np.round(f1_score(y_test, y_preds_svm) * 100, 2)

# cm = confusion_matrix(y_test, y_preds_svm, labels=[0, 1])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

# disp.plot()
# plt.title('SVM Confusion Matrix')
# plt.show()

# print(f'\nSVM train accuracy: {score_svm_train}%')
# print(f'SVM test accuracy: {score_svm_test}%')
# print(f'SVM F1 score: {f1_svm}%')

"""
K-Nearest Neighbor
"""
# model_knn = make_pipeline(StandardScaler(), 
#                           KNeighborsClassifier(n_neighbors=1))
# model_knn.fit(features_train, y_train)
# y_preds_knn = model_knn.predict(features_test)

# score_knn_train = np.round(model_knn.score(features_train, y_train) * 100, 2)
# score_knn_test = np.round(model_knn.score(features_test, y_test) * 100, 2)
# f1_knn = np.round(f1_score(y_test, y_preds_knn) * 100, 2)

# cm = confusion_matrix(y_test, y_preds_knn, labels=[0, 1])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

# disp.plot()
# plt.title('KNN Confusion Matrix')
# plt.show()

# print(f'\nKNN train accuracy: {score_knn_train}%')
# print(f'KNN test accuracy: {score_knn_test}%')
# print(f'KNN F1 score: {f1_knn}%')

"""
Random Forest
"""
# model_rf = make_pipeline(StandardScaler(), 
#                          RandomForestClassifier())
# model_rf.fit(features_train, y_train)
# y_preds_rf = model_rf.predict(features_test)

# score_rf_train = np.round(model_rf.score(features_train, y_train) * 100, 2)
# score_rf_test = np.round(model_rf.score(features_test, y_test) * 100, 2)
# f1_rf = np.round(f1_score(y_test, y_preds_rf) * 100, 2)

# print(f'\nRandom Forest train accuracy: {score_rf_train}%')
# print(f'Random Forest test accuracy: {score_rf_test}%')
# print(f'Random Forest F1 score: {f1_rf}%')

# cm = confusion_matrix(y_test, y_preds_rf, labels=[0, 1])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

# disp.plot()
# plt.title('Random Forest Confusion Matrix')
# plt.show()


"""
Decision Tree
"""
# model_dt = make_pipeline(StandardScaler(), 
#                          DecisionTreeClassifier())
# model_dt.fit(features_train, y_train)
# y_preds_dt = model_dt.predict(features_test)

# score_dt_train = np.round(model_dt.score(features_train, y_train) * 100, 2)
# score_dt_test = np.round(model_dt.score(features_test, y_test) * 100, 2)
# f1_dt = np.round(f1_score(y_test, y_preds_dt) * 100, 2)

# print(f'\nDecision Tree train accuracy: {score_dt_train}%')
# print(f'Decision Tree test accuracy: {score_dt_test}%')
# print(f'Decision Tree F1 score: {f1_dt}%')

# cm = confusion_matrix(y_test, y_preds_dt, labels=[0, 1])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classnames)

# disp.plot()
# plt.title('Decision Tree Confusion Matrix')
# plt.show()

# misclassifications = find_misclassifications(y_test, predictions)

# show_misclassifications(X_test, misclassifications, y_test, predictions)