# %%

from data_augmentation import augment_images
from evaluation import create_metrics_df, create_metrics_df_v2, evaluate_model, evaluate_model_v2, plot_confusion_matrices, tune_model, plot_metric_graphs
from feature_extraction import extract_lbp, create_histograms, extract_glcm_noloop, split_image
from utils import load_images, show_raw_images, crop_images, preprocess_images

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import loguniform

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
from imblearn.pipeline import Pipeline

sns.set_theme(style="ticks")
plt.rcParams['figure.dpi'] = 600
RANDOM_STATE = 42

# %% Data Loading

main_dir = 'cropped'
classnames = ['healthy', 'wssv']

class_0 = load_images(f'{main_dir}/healthy')
class_1 = load_images(f'{main_dir}/wssv')

class_0 = preprocess_images(class_0)
class_1 = preprocess_images(class_1)

class_0_num_samples = len(class_0)
class_1_num_samples = len(class_1)

non_augmented_labels = np.array(
    [0] * class_0_num_samples + [1] * class_1_num_samples)

non_augmented_images = np.vstack((class_0, class_1))
non_augmented_images = list(zip(non_augmented_images, non_augmented_labels))

augmented_images = augment_images(
    non_augmented_images, non_augmented_labels, 10)
augmented_labels = np.array([label for _, label in augmented_images])

# %% Exploratory Data Analysis - Class Distribution

df = pd.DataFrame(non_augmented_images, columns=['Image', 'Class'])
df['Class'] = df['Class'].map({0: 'healthy', 1: 'wssv'})
df['Class'].value_counts().plot(
    kind='bar', color=plt.get_cmap("Set2").colors,  # type: ignore
    title='Distribution of Classes',
    xlabel='Class Name',
    ylabel='# of Images')

# %% Feature Extraction

non_augmented_lbps = extract_lbp([image for image, _ in non_augmented_images])
augmented_lbps = extract_lbp([image for image, _ in augmented_images])

non_augmented_lbp_histograms = create_histograms(non_augmented_lbps,
                                                 sub_images_num=3,
                                                 bins_per_sub_images=128)

augmented_lbp_histograms = create_histograms(augmented_lbps,
                                             sub_images_num=3,
                                             bins_per_sub_images=128)

# %% Define Models and Pipelines

standard_scaler = StandardScaler()
stratified_kfold = StratifiedKFold(n_splits=5,
                                   shuffle=True,
                                   random_state=RANDOM_STATE)

smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
svm = SVC(random_state=RANDOM_STATE)
logreg = LogisticRegression(max_iter=1000,
                            random_state=RANDOM_STATE)
rf = RandomForestClassifier(random_state=RANDOM_STATE)

models = {
    'svm': {
        'pipeline_no_sample': Pipeline(steps=[('scaler', standard_scaler),
                                              ('classifier', svm)
                                              ]),
        'pipeline_with_sample': Pipeline(steps=[('scaler', standard_scaler),
                                                ('sampler', smote_tomek),
                                                ('classifier', svm)
                                                ]),
        'param_grid': {
            'classifier__C': [1, 1e2, 1e3, 1e4, 1e5],
            'classifier__gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        }
    },
    'logreg': {
        'pipeline_no_sample': Pipeline(steps=[('scaler', standard_scaler),
                                              ('classifier', logreg)
                                              ]),
        'pipeline_with_sample': Pipeline(steps=[('scaler', standard_scaler),
                                                ('sampler', smote_tomek),
                                                ('classifier', logreg)
                                                ]),
        'param_grid': {
            'classifier__C': [1, 1e2, 1e3, 1e4, 1e5],
            'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear']
        }
    },
    'rf': {
        'pipeline_no_sample': Pipeline(steps=[('scaler', standard_scaler),
                                              ('classifier', rf)
                                              ]),
        'pipeline_with_sample': Pipeline(steps=[('scaler', standard_scaler),
                                                ('sampler', smote_tomek),
                                                ('classifier', rf)
                                                ]),
        'param_grid': {
            'classifier__n_estimators': [100, 200, 300, 400, 500],
            'classifier__max_depth': [5, 10, 15, 20, 25, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
    }
}

# %% Model Training

print("\nNo Hyperparameter Tuning")
print("\nTraining Scheme A - No Sampling")
a_scores = evaluate_model_v2(pipeline=models['svm']['pipeline_no_sample'],
                             cv=stratified_kfold,
                             X=non_augmented_lbp_histograms,
                             y=non_augmented_labels,
                             params=None)


print("\nTraining Scheme B - Data Augmentation Only")
b_scores = evaluate_model_v2(pipeline=models['svm']['pipeline_no_sample'],
                             cv=stratified_kfold,
                             X=augmented_lbp_histograms,
                             y=augmented_labels,
                             params=None)


print("\nTraining Scheme C - SMOTE and Tomek Links")
c_scores = evaluate_model_v2(pipeline=models['svm']['pipeline_with_sample'],
                             cv=stratified_kfold,
                             X=non_augmented_lbp_histograms,
                             y=non_augmented_labels,
                             params=None)


print("\nTraining Scheme D - Data Augmentation + SMOTE and Tomek Links")
d_scores = evaluate_model_v2(pipeline=models['svm']['pipeline_with_sample'],
                             cv=stratified_kfold,
                             X=augmented_lbp_histograms,
                             y=augmented_labels,
                             params=None)


print("\nHyperparameter Optimization")
print("\nTraining Scheme A - No Sampling")

a_best_params = tune_model(pipeline=models['svm']['pipeline_no_sample'],
                           cv=stratified_kfold,
                           param_grid=models['svm']['param_grid'],
                           X=non_augmented_lbp_histograms,
                           y=non_augmented_labels)

a_hp_scores = evaluate_model_v2(pipeline=models['svm']['pipeline_no_sample'],
                                cv=stratified_kfold,
                                X=non_augmented_lbp_histograms,
                                y=non_augmented_labels,
                                params=a_best_params)

print("\nTraining Scheme B - Data Augmentation Only")
b_best_params = tune_model(pipeline=models['svm']['pipeline_no_sample'],
                           cv=stratified_kfold,
                           param_grid=models['svm']['param_grid'],
                           X=augmented_lbp_histograms,
                           y=augmented_labels)

b_hp_scores = evaluate_model_v2(pipeline=models['svm']['pipeline_no_sample'],
                                cv=stratified_kfold,
                                X=augmented_lbp_histograms,
                                y=augmented_labels,
                                params=b_best_params)

print("\nTraining Scheme C - SMOTE and Tomek Links")
c_best_params = tune_model(pipeline=models['svm']['pipeline_with_sample'],
                           cv=stratified_kfold,
                           param_grid=models['svm']['param_grid'],
                           X=non_augmented_lbp_histograms,
                           y=non_augmented_labels)

c_hp_scores = evaluate_model_v2(pipeline=models['svm']['pipeline_with_sample'],
                                cv=stratified_kfold,
                                X=non_augmented_lbp_histograms,
                                y=non_augmented_labels,
                                params=c_best_params)

print("\nTraining Scheme D - Data Augmentation + SMOTE and Tomek Links")
d_best_params = tune_model(pipeline=models['svm']['pipeline_with_sample'],
                           cv=stratified_kfold,
                           param_grid=models['svm']['param_grid'],
                           X=augmented_lbp_histograms,
                           y=augmented_labels)

d_hp_scores = evaluate_model_v2(pipeline=models['svm']['pipeline_with_sample'],
                                cv=stratified_kfold,
                                X=augmented_lbp_histograms,
                                y=augmented_labels,
                                params=d_best_params)

# %% Metric Graphs

no_hp_df = create_metrics_df_v2((a_scores, b_scores, c_scores, d_scores))
plot_metric_graphs(no_hp_df, 'SVM - No Hyperparameter Tuning')

hp_df = create_metrics_df_v2(
    (a_hp_scores, b_hp_scores, c_hp_scores, d_hp_scores))
plot_metric_graphs(hp_df, 'SVM - With Hyperparameter Tuning')


