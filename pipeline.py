# %%

import onnxruntime as ort
from onnxmltools.utils import save_model
from skl2onnx import get_latest_tested_opset_version
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from data_augmentation import augment_images
from evaluation import create_metrics_df, evaluate_model, plot_confusion_matrices, tune_model, plot_metric_graphs
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

x_train_raw, y_train_raw = x_train, y_train

x_train, y_train = augment_images(x_train, y_train, 5)

# %% Exploratory Data Analysis - Class Distribution

df = pd.DataFrame(all_images, columns=['Image', 'Class'])
df['Class'] = df['Class'].map({0: 'healthy', 1: 'wssv'})
df['Class'].value_counts().plot(
    kind='bar', color=plt.get_cmap("Set2").colors)  # type: ignore

plt.title("Distribution of Classes")
plt.xlabel('Class Name')
plt.ylabel('# of Images')
plt.show()

# %% Image Preprocessing and Feature Extraction

x_train_raw_images = [i for i, j in x_train_raw]
x_train_images = [i for i, j in x_train]
x_test_images = [i for i, j in x_test]

x_train_raw_lbp = extract_lbp(x_train_raw_images)
x_train_lbp = extract_lbp(x_train_images)
x_test_lbp = extract_lbp(x_test_images)

# %% Create Histogram from LBP Images

x_train_raw_lbp_hist = create_histograms(x_train_raw_lbp,
                                         sub_images_num=3,
                                         bins_per_sub_images=64)

x_train_lbp_hist = create_histograms(x_train_lbp,
                                     sub_images_num=3,
                                     bins_per_sub_images=64)

x_test_lbp_hist = create_histograms(x_test_lbp,
                                    sub_images_num=3,
                                    bins_per_sub_images=64)

# %% Define Models and Pipelines

scaler = StandardScaler()
strat_kfold = StratifiedKFold(n_splits=5,
                              shuffle=True,
                              random_state=random_state)
sampler = SMOTETomek(sampling_strategy='auto')
svc = SVC(random_state=random_state)
logreg = LogisticRegression(random_state=random_state)
rf = RandomForestClassifier(random_state=random_state)
models = {
    'svm': {
        'pipeline_raw': Pipeline(steps=[('scaler', scaler),
                                        ('classifier', svc)
                                        ]),
        'pipeline_sampled': Pipeline(steps=[('scaler', scaler),
                                            ('sampler', sampler),
                                            ('classifier', svc)
                                            ]),
        'param_grid': {
            'classifier__kernel': ['rbf'],
            'classifier__C': [1000, 100, 10, 1.0, 0.1, 0.01],
            'classifier__gamma': [1, 0.1, 0.01, 0.001],
        }
    },
    'logreg': {
        'pipeline_raw': Pipeline(steps=[('scaler', scaler),
                                        ('classifier', logreg)
                                        ]),
        'pipeline_sampled': Pipeline(steps=[('scaler', scaler),
                                            ('sampler', sampler),
                                            ('classifier', logreg)
                                            ]),
        'param_grid': {
            'classifier__penalty': ['l2'],
            'classifier__C': [100, 10, 1.0, 0.1, 0.01],
            'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear']
        }
    },
    'rf': {
        'pipeline_raw': Pipeline(steps=[('scaler', scaler),
                                        ('classifier', rf)
                                        ]),
        'pipeline_sampled': Pipeline(steps=[('scaler', scaler), 
                                            ('sampler', sampler),
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

# %% SVM
print("No Hyperparameter Tuning")
print("\nTraining Scheme A - No Sampling")
a_train, a_test = evaluate_model(pipeline=models['svm']['pipeline_raw'],
                                 cv=strat_kfold,
                                 x_train=x_train_raw_lbp_hist,
                                 y_train=y_train_raw,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(a_test['train_cm'], 'Training',
                        a_test['test_cm'], 'Testing',
                        'Training Scheme A - No Sampling',
                        classnames=classnames)

print("\nTraining Scheme B - Data Augmentation Only")
b_train, b_test = evaluate_model(pipeline=models['svm']['pipeline_raw'],
                                 cv=strat_kfold,
                                 x_train=x_train_lbp_hist,
                                 y_train=y_train,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(b_test['train_cm'], 'Training',
                        b_test['test_cm'], 'Testing',
                        'Training Scheme B - Data Augmentation Only',
                        classnames=classnames)

print("\nTraining Scheme C - SMOTE and Tomek Links")
c_train, c_test = evaluate_model(pipeline=models['svm']['pipeline_sampled'],
                                 cv=strat_kfold,
                                 x_train=x_train_raw_lbp_hist,
                                 y_train=y_train_raw,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(c_test['train_cm'], 'Training',
                        c_test['test_cm'], 'Testing',
                        'Training Scheme C - SMOTE and Tomek Links',
                        classnames=classnames)

print("\nTraining Scheme D - Data Augmentation + SMOTE and Tomek Links")
d_train, d_test = evaluate_model(pipeline=models['svm']['pipeline_sampled'],
                                 cv=strat_kfold,
                                 x_train=x_train_lbp_hist,
                                 y_train=y_train,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(d_test['train_cm'], 'Training',
                        d_test['test_cm'], 'Testing',
                        'Training Scheme D - Data Augmentation + SMOTE and Tomek Links',
                        classnames=classnames)

print("Hyperparameter Optimization")
print("\nTraining Scheme A - No Sampling")
a_best_params = tune_model(pipeline=models['svm']['pipeline_raw'],
                           strat_kfold=strat_kfold,
                           param_grid=models['svm']['param_grid'],
                           x_train=x_train_raw_lbp_hist,
                           y_train=y_train_raw)

hp_a_train, hp_a_test = evaluate_model(pipeline=models['svm']['pipeline_raw'],
                                       cv=strat_kfold,
                                       x_train=x_train_raw_lbp_hist,
                                       y_train=y_train_raw,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=a_best_params)

plot_confusion_matrices(a_test['train_cm'], 'Training',
                        a_test['test_cm'], 'Testing',
                        'Training Scheme A - No Sampling',
                        classnames=classnames)


print("\nTraining Scheme B - Data Augmentation Only")
b_best_params = tune_model(pipeline=models['svm']['pipeline_raw'],
                           strat_kfold=strat_kfold,
                           param_grid=models['svm']['param_grid'],
                           x_train=x_train_lbp_hist,
                           y_train=y_train)

hp_b_train, hp_b_test = evaluate_model(pipeline=models['svm']['pipeline_raw'],
                                       cv=strat_kfold,
                                       x_train=x_train_lbp_hist,
                                       y_train=y_train,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=b_best_params)

plot_confusion_matrices(b_test['train_cm'], 'Training',
                        b_test['test_cm'], 'Testing',
                        'Training Scheme B - Data Augmentation Only',
                        classnames=classnames)


print("\nTraining Scheme C - SMOTE and Tomek Links")
c_best_params = tune_model(pipeline=models['svm']['pipeline_sampled'],
                           strat_kfold=strat_kfold,
                           param_grid=models['svm']['param_grid'],
                           x_train=x_train_raw_lbp_hist,
                           y_train=y_train_raw)

hp_c_train, hp_c_test = evaluate_model(pipeline=models['svm']['pipeline_sampled'],
                                       cv=strat_kfold,
                                       x_train=x_train_raw_lbp_hist,
                                       y_train=y_train_raw,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=c_best_params)

plot_confusion_matrices(c_test['train_cm'], 'Training',
                        c_test['test_cm'], 'Testing',
                        'Training Scheme C - SMOTE and Tomek Links',
                        classnames=classnames)


print("\nTraining Scheme D - Data Augmentation + SMOTE and Tomek Links")
d_best_params = tune_model(pipeline=models['svm']['pipeline_sampled'],
                           strat_kfold=strat_kfold,
                           param_grid=models['svm']['param_grid'],
                           x_train=x_train_lbp_hist,
                           y_train=y_train)

hp_d_train, hp_d_test = evaluate_model(pipeline=models['svm']['pipeline_sampled'],
                                       cv=strat_kfold,
                                       x_train=x_train_lbp_hist,
                                       y_train=y_train,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=d_best_params)

plot_confusion_matrices(d_test['train_cm'], 'Training',
                        d_test['test_cm'], 'Testing',
                        'Training Scheme D - Data Augmentation + SMOTE and Tomek Links',
                        classnames=classnames)

#%%

# %% Metric Graphs
train_set_df = create_metrics_df((a_train, b_train, c_train, d_train), 'train')
valid_set_df = create_metrics_df((a_train, b_train, c_train, d_train), 'test')
test_set_df = create_metrics_df((a_test, b_test, c_test, d_test), 'test')

plot_metric_graphs(train_set_df, 'Train')
plot_metric_graphs(valid_set_df, 'Valid')
plot_metric_graphs(test_set_df, 'Test')

# %%

hp_train_set_df = create_metrics_df(
    (hp_a_train, hp_b_train, hp_c_train, hp_d_train), 'train')
hp_valid_set_df = create_metrics_df(
    (hp_a_train, hp_b_train, hp_c_train, hp_d_train), 'test')
hp_test_set_df = create_metrics_df(
    (hp_a_test, hp_b_test, hp_c_test, hp_d_test), 'test')

plot_metric_graphs(hp_train_set_df, 'HP Tuned - Train')
plot_metric_graphs(hp_valid_set_df, 'HP Tuned - Valid')
plot_metric_graphs(hp_test_set_df, 'HP Tuned - Test')

# %% Logistic Regression
print("No Hyperparameter Tuning")
print("\nTraining Scheme A - No Sampling")
a_train, a_test = evaluate_model(pipeline=models['logreg']['pipeline_raw'],
                                 cv=strat_kfold,
                                 x_train=x_train_raw_lbp_hist,
                                 y_train=y_train_raw,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(a_test['train_cm'], 'Training',
                        a_test['test_cm'], 'Testing',
                        'Training Scheme A - No Sampling',
                        classnames=classnames)

print("\nTraining Scheme B - Data Augmentation Only")
b_train, b_test = evaluate_model(pipeline=models['logreg']['pipeline_raw'],
                                 cv=strat_kfold,
                                 x_train=x_train_lbp_hist,
                                 y_train=y_train,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(b_test['train_cm'], 'Training',
                        b_test['test_cm'], 'Testing',
                        'Training Scheme B - Data Augmentation Only',
                        classnames=classnames)

print("\nTraining Scheme C - SMOTE and Tomek Links")
c_train, c_test = evaluate_model(pipeline=models['logreg']['pipeline_sampled'],
                                 cv=strat_kfold,
                                 x_train=x_train_raw_lbp_hist,
                                 y_train=y_train_raw,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(c_test['train_cm'], 'Training',
                        c_test['test_cm'], 'Testing',
                        'Training Scheme C - SMOTE and Tomek Links',
                        classnames=classnames)

print("\nTraining Scheme D - Data Augmentation + SMOTE and Tomek Links")
d_train, d_test = evaluate_model(pipeline=models['logreg']['pipeline_sampled'],
                                 cv=strat_kfold,
                                 x_train=x_train_lbp_hist,
                                 y_train=y_train,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(d_test['train_cm'], 'Training',
                        d_test['test_cm'], 'Testing',
                        'Training Scheme D - Data Augmentation + SMOTE and Tomek Links',
                        classnames=classnames)

print("Hyperparameter Optimization")
print("\nTraining Scheme A - No Sampling")
a_best_params = tune_model(pipeline=models['logreg']['pipeline_raw'],
                           strat_kfold=strat_kfold,
                           param_grid=models['logreg']['param_grid'],
                           x_train=x_train_raw_lbp_hist,
                           y_train=y_train_raw)

hp_a_train, hp_a_test = evaluate_model(pipeline=models['logreg']['pipeline_raw'],
                                       cv=strat_kfold,
                                       x_train=x_train_raw_lbp_hist,
                                       y_train=y_train_raw,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=a_best_params)

plot_confusion_matrices(a_test['train_cm'], 'Training',
                        a_test['test_cm'], 'Testing',
                        'Training Scheme A - No Sampling',
                        classnames=classnames)


print("\nTraining Scheme B - Data Augmentation Only")
b_best_params = tune_model(pipeline=models['logreg']['pipeline_raw'],
                           strat_kfold=strat_kfold,
                           param_grid=models['logreg']['param_grid'],
                           x_train=x_train_lbp_hist,
                           y_train=y_train)

hp_b_train, hp_b_test = evaluate_model(pipeline=models['logreg']['pipeline_raw'],
                                       cv=strat_kfold,
                                       x_train=x_train_lbp_hist,
                                       y_train=y_train,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=b_best_params)

plot_confusion_matrices(b_test['train_cm'], 'Training',
                        b_test['test_cm'], 'Testing',
                        'Training Scheme B - Data Augmentation Only',
                        classnames=classnames)


print("\nTraining Scheme C - SMOTE and Tomek Links")
c_best_params = tune_model(pipeline=models['logreg']['pipeline_sampled'],
                           strat_kfold=strat_kfold,
                           param_grid=models['logreg']['param_grid'],
                           x_train=x_train_raw_lbp_hist,
                           y_train=y_train_raw)

hp_c_train, hp_c_test = evaluate_model(pipeline=models['logreg']['pipeline_sampled'],
                                       cv=strat_kfold,
                                       x_train=x_train_raw_lbp_hist,
                                       y_train=y_train_raw,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=c_best_params)

plot_confusion_matrices(c_test['train_cm'], 'Training',
                        c_test['test_cm'], 'Testing',
                        'Training Scheme C - SMOTE and Tomek Links',
                        classnames=classnames)


print("\nTraining Scheme D - Data Augmentation + SMOTE and Tomek Links")
d_best_params = tune_model(pipeline=models['logreg']['pipeline_sampled'],
                           strat_kfold=strat_kfold,
                           param_grid=models['logreg']['param_grid'],
                           x_train=x_train_lbp_hist,
                           y_train=y_train)

hp_d_train, hp_d_test = evaluate_model(pipeline=models['logreg']['pipeline_sampled'],
                                       cv=strat_kfold,
                                       x_train=x_train_lbp_hist,
                                       y_train=y_train,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=d_best_params)

plot_confusion_matrices(d_test['train_cm'], 'Training',
                        d_test['test_cm'], 'Testing',
                        'Training Scheme D - Data Augmentation + SMOTE and Tomek Links',
                        classnames=classnames)

#%%

# %% Metric Graphs
train_set_df = create_metrics_df((a_train, b_train, c_train, d_train), 'train')
valid_set_df = create_metrics_df((a_train, b_train, c_train, d_train), 'test')
test_set_df = create_metrics_df((a_test, b_test, c_test, d_test), 'test')

plot_metric_graphs(train_set_df, 'Train')
plot_metric_graphs(valid_set_df, 'Valid')
plot_metric_graphs(test_set_df, 'Test')

# %%

hp_train_set_df = create_metrics_df(
    (hp_a_train, hp_b_train, hp_c_train, hp_d_train), 'train')
hp_valid_set_df = create_metrics_df(
    (hp_a_train, hp_b_train, hp_c_train, hp_d_train), 'test')
hp_test_set_df = create_metrics_df(
    (hp_a_test, hp_b_test, hp_c_test, hp_d_test), 'test')

plot_metric_graphs(hp_train_set_df, 'HP Tuned - Train')
plot_metric_graphs(hp_valid_set_df, 'HP Tuned - Valid')
plot_metric_graphs(hp_test_set_df, 'HP Tuned - Test')

# %% Random Forest
print("No Hyperparameter Tuning")
print("\nTraining Scheme A - No Sampling")
a_train, a_test = evaluate_model(pipeline=models['rf']['pipeline_raw'],
                                 cv=strat_kfold,
                                 x_train=x_train_raw_lbp_hist,
                                 y_train=y_train_raw,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(a_test['train_cm'], 'Training',
                        a_test['test_cm'], 'Testing',
                        'Training Scheme A - No Sampling',
                        classnames=classnames)

print("\nTraining Scheme B - Data Augmentation Only")
b_train, b_test = evaluate_model(pipeline=models['rf']['pipeline_raw'],
                                 cv=strat_kfold,
                                 x_train=x_train_lbp_hist,
                                 y_train=y_train,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(b_test['train_cm'], 'Training',
                        b_test['test_cm'], 'Testing',
                        'Training Scheme B - Data Augmentation Only',
                        classnames=classnames)

print("\nTraining Scheme C - SMOTE and Tomek Links")
c_train, c_test = evaluate_model(pipeline=models['rf']['pipeline_sampled'],
                                 cv=strat_kfold,
                                 x_train=x_train_raw_lbp_hist,
                                 y_train=y_train_raw,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(c_test['train_cm'], 'Training',
                        c_test['test_cm'], 'Testing',
                        'Training Scheme C - SMOTE and Tomek Links',
                        classnames=classnames)

print("\nTraining Scheme D - Data Augmentation + SMOTE and Tomek Links")
d_train, d_test = evaluate_model(pipeline=models['rf']['pipeline_sampled'],
                                 cv=strat_kfold,
                                 x_train=x_train_lbp_hist,
                                 y_train=y_train,
                                 x_test=x_test_lbp_hist,
                                 y_test=y_test,
                                 params=None)

plot_confusion_matrices(d_test['train_cm'], 'Training',
                        d_test['test_cm'], 'Testing',
                        'Training Scheme D - Data Augmentation + SMOTE and Tomek Links',
                        classnames=classnames)

print("Hyperparameter Optimization")
print("\nTraining Scheme A - No Sampling")
a_best_params = tune_model(pipeline=models['rf']['pipeline_raw'],
                           strat_kfold=strat_kfold,
                           param_grid=models['rf']['param_grid'],
                           x_train=x_train_raw_lbp_hist,
                           y_train=y_train_raw)

hp_a_train, hp_a_test = evaluate_model(pipeline=models['rf']['pipeline_raw'],
                                       cv=strat_kfold,
                                       x_train=x_train_raw_lbp_hist,
                                       y_train=y_train_raw,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=a_best_params)

plot_confusion_matrices(a_test['train_cm'], 'Training',
                        a_test['test_cm'], 'Testing',
                        'Training Scheme A - No Sampling',
                        classnames=classnames)


print("\nTraining Scheme B - Data Augmentation Only")
b_best_params = tune_model(pipeline=models['rf']['pipeline_raw'],
                           strat_kfold=strat_kfold,
                           param_grid=models['rf']['param_grid'],
                           x_train=x_train_lbp_hist,
                           y_train=y_train)

hp_b_train, hp_b_test = evaluate_model(pipeline=models['rf']['pipeline_raw'],
                                       cv=strat_kfold,
                                       x_train=x_train_lbp_hist,
                                       y_train=y_train,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=b_best_params)

plot_confusion_matrices(b_test['train_cm'], 'Training',
                        b_test['test_cm'], 'Testing',
                        'Training Scheme B - Data Augmentation Only',
                        classnames=classnames)


print("\nTraining Scheme C - SMOTE and Tomek Links")
c_best_params = tune_model(pipeline=models['rf']['pipeline_sampled'],
                           strat_kfold=strat_kfold,
                           param_grid=models['rf']['param_grid'],
                           x_train=x_train_raw_lbp_hist,
                           y_train=y_train_raw)

hp_c_train, hp_c_test = evaluate_model(pipeline=models['rf']['pipeline_sampled'],
                                       cv=strat_kfold,
                                       x_train=x_train_raw_lbp_hist,
                                       y_train=y_train_raw,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=c_best_params)

plot_confusion_matrices(c_test['train_cm'], 'Training',
                        c_test['test_cm'], 'Testing',
                        'Training Scheme C - SMOTE and Tomek Links',
                        classnames=classnames)


print("\nTraining Scheme D - Data Augmentation + SMOTE and Tomek Links")
d_best_params = tune_model(pipeline=models['rf']['pipeline_sampled'],
                           strat_kfold=strat_kfold,
                           param_grid=models['rf']['param_grid'],
                           x_train=x_train_lbp_hist,
                           y_train=y_train)

hp_d_train, hp_d_test = evaluate_model(pipeline=models['rf']['pipeline_sampled'],
                                       cv=strat_kfold,
                                       x_train=x_train_lbp_hist,
                                       y_train=y_train,
                                       x_test=x_test_lbp_hist,
                                       y_test=y_test,
                                       params=d_best_params)

plot_confusion_matrices(d_test['train_cm'], 'Training',
                        d_test['test_cm'], 'Testing',
                        'Training Scheme D - Data Augmentation + SMOTE and Tomek Links',
                        classnames=classnames)

#%%

# %% Metric Graphs
train_set_df = create_metrics_df((a_train, b_train, c_train, d_train), 'train')
valid_set_df = create_metrics_df((a_train, b_train, c_train, d_train), 'test')
test_set_df = create_metrics_df((a_test, b_test, c_test, d_test), 'test')

plot_metric_graphs(train_set_df, 'RF - Train')
plot_metric_graphs(valid_set_df, 'RF - Valid')
plot_metric_graphs(test_set_df, 'RF - Test')

# %%

hp_train_set_df = create_metrics_df(
    (hp_a_train, hp_b_train, hp_c_train, hp_d_train), 'train')
hp_valid_set_df = create_metrics_df(
    (hp_a_train, hp_b_train, hp_c_train, hp_d_train), 'test')
hp_test_set_df = create_metrics_df(
    (hp_a_test, hp_b_test, hp_c_test, hp_d_test), 'test')

plot_metric_graphs(hp_train_set_df, 'RF - HP Tuned - Train')
plot_metric_graphs(hp_valid_set_df, 'RF - HP Tuned - Valid')
plot_metric_graphs(hp_test_set_df, 'RF - HP Tuned - Test')

# %% ONNX Export
clf = Pipeline(steps=[('scaler', scaler),
                      ('classifier', logreg)
                      ])
clf.set_params(**{'classifier__solver': 'liblinear', 'classifier__penalty': 'l2', 'classifier__C': 0.01})
clf.fit(x_train_lbp_hist, y_train)

target_opset = get_latest_tested_opset_version()
n_features = x_train_lbp_hist.shape[1]
onnx_clf = convert_sklearn(
    clf,
    "logreg_model",
    initial_types=[("input", FloatTensorType([None, n_features]))],
    target_opset={"": target_opset, "ai.onnx.ml": 1}
)
save_model(onnx_clf, "logreg_model.onnx")

# %%
model_name = "logreg_model.onnx"
sess = ort.InferenceSession(model_name)
preds, _ = sess.run(
    None, {"input": x_test_lbp_hist.astype(np.float32)}
)

print(classification_report(y_test, preds))
# %%
