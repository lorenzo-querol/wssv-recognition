# %%

from data_augmentation import augment_images
from feature_extraction import extract_lbp, create_histograms, extract_glcm_noloop, split_image
from utils import load_images, show_raw_images, crop_images, preprocess_images

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
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


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)

    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    fnr = FN / (TP + FN)

    return {
        'accuracy': balanced_accuracy_score(y, y_pred),
        'f1_weighted': f1_score(y, y_pred, average='weighted'),
        'fnr': fnr[1]
    }


def get_test_metrics(pipeline, model_str):
    pipeline.fit(x_train_lbp_hist, y_train)

    # Training Confusion Matrix
    y_pred = pipeline.predict(x_train_lbp_hist)
    cm = confusion_matrix(y_train, y_pred, labels=[0, 1])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classnames)

    disp.plot(cmap=plt.cm.Oranges, values_format='d')
    plt.title(model_str + 'Training Confusion Matrix')
    plt.show()

    # Test Confusion Matrix
    y_pred = pipeline.predict(x_test_lbp_hist)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    test_fnr = FN / (TP + FN)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classnames)

    disp.plot(cmap=plt.cm.Oranges, values_format='d')
    plt.title(model_str + 'Test Confusion Matrix')
    plt.show()

    return {
        'test_accuracy': balanced_accuracy_score(y_test, y_pred),
        'test_f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'test_fnr': test_fnr[1],
    }


def evaluate(pipeline, cv, best_params=None, model_str=""):

    if best_params:
        pipeline.set_params(**best_params)

    scores = cross_validate(pipeline,
                            x_train_lbp_hist,
                            y_train,
                            scoring=confusion_matrix_scorer,
                            cv=cv,
                            n_jobs=-1,
                            return_train_score=True
                            )

    print('\nTraining')
    print('Accuracy: %.4f' % max(scores['train_accuracy']))
    print('F1 Score: %.4f' % max(scores['train_f1_weighted']))
    print('FNR (WSSV): %.4f' % min(scores['train_fnr']))

    print('\nValidation')
    print('Accuracy: %.4f' % max(scores['test_accuracy']))
    print('F1 Score: %.4f' % max(scores['test_f1_weighted']))
    print('FNR: %.4f' % min(scores['test_fnr']))

    test_scores = get_test_metrics(pipeline, model_str)

    print('\nTest')
    print('Accuracy: %.4f' % test_scores['test_accuracy'])
    print('F1 score: %.4f' % test_scores['test_f1_weighted'])
    print('FNR (WSSV): %.4f' % test_scores['test_fnr'])


def tune_model(pipeline, param_grid):
    classifier = GridSearchCV(pipeline,
                              param_grid,
                              cv=5,
                              refit=True,
                              n_jobs=-1)

    classifier.fit(x_train_lbp_hist, y_train)
    print("\nBest Parameters: ", classifier.best_params_)

    return classifier.best_params_


# %% SVM

scaler = StandardScaler()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

oversampler = BorderlineSMOTE(random_state=random_state)
undersampler = TomekLinks()

classifier = SVC(random_state=random_state)
pipeline1 = Pipeline(steps=[('scaler', scaler),
                            ('svm', classifier)
                            ])

pipeline2 = Pipeline(steps=[('scaler', scaler),
                            ('oversample', oversampler),
                            ('undersample', undersampler),
                            ('svm', classifier)
                            ])

print("\nWithout SMOTE and Tomek-Links")
evaluate(pipeline1, cv, model_str="SVM - ")

print("\nAfter Applying SMOTE and Tomek-Links")
evaluate(pipeline2, cv, model_str="SVM SMOTE & TL - ")

# %% With Hyperparameter Tuning

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

param_grid_svm = {'svm__C': C, 'svm__gamma': gamma}

print("\nHyperparameter Tuning without SMOTE and Tomek-Links")
best_params1 = tune_model(pipeline1, param_grid_svm)
evaluate(pipeline1, cv, best_params1, model_str="SVM HP Tuned - ")

print("\nHyperparameter Tuning after Applying SMOTE and Tomek-Links")
best_params2 = tune_model(pipeline2, param_grid_svm)
evaluate(pipeline2, cv, best_params2, model_str="SVM HP Tuned SMOTE & TL - ")

# # %% Logistic Regression

# classifier = LogisticRegression(random_state=random_state)
# pipeline1 = Pipeline(steps=[('scaler', scaler),
#                             ('lr', classifier)
#                             ])

# pipeline2 = Pipeline(steps=[('scaler', scaler),
#                             ('sampler', sampler),
#                             ('lr', classifier)
#                             ])

# print("\nWithout SMOTE and Tomek-Links")
# evaluate(pipeline1, cv)

# print("\nAfter Applying SMOTE and Tomek-Links")
# evaluate(pipeline2, cv)

# # %% With Hyperparameter Tuning

# C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# param_grid_lr = {'lr__C': C}

# print("\nHyperparameter Tuning without SMOTE and Tomek-Links")
# best_params1 = tune_model(pipeline1, param_grid_lr)
# evaluate(pipeline1, cv, best_params1)

# print("\nHyperparameter Tuning after Applying SMOTE and Tomek-Links")
# best_params2 = tune_model(pipeline2, param_grid_lr)
# evaluate(pipeline2, cv, best_params2)


# %% MLP

# classifier = MLPClassifier(random_state=random_state)
# pipeline1 = Pipeline(steps=[('scaler', scaler),
#                             ('mlp', classifier)
#                             ])

# pipeline2 = Pipeline(steps=[('scaler', scaler),
#                             ('sampler', sampler),
#                             ('mlp', classifier)
#                             ])

# print("\nWithout SMOTE and Tomek-Links")
# evaluate(pipeline1, cv)

# print("\nAfter Applying SMOTE and Tomek-Links")
# evaluate(pipeline2, cv)

# # %% With Hyperparameter Tuning

# hidden_layer_sizes = [(100, 100, 100), (100, 100), (100,)]
# activation = ['tanh', 'relu']
# solver = ['sgd', 'adam']
# alpha = [0.0001, 0.05]
# learning_rate = ['constant', 'adaptive']

# param_grid_mlp = {'mlp__hidden_layer_sizes': hidden_layer_sizes,
#                   'mlp__activation': activation,
#                   'mlp__solver': solver,
#                   'mlp__alpha': alpha,
#                   'mlp__learning_rate': learning_rate
#                   }

# print("\nHyperparameter Tuning without SMOTE and Tomek-Links")
# best_params1 = tune_model(pipeline1, param_grid_mlp)
# evaluate(pipeline1, cv, best_params1)

# print("\nHyperparameter Tuning after Applying SMOTE and Tomek-Links")
# best_params2 = tune_model(pipeline2, param_grid_mlp)
# evaluate(pipeline2, cv, best_params2)

# %% Decision Tree

# classifier = DecisionTreeClassifier(random_state=random_state)
# pipeline1 = Pipeline(steps=[('scaler', scaler),
#                             ('dt', classifier)
#                             ])

# pipeline2 = Pipeline(steps=[('scaler', scaler),
#                             ('sampler', sampler),
#                             ('dt', classifier)
#                             ])

# print("\nWithout SMOTE and Tomek-Links")
# evaluate(pipeline1, cv)

# print("\nAfter Applying SMOTE and Tomek-Links")
# evaluate(pipeline2, cv)

# # %% With Hyperparameter Tuning

# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# criterion = ['gini', 'entropy']

# param_grid_dt = {'dt__max_depth': max_depth,
#                  'dt__min_samples_split': min_samples_split,
#                  'dt__min_samples_leaf': min_samples_leaf,
#                  'dt__criterion': criterion
#                  }

# print("\nHyperparameter Tuning without SMOTE and Tomek-Links")
# best_params1 = tune_model(pipeline1, param_grid_dt)
# evaluate(pipeline1, cv, best_params1)

# print("\nHyperparameter Tuning after Applying SMOTE and Tomek-Links")
# best_params2 = tune_model(pipeline2, param_grid_dt)
# evaluate(pipeline2, cv, best_params2)
