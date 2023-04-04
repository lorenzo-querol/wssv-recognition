import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, cross_validate

RANDOM_STATE = 42


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


def get_test_metrics(pipeline, x_train, y_train, x_test, y_test):
    pipeline.fit(x_train, y_train)

    # Training Confusion Matrix
    y_pred = pipeline.predict(x_train)
    train_cm = confusion_matrix(y_train, y_pred, labels=[0, 1])

    # Test Confusion Matrix
    y_pred = pipeline.predict(x_test)
    test_cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Get Test FNR
    FN = test_cm.sum(axis=1) - np.diag(test_cm)
    TP = np.diag(test_cm)
    test_fnr = FN / (TP + FN)

    return {
        'test_accuracy': balanced_accuracy_score(y_test, y_pred),
        'test_f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'test_fnr': test_fnr[1],
        'train_cm': train_cm,
        'test_cm': test_cm
    }


def plot_metric_graphs(df, split):
    bar = df.plot(x='Training Scheme',
                  kind='bar',
                  width=0.9,
                  rot=0,
                  stacked=False,
                  title=split)

    plt.xticks(rotation=0)
    bar.figure.set_size_inches(10, 5)

    for p in bar.containers:  # type: ignore
        bar.bar_label(p, fmt='%.2f', label_type='edge')

    bar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, ncol=3)

    plt.show()


def create_metrics_df(schemes, split):
    scheme_a, scheme_b, scheme_c, scheme_d = schemes

    df = pd.DataFrame(
        [
            ['A', scheme_a[f'{split}_accuracy'],
                scheme_a[f'{split}_f1_weighted'], scheme_a[f'{split}_fnr']],
            ['B', scheme_b[f'{split}_accuracy'],
                scheme_b[f'{split}_f1_weighted'], scheme_b[f'{split}_fnr']],
            ['C', scheme_c[f'{split}_accuracy'],
                scheme_c[f'{split}_f1_weighted'], scheme_c[f'{split}_fnr']],
            ['D', scheme_d[f'{split}_accuracy'],
                scheme_d[f'{split}_f1_weighted'], scheme_d[f'{split}_fnr']]
        ],
        columns=['Training Scheme', 'Accuracy', 'F1-Score', 'FNR'])

    return df


def plot_confusion_matrices(cm1, title1, cm2, title2, sup_title, classnames):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5), sharex=True, sharey=True)
    fig.suptitle(sup_title)

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,
                                   display_labels=classnames)

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                                   display_labels=classnames)

    disp1.plot(ax=ax1, colorbar=False, cmap='OrRd', values_format='d')
    disp2.plot(ax=ax2, colorbar=False, cmap='OrRd', values_format='d')

    ax1.set_title(title1)
    ax2.set_title(title2)

    plt.show()


def tune_model(pipeline, cv, param_grid, X, y):
    search = RandomizedSearchCV(pipeline,
                                param_grid,
                                cv=cv,
                                scoring='f1_weighted',
                                random_state=RANDOM_STATE,
                                n_jobs=-1)

    search.fit(X, y)
    print("\nBest Parameters: ", search.best_params_)

    return search.best_params_


def evaluate_model(pipeline, cv, x_train, y_train, x_test, y_test, params):

    if params:
        pipeline.set_params(**params)

    train_scores = cross_validate(pipeline,
                                  x_train,
                                  y_train,
                                  scoring=confusion_matrix_scorer,
                                  cv=cv,
                                  n_jobs=-1,
                                  return_train_score=True
                                  )

    print('\nTraining')
    print('Accuracy: %.4f' % max(train_scores['train_accuracy']))
    print('F1 Score: %.4f' % max(train_scores['train_f1_weighted']))
    print('FNR (WSSV): %.4f' % min(train_scores['train_fnr']))

    print('\nValidation')
    print('Accuracy: %.4f' % max(train_scores['test_accuracy']))
    print('F1 Score: %.4f' % max(train_scores['test_f1_weighted']))
    print('FNR: %.4f' % min(train_scores['test_fnr']))

    test_scores = get_test_metrics(pipeline,
                                   x_train,
                                   y_train,
                                   x_test,
                                   y_test)

    print('\nTest')
    print('Accuracy: %.4f' % test_scores['test_accuracy'])
    print('F1 score: %.4f' % test_scores['test_f1_weighted'])
    print('FNR (WSSV): %.4f' % test_scores['test_fnr'])

    train_scores = {
        'train_accuracy': max(train_scores['train_accuracy']),
        'train_f1_weighted': max(train_scores['train_f1_weighted']),
        'train_fnr': min(train_scores['train_fnr']),
        'test_accuracy': max(train_scores['test_accuracy']),
        'test_f1_weighted': max(train_scores['test_f1_weighted']),
        'test_fnr': min(train_scores['test_fnr'])
    }

    return train_scores, test_scores


def fnr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    fnr = FN / (TP + FN)
    return fnr


def evaluate_model_v2(pipeline, cv, X, y, params):

    if params:
        pipeline.set_params(**params)

    accuracy_scores = []
    f1_scores = []
    fnr_scores = []

    for i, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        x_train, y_train = X[train_idx], y[train_idx]
        x_valid, y_valid = X[valid_idx], y[valid_idx]

        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_valid)

        accuracy_scores.append(balanced_accuracy_score(y_valid, y_pred))
        f1_scores.append(f1_score(y_valid, y_pred, average='weighted'))
        fnr_scores.append(fnr(y_valid, y_pred))

    print('Accuracy: %.4f' % np.mean(accuracy_scores))
    print('F1 Score: %.4f' % np.mean(f1_scores))
    print('FNR (WSSV): %.4f' % np.mean(fnr_scores))

    return {"accuracy_avg": np.mean(accuracy_scores),
            "f1_score_avg": np.mean(f1_scores),
            "fnr_avg": np.mean(fnr_scores)}


def create_metrics_df_v2(schemes):
    scheme_a, scheme_b, scheme_c, scheme_d = schemes

    df = pd.DataFrame(
        [
            ['A', scheme_a[f'accuracy_avg'],
                scheme_a[f'f1_score_avg'], scheme_a[f'fnr_avg']],
            ['B', scheme_b[f'accuracy_avg'],
                scheme_b[f'f1_score_avg'], scheme_b[f'fnr_avg']],
            ['C', scheme_c[f'accuracy_avg'],
                scheme_c[f'f1_score_avg'], scheme_c[f'fnr_avg']],
            ['D', scheme_d[f'accuracy_avg'],
                scheme_d[f'f1_score_avg'], scheme_d[f'fnr_avg']]
        ],
        columns=['Training Scheme', 'Accuracy', 'F1-Score', 'FNR'])

    return df
