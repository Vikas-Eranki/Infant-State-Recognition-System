"""
src/model_ml.py
Reusable SVM helper functions for the Infant State Recognition System.
Primary metric: Macro F1-Score (15.9:1 class imbalance in this dataset).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, recall_score,
)


def train_svm(X_train, y_train, C=10, gamma="scale",
              class_weight="balanced", random_state=42):
    """
    Fit a balanced RBF-SVM on pre-scaled training features.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        StandardScaler-transformed feature matrix.
    y_train : np.ndarray, shape (n_train,)
        Integer class labels.
    C : float
        Regularisation parameter. Default 10 (Liu et al. 2019).
        C_i = C * N / (n_classes * n_i) for balanced weighting.
    gamma : str or float
        RBF bandwidth. 'scale' => 1/(n_features * Var(X_train)).
    class_weight : str
        'balanced' reweights penalty inversely by class frequency.
    random_state : int

    Returns
    -------
    sklearn.svm.SVC  Fitted model (OvO, 28 binary classifiers for 8 classes).
    """
    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        decision_function_shape="ovo",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, class_names, zero_division=0):
    """
    Compute evaluation metrics. Primary metric is Macro F1-Score.

    Accuracy is misleading with 15.9:1 imbalance: a dummy classifier
    predicting 'hungry' for every sample achieves ~35.3% accuracy.
    Macro F1 weights every class equally regardless of sample count.

    Returns
    -------
    dict: accuracy, macro_f1, weighted_f1, per_class_f1, y_pred, report_str.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy"     : accuracy_score(y_test, y_pred),
        "macro_f1"     : f1_score(y_test, y_pred, average="macro",
                                  zero_division=zero_division),
        "weighted_f1"  : f1_score(y_test, y_pred, average="weighted",
                                  zero_division=zero_division),
        "per_class_f1" : f1_score(y_test, y_pred, average=None,
                                  zero_division=zero_division),
        "y_pred"       : y_pred,
        "report_str"   : classification_report(
                             y_test, y_pred,
                             target_names=class_names,
                             zero_division=zero_division, digits=4),
    }


def plot_confusion_matrix(y_true, y_pred, class_names,
                           save_path=None, figsize=(20, 7)):
    """
    Plot raw-count and row-normalised confusion matrices side by side.

    Parameters
    ----------
    y_true, y_pred : array-like of int
    class_names    : list of str (in label-index order)
    save_path      : str or None
    figsize        : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, data, fmt, title in [
        (axes[0], cm,      "d",   "Confusion Matrix (Raw Counts)"),
        (axes[1], cm_norm, ".2f", "Confusion Matrix (Row-Normalised Recall)"),
    ]:
        kwargs = dict(vmin=0, vmax=1) if fmt == ".2f" else {}
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="white", ax=ax, **kwargs)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def get_failure_cases(model, X_test, y_test, class_names):
    """
    Return a DataFrame of every misclassified test sample.

    Parameters
    ----------
    model       : fitted sklearn estimator
    X_test      : np.ndarray, shape (n_test, n_features)
    y_test      : np.ndarray, shape (n_test,)
    class_names : list of str

    Returns
    -------
    pd.DataFrame with columns:
        test_pos, true_label_idx, pred_label_idx,
        true_class, pred_class, pair_key.
    """
    y_pred         = model.predict(X_test)
    fail_positions = np.where(y_pred != y_test)[0]
    decoder        = {i: c for i, c in enumerate(class_names)}
    records = []
    for pos in fail_positions:
        t = int(y_test[pos])
        p = int(y_pred[pos])
        records.append({
            "test_pos"       : pos,
            "true_label_idx" : t,
            "pred_label_idx" : p,
            "true_class"     : decoder[t],
            "pred_class"     : decoder[p],
            "pair_key"       : "|".join(sorted([decoder[t], decoder[p]])),
        })
    return pd.DataFrame(records)
