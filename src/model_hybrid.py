"""
src/model_hybrid.py
-------------------
Phase 3 — hybrid / probability recalibration helpers.

The selected Model C uses **8-dim CNN softmax probabilities → LR meta-classifier**
(probability recalibration). The ablation study proved that adding SVM decision
scores to the fusion vector degrades performance on this corpus.

This module also provides helpers for the 264-dim (GAP + SVM) and 16-dim
(probs + SVM) fusion strategies explored during ablation.

SVM path must use the same inputs as Phase 2 Section 11: flattened Log-Mel in dB
(~[-80, 0]) + StandardScaler fit on train — NOT [0,1] normalised mel.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def extract_cnn_features(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Frozen CNN embeddings from the last GlobalAveragePooling2D layer.

    Parameters
    ----------
    model : tf.keras.Model
        Full trained DS-CNN (expects X shape (N, 128, 302, 1)).
    X : np.ndarray
        Normalised spectrograms, shape (N, 128, 302, 1), float32.

    Returns
    -------
    np.ndarray, shape (N, D) where D is GAP dimension (256 for project DS-CNN).
    """
    gap_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, GlobalAveragePooling2D):
            gap_layer = layer
            break
    if gap_layer is None:
        raise ValueError("No GlobalAveragePooling2D layer found in model.")

    feat_model = Model(inputs=model.input, outputs=gap_layer.output)
    out = feat_model.predict(X, verbose=0)
    return np.asarray(out, dtype=np.float32)


def extract_cnn_probabilities(
    model: tf.keras.Model, X: np.ndarray
) -> np.ndarray:
    """
    CNN softmax output (8-dim class probabilities).

    Used by the selected Model C (probability recalibration hybrid).
    Compared to the 256-dim GAP embedding, the 8-dim softmax provides a
    cleaner, lower-dimensional signal that the LR meta-learner can
    exploit more effectively.

    Parameters
    ----------
    model : tf.keras.Model
        Full trained DS-CNN.
    X : np.ndarray
        Normalised spectrograms, shape (N, 128, 302, 1), float32.

    Returns
    -------
    np.ndarray, shape (N, 8).
    """
    probs = model.predict(X, verbose=0)
    return np.asarray(probs, dtype=np.float32)


def extract_svm_scores(
    svm_model,
    svm_scaler,
    X_db: np.ndarray,
    *,
    ovr: bool = True,
) -> np.ndarray:
    """
    SVM decision-function scores (8-dim OvR view by default).

    Parameters
    ----------
    svm_model : sklearn.svm.SVC
    svm_scaler : sklearn.preprocessing.StandardScaler
        Fitted on flattened train mel in dB (same as Phase 2).
    X_db : np.ndarray
        Log-Mel in dB, shape (N, 128, 302) — same scale as Phase 2 SVM training.
    ovr : bool
        If True, set decision_function_shape='ovr' before calling decision_function.

    Returns
    -------
    np.ndarray, shape (N, 8) when OvR aggregation applies.
    """
    if ovr:
        svm_model.decision_function_shape = "ovr"
    flat = X_db.reshape(len(X_db), -1)
    Xs = svm_scaler.transform(flat)
    scores = svm_model.decision_function(Xs)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return scores


def build_hybrid_features(cnn_feat: np.ndarray, svm_scores: np.ndarray) -> np.ndarray:
    """Concatenate CNN GAP features and SVM scores; shape (N, D_cnn + D_svm)."""
    if cnn_feat.shape[0] != svm_scores.shape[0]:
        raise ValueError(
            f"Batch mismatch: CNN {cnn_feat.shape[0]} vs SVM {svm_scores.shape[0]}"
        )
    return np.concatenate([cnn_feat, svm_scores], axis=1).astype(np.float32)


def train_meta_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    cv_splits: int = 3,
    random_state: int = 42,
    verbose: int = 0,
) -> GridSearchCV:
    """
    Grid-search multinomial LogisticRegression inside Pipeline(StandardScaler + LR).

    Scaler is refit inside each CV fold (no leakage).

    Returns
    -------
    GridSearchCV
        Fitted grid; ``best_estimator_`` is the fitted Pipeline.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    param_grid = {
        "lr__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "lr__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state
    )
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        refit=True,
        n_jobs=-1,
        verbose=verbose,
    )
    grid.fit(X_train, y_train)
    return grid


def evaluate_hybrid(
    meta_pipeline: Pipeline,
    X: np.ndarray,
    y_true: np.ndarray,
    class_names: list,
    *,
    zero_division: int = 0,
) -> dict:
    """
    Evaluate fitted meta-classifier (Pipeline with scaler + LR).

    X : unscaled hybrid features (N, D); the Pipeline applies scaling.
    """
    y_pred = meta_pipeline.predict(X)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=zero_division
        ),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, zero_division=zero_division
        ),
        "y_pred": y_pred,
        "report_str": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=zero_division,
            digits=4,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def validate_svm_ovr_shape(svm_model, svm_scaler, X_db_sample: np.ndarray) -> int:
    """
    Return SVM decision dimension (8 or 28) after OvR view.

    X_db_sample : (n, 128, 302) mel in dB.
    """
    svm_model.decision_function_shape = "ovr"
    flat = svm_scaler.transform(X_db_sample.reshape(len(X_db_sample), -1))
    s = np.atleast_2d(np.asarray(svm_model.decision_function(flat)))
    return int(s.shape[1])


def train_mlp_meta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
) -> Pipeline:
    """
    MLP meta-classifier: non-linear fusion (ablation variant).

    Small hidden layer (64 units) with L2 regularisation and early stopping.
    Used as an ablation variant to compare linear (LR) vs non-linear (MLP)
    meta-classification.

    Returns
    -------
    Pipeline
        Fitted Pipeline(StandardScaler, MLPClassifier).
    """
    from sklearn.neural_network import MLPClassifier

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    alpha=1e-3,
                    max_iter=2000,
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe
