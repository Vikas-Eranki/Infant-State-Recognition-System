"""
model_ml.py — Baseline ML models (SVM / GMM) for infant state classification.

This module provides functions to train and evaluate traditional
machine learning classifiers on extracted audio features.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def train_svm(X, y, test_size=0.2, random_state=42, kernel='rbf', C=1.0):
    """
    Train an SVM classifier.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        test_size (float): Fraction of data used for testing.
        random_state (int): Random seed.
        kernel (str): SVM kernel type.
        C (float): Regularization parameter.

    Returns:
        tuple: (trained model, scaler, X_test, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel=kernel, C=C, random_state=random_state)
    model.fit(X_train, y_train)

    print("SVM Classification Report:")
    print(classification_report(y_test, model.predict(X_test)))

    return model, scaler, X_test, y_test


def train_gmm(X, y, n_components=4, random_state=42):
    """
    Train a Gaussian Mixture Model per class.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        n_components (int): Number of Gaussian components per class.
        random_state (int): Random seed.

    Returns:
        dict: Mapping from class label to fitted GMM.
    """
    gmm_models = {}
    classes = np.unique(y)
    for cls in classes:
        X_cls = X[y == cls]
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(X_cls)
        gmm_models[cls] = gmm
    return gmm_models
