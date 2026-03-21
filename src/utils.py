"""
utils.py — Reusable helper functions for the Infant State Recognition System.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_features(processed_dir):
    """
    Load pre-extracted features and labels from the processed directory.

    Args:
        processed_dir (str): Path to the processed data directory.

    Returns:
        tuple: (X, y) feature matrix and label array.
    """
    X = np.load(os.path.join(processed_dir, 'features.npy'))
    y = np.load(os.path.join(processed_dir, 'labels.npy'))
    return X, y


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix'):
    """
    Plot a confusion matrix using sklearn's ConfusionMatrixDisplay.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names (list): Optional list of class name strings.
        title (str): Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def ensure_dir(path):
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)
