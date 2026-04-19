"""
src/model_dl.py
---------------
Reusable Deep Learning functions for the Infant State Recognition System.
Phase 2 — Depthwise Separable CNN (Model B) on Log-Mel Spectrograms.

Used by:
    05_dscnn.ipynb                -- main DS-CNN training + ablation + Grad-CAM
    Phase 3 hybrid notebook       -- loads best_dscnn.keras for fusion

Design notes:
    - All functions are stateless; random_state is enforced by the caller
      via set_random_seed() from src/utils.py.
    - Primary evaluation metric is Macro F1-Score (15.9:1 imbalance).
      Training uses MacroF1Score (batch-approximated) for EarlyStopping;
      final reported numbers use sklearn.metrics on the full test set.
    - Mirrors the API style of src/model_ml.py (Phase 1 SVM helpers).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Project-wide constants (must match src/features.py and 05_dscnn.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

INPUT_SHAPE = (128, 302, 1)   # (n_mels, T_frames, channels)
N_CLASSES   = 8


# ─────────────────────────────────────────────────────────────────────────────
# 1. Macro F1-Score as a Keras metric (for EarlyStopping monitor)
# ─────────────────────────────────────────────────────────────────────────────

class MacroF1Score(tf.keras.metrics.Metric):
    """
    Batch-averaged Macro F1-Score for multi-class classification.

    Required by EarlyStopping(monitor='val_macro_f1') because TensorFlow has
    no built-in macro F1. Computes per-class F1 from TP/FP/FN within each
    batch, averages across classes (macro), then averages across batches
    over the epoch. This is an approximation -- final reported numbers
    use sklearn.metrics.f1_score on the full test set.

    Mathematical definition
    -----------------------
        precision_c = TP_c / (TP_c + FP_c + epsilon)
        recall_c    = TP_c / (TP_c + FN_c + epsilon)
        F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + epsilon)
        Macro F1    = (1 / C) * sum_c F1_c

    where C = num_classes and epsilon = 1e-7 prevents division by zero.

    Parameters
    ----------
    num_classes : int
        Number of target classes. Default 8 (this project).
    name : str
        Metric name used in training logs and callbacks. Default 'macro_f1'.
    """

    def __init__(self, num_classes=N_CLASSES, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.f1_scores   = self.add_weight(name="f1",    initializer="zeros")
        self.count       = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=1)
        y_true_classes = tf.argmax(y_true, axis=1)
        f1 = 0.0
        for c in range(self.num_classes):
            tp = tf.reduce_sum(tf.cast(
                (y_pred_classes == c) & (y_true_classes == c), tf.float32))
            fp = tf.reduce_sum(tf.cast(
                (y_pred_classes == c) & (y_true_classes != c), tf.float32))
            fn = tf.reduce_sum(tf.cast(
                (y_pred_classes != c) & (y_true_classes == c), tf.float32))
            precision = tp / (tp + fp + 1e-7)
            recall    = tp / (tp + fn + 1e-7)
            f1       += 2 * precision * recall / (precision + recall + 1e-7)
        self.f1_scores.assign_add(f1 / self.num_classes)
        self.count.assign_add(1.0)

    def result(self):
        return self.f1_scores / (self.count + 1e-7)

    def reset_state(self):
        self.f1_scores.assign(0.0)
        self.count.assign(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_dscnn(input_shape=INPUT_SHAPE, n_classes=N_CLASSES):
    """
    Build a Depthwise Separable CNN for Log-Mel Spectrogram classification.

    Architecture
    ------------
        Conv2D(32, 3x3) + BN + ReLU + MaxPool
          -> DS Block 1: DepthwiseConv(3x3) + BN + ReLU + Conv(64, 1x1) + BN + ReLU + MaxPool
          -> DS Block 2: DepthwiseConv(3x3) + BN + ReLU + Conv(128, 1x1) + BN + ReLU + MaxPool
          -> DS Block 3: DepthwiseConv(3x3) + BN + ReLU + Conv(256, 1x1) + BN + ReLU
          -> GlobalAveragePooling2D + Dropout(0.15) + Dense(128, ReLU) + Dropout(0.1) + Dense(softmax)

    Regularisation (tuned for 974-sample training set)
    --------------------------------------------------
        Conv blocks: NO dropout (BatchNorm provides implicit regularisation).
        Head:        Dropout(0.15) after GAP and Dropout(0.1) after Dense(128).

    Phase 1 failure mode fixes
    --------------------------
        FM1 (translation sensitivity): Conv2D + MaxPool -> translation equivariance.
        FM2 (temporal blindness):      3x3 kernels span multiple time frames.
        FM3 (silence amplification):   GAP averages spatial dims so zero-padded
                                       frames contribute near-zero activation.

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape without batch dim. Default (128, 302, 1)
        matches Log-Mel spectrograms produced by src/features.py.
    n_classes : int
        Number of target classes. Default 8.

    Returns
    -------
    tf.keras.Model  (uncompiled) DS-CNN.

    Example
    -------
    >>> model = build_dscnn()
    >>> model.compile(optimizer='adam', loss='categorical_crossentropy',
    ...               metrics=['accuracy', MacroF1Score()])
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Initial standard conv -- learns low-level spectral edge features
    x = layers.Conv2D(32, (3, 3), padding="same",
                      kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # DS Block 1
    x = layers.DepthwiseConv2D((3, 3), padding="same",
                               depthwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (1, 1), kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # DS Block 2
    x = layers.DepthwiseConv2D((3, 3), padding="same",
                               depthwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (1, 1), kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # DS Block 3 (no MaxPool -- deepest features)
    x = layers.DepthwiseConv2D((3, 3), padding="same",
                               depthwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (1, 1), kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Classification head
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.15)(x)
    x       = layers.Dense(128, activation="relu",
                           kernel_initializer="he_normal")(x)
    x       = layers.Dropout(0.1)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="DS_CNN")


def build_standard_cnn(input_shape=INPUT_SHAPE, n_classes=N_CLASSES):
    """
    Build an equivalent CNN using standard Conv2D (no depthwise separation).

    Used as ablation variant B4 to quantify the cost of depthwise separation.
    Has the same channel progression (32 -> 64 -> 128 -> 256) and identical
    head as build_dscnn -- only the 3x3 convolutions differ.

    Standard Conv2D cost per block: D_K^2 * M * N * D_F^2
    DS-CNN cost per block:          D_K^2 * M * D_F^2 + M * N * D_F^2

    Reduction ratio: 1/N + 1/D_K^2 (approximately 1/8 for our blocks).

    Parameters
    ----------
    input_shape : tuple of int
    n_classes : int

    Returns
    -------
    tf.keras.Model  (uncompiled) Standard CNN.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(32,  (3, 3), padding="same",
                      kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64,  (3, 3), padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.15)(x)
    x       = layers.Dense(128, activation="relu",
                           kernel_initializer="he_normal")(x)
    x       = layers.Dropout(0.1)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="Standard_CNN")


def build_dscnn_no_dropout(input_shape=INPUT_SHAPE, n_classes=N_CLASSES):
    """
    Build DS-CNN with all Dropout layers removed (ablation variant B3).

    Used to quantify how much head dropout contributes to generalisation
    on this 974-sample training set. Architecture is otherwise identical
    to build_dscnn.

    Parameters
    ----------
    input_shape : tuple of int
    n_classes : int

    Returns
    -------
    tf.keras.Model  (uncompiled).
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same",
                      kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    for out_ch, pool in [(64, True), (128, True), (256, False)]:
        x = layers.DepthwiseConv2D((3, 3), padding="same",
                                   depthwise_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(out_ch, (1, 1), kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if pool:
            x = layers.MaxPooling2D((2, 2))(x)

    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(128, activation="relu",
                           kernel_initializer="he_normal")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="DS_CNN_no_dropout")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def compile_model(model, learning_rate=5e-4, label_smoothing=0.1,
                  num_classes=N_CLASSES):
    """
    Compile a Keras model with the project-standard training configuration.

    Adam optimizer at lr=5e-4 (conservative for small datasets).
    Categorical cross-entropy with label_smoothing=0.1 softens hard targets
    and acts as a mild regulariser without the gradient instability of
    per-sample class weights.

    Parameters
    ----------
    model : tf.keras.Model
        Uncompiled model.
    learning_rate : float
        Adam learning rate. Default 5e-4.
    label_smoothing : float
        CategoricalCE label smoothing factor. Default 0.1.
    num_classes : int
        For the MacroF1Score metric. Default 8.

    Returns
    -------
    tf.keras.Model  The same model, compiled in place and returned for chaining.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing),
        metrics=["accuracy", MacroF1Score(num_classes=num_classes)],
    )
    return model


def default_callbacks(checkpoint_path, patience_es=25, patience_lr=10,
                      lr_factor=0.5, min_lr=1e-6):
    """
    Return the project-standard callback list for DS-CNN training.

    EarlyStopping(patience=25) on val_macro_f1 -- generous for small dataset.
    ReduceLROnPlateau(factor=0.5, patience=10) on val_loss.
    ModelCheckpoint on val_macro_f1 saves the best model.

    Parameters
    ----------
    checkpoint_path : str
        Path where best_weights.keras will be written.
    patience_es : int
        EarlyStopping patience (epochs without improvement). Default 25.
    patience_lr : int
        ReduceLROnPlateau patience. Default 10.
    lr_factor : float
        Multiplier applied to LR on plateau. Default 0.5.
    min_lr : float
        Minimum learning rate. Default 1e-6.

    Returns
    -------
    list of tf.keras.callbacks.Callback
    """
    return [
        EarlyStopping(
            monitor="val_macro_f1", patience=patience_es,
            restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(
            monitor="val_loss", factor=lr_factor,
            patience=patience_lr, min_lr=min_lr, verbose=1),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_macro_f1",
            save_best_only=True, mode="max", verbose=1),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation (sklearn-based -- authoritative numbers)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, class_names, zero_division=0):
    """
    Compute evaluation metrics using sklearn on the full test set.

    Final reported numbers come from this function, not from the Keras
    MacroF1Score metric (which is batch-averaged during training and
    used only for EarlyStopping). Mirrors the signature of the Phase 1
    helper in src/model_ml.py for consistency.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model with softmax output.
    X_test : np.ndarray, shape (n, 128, 302, 1)
        Normalised spectrograms.
    y_test : np.ndarray, shape (n,)
        Integer class labels.
    class_names : list of str
        Class names in label-index order.
    zero_division : int
        Returned when precision/recall is undefined. Default 0.

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, weighted_f1, per_class_f1,
        y_pred, y_pred_prob, report_str.
    """
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    return {
        "accuracy"    : accuracy_score(y_test, y_pred),
        "macro_f1"    : f1_score(y_test, y_pred, average="macro",
                                 zero_division=zero_division),
        "weighted_f1" : f1_score(y_test, y_pred, average="weighted",
                                 zero_division=zero_division),
        "per_class_f1": f1_score(y_test, y_pred, average=None,
                                 zero_division=zero_division),
        "y_pred"      : y_pred,
        "y_pred_prob" : y_pred_prob,
        "report_str"  : classification_report(
                            y_test, y_pred,
                            target_names=class_names,
                            zero_division=zero_division, digits=4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Grad-CAM (Pillar 4 interpretability)
# ─────────────────────────────────────────────────────────────────────────────

def get_last_conv_layer(model):
    """
    Return the name of the last DepthwiseConv2D layer in the model.

    Grad-CAM prefers DepthwiseConv2D (operates spatially) over 1x1 pointwise
    Conv2D because its feature maps retain spatial structure suitable for
    visualisation. Falls back to any Conv2D if no DepthwiseConv2D exists.

    Parameters
    ----------
    model : tf.keras.Model

    Returns
    -------
    str  Layer name.

    Raises
    ------
    ValueError  If the model has no convolutional layer.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, layers.DepthwiseConv2D):
            return layer.name
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def generate_gradcam(model, spectrogram, class_idx, layer_name=None,
                     out_size=(128, 302)):
    """
    Compute a Grad-CAM heatmap for a single spectrogram.

    Mathematical pipeline (Selvaraju et al., 2017)
    ----------------------------------------------
    alpha_k^c = (1/Z) * sum_{i,j} d(y^c) / d(A_{i,j}^k)
    L^c       = ReLU(sum_k alpha_k^c * A^k)

    Steps:
        1. Forward pass through a sub-model that returns both the target
           conv layer activations A and the final predictions y.
        2. Compute gradients d(y^c) / dA for the chosen class c.
        3. Global-average-pool gradients across spatial dims -> alpha_k.
        4. Weighted sum of feature maps, apply ReLU (positive influence).
        5. Normalise to [0, 1] and resize to the spectrogram dimensions.

    Parameters
    ----------
    model : tf.keras.Model
        Trained DS-CNN.
    spectrogram : np.ndarray, shape (128, 302, 1)
        One normalised input sample (no batch dim).
    class_idx : int
        Target class for the gradient (0 to n_classes-1).
    layer_name : str or None
        Convolutional layer to visualise. If None, uses get_last_conv_layer().
    out_size : tuple of int
        Output heatmap size. Default (128, 302) matches input spectrograms.

    Returns
    -------
    np.ndarray, shape out_size, dtype float32
        Heatmap normalised to [0, 1].
    """
    if layer_name is None:
        layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    spec_tensor = tf.cast(spectrogram[np.newaxis], tf.float32)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(spec_tensor)
        tape.watch(conv_output)
        class_score = predictions[:, class_idx]

    grads        = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (n_filters,)

    conv_output_val = conv_output[0]                              # (H, W, n_filters)
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_output_val),
                        axis=-1)                                  # (H, W)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)

    cam_4d      = cam[tf.newaxis, ..., tf.newaxis]
    cam_resized = tf.image.resize(cam_4d, out_size).numpy()[0, :, :, 0]
    return cam_resized.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Confusion matrix plot (parity with Phase 1 helper)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names,
                           save_path=None, figsize=(20, 7), title_suffix=""):
    """
    Plot raw-count and row-normalised confusion matrices side by side.

    Same signature as src/model_ml.plot_confusion_matrix so Phase 1 and
    Phase 2 figures remain visually consistent.

    Parameters
    ----------
    y_true, y_pred : array-like of int
    class_names    : list of str in label-index order
    save_path      : str or None
    figsize        : tuple
    title_suffix   : str  Appended to the figure suptitle.

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
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                    exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Training curves plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, save_path=None, figsize=(20, 5)):
    """
    Plot loss, Macro F1 and learning-rate curves from a Keras History.

    Handles both TF < 2.16 (history key 'lr') and TF >= 2.16
    (history key 'learning_rate') transparently.

    Parameters
    ----------
    history : tf.keras.callbacks.History
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(history.history["loss"],     label="Train Loss",  linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss",    linewidth=2)
    axes[0].set_title("Loss Curves", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical Cross-Entropy")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["macro_f1"],     label="Train Macro F1", linewidth=2)
    axes[1].plot(history.history["val_macro_f1"], label="Val Macro F1",   linewidth=2)
    axes[1].set_title("Macro F1 Curves (Primary Metric)",
                      fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1-Score")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    lr_key = "learning_rate" if "learning_rate" in history.history else (
             "lr" if "lr" in history.history else None)
    if lr_key is not None:
        axes[2].plot(history.history[lr_key], linewidth=2, color="green")
        axes[2].set_title("Learning Rate Schedule",
                          fontweight="bold", fontsize=13)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "LR history\nnot available",
                     ha="center", va="center", fontsize=14)
        axes[2].set_title("Learning Rate Schedule",
                          fontweight="bold", fontsize=13)

    plt.suptitle("DS-CNN Training -- Proof of Regularisation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                    exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. TFLite conversion for edge deployment
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_tflite(model, save_path, optimize=True):
    """
    Convert a Keras model to TFLite with optional dynamic-range quantisation.

    Post-training dynamic-range quantisation converts Float32 weights to INT8,
    achieving roughly 4x size reduction with minimal accuracy loss -- critical
    for the Phase 3 ESP32 deployment target (512 KB SRAM).

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    save_path : str
        Path to write the .tflite FlatBuffer.
    optimize : bool
        If True, applies tf.lite.Optimize.DEFAULT (dynamic-range quantisation).
        Default True.

    Returns
    -------
    dict with keys:
        tflite_bytes : bytes -- the serialised model
        size_kb      : float -- size in kilobytes
        save_path    : str
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(tflite_model)

    return {
        "tflite_bytes": tflite_model,
        "size_kb"     : len(tflite_model) / 1024.0,
        "save_path"   : save_path,
    }
