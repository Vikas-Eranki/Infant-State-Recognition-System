"""
src/utils.py
------------
Reusable utility functions for the Infant State Recognition System.

Used across all notebooks:
    01_EDA.ipynb
    02_DataPreprocessing.ipynb
    03_feature_engineering.ipynb
    04_baseline_ml.ipynb
    Phase 2 and Phase 3 notebooks

Design notes:
    - All functions are stateless (no global side effects).
    - random_state=42 is enforced via set_random_seed() for reproducibility.
    - Primary evaluation metric is Macro F1-Score throughout (15.9:1 imbalance).
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import soundfile as sf


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset metadata
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_metadata(data_dir):
    """
    Walk a dataset directory and return a metadata DataFrame.

    Expects the directory structure:
        data_dir/
            <category_name>/
                <audio_file>.<ext>

    Parameters
    ----------
    data_dir : str
        Root directory containing one sub-folder per class.

    Returns
    -------
    pd.DataFrame
        Columns: file_path, file_name, category, format, file_size_KB.
        Sorted by category then file_name for deterministic ordering.

    Raises
    ------
    FileNotFoundError
        If data_dir does not exist.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    records = []
    for category_name in sorted(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category_name)
        if not os.path.isdir(category_path):
            continue
        for file_name in sorted(os.listdir(category_path)):
            file_path = os.path.join(category_path, file_name)
            if not os.path.isfile(file_path):
                continue
            records.append({
                "file_path"    : file_path,
                "file_name"    : file_name,
                "category"     : category_name,
                "format"       : os.path.splitext(file_name)[1].lower(),
                "file_size_KB" : round(os.path.getsize(file_path) / 1024.0, 2),
            })

    df = pd.DataFrame(records)
    return df.sort_values(["category", "file_name"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class weights for imbalanced training
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(labels):
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    The formula mirrors sklearn's class_weight='balanced':
        w_i = N / (n_classes * n_i)

    where N is the total number of samples, n_classes is the number of
    unique classes, and n_i is the sample count for class i.

    With a 15.9:1 imbalance ratio (hungry: 397 vs lonely: 25), the lonely
    class receives a weight ~16x larger than hungry, ensuring the SVM and
    any weighted loss function treat all classes as equally important.

    Parameters
    ----------
    labels : array-like of int or str
        Class label for every sample (integer indices or string names).

    Returns
    -------
    dict
        Mapping {class_label: weight}. Keys match the dtype of labels.

    Example
    -------
    >>> weights = get_class_weights(y_train)
    >>> svm = SVC(class_weight=weights)
    """
    labels      = np.array(labels)
    classes     = np.unique(labels)
    n_samples   = len(labels)
    n_classes   = len(classes)
    weights     = {}

    for cls in classes:
        n_cls        = int(np.sum(labels == cls))
        weights[cls] = n_samples / (n_classes * n_cls)

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. Save evaluation results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results_dict, save_path):
    """
    Serialise an evaluation metrics dictionary to a JSON file.

    NumPy scalars and arrays are converted to native Python types before
    serialisation so the file is human-readable and git-diffable.

    Parameters
    ----------
    results_dict : dict
        Arbitrary metrics dictionary. Values may be int, float, np.floating,
        np.integer, np.ndarray, list, str, or nested dicts.
    save_path : str
        Full path to the output .json file. Parent directory is created if
        it does not exist.

    Returns
    -------
    None

    Example
    -------
    >>> save_results({"macro_f1": 0.72, "accuracy": 0.81}, "reports/svm_results.json")
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    serialisable = _convert(results_dict)

    with open(save_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"Results saved: {save_path}  ({os.path.getsize(save_path)} bytes)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Class distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df, save_path, category_col="category",
                             title="Class Distribution", figsize=(12, 5)):
    """
    Plot a horizontal bar chart of sample counts per class and save to disk.

    The plot uses a seaborn viridis palette sorted by ascending count so
    the most imbalanced relationship is immediately visible. Each bar is
    annotated with its raw count. A vertical line marks the mean count
    across classes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the column specified by category_col.
    save_path : str
        Full path to save the .png figure.
    category_col : str
        Name of the column containing class labels. Default "category".
    title : str
        Figure title.
    figsize : tuple
        Matplotlib figure size. Default (12, 5).

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> plot_class_distribution(df_manifest, "reports/class_dist.png")
    """
    counts  = df[category_col].value_counts().sort_values(ascending=True)
    n_cats  = len(counts)
    colors  = sns.color_palette("viridis", n_colors=n_cats)

    fig, ax = plt.subplots(figsize=figsize)
    bars    = ax.barh(counts.index, counts.values, color=colors)

    # Annotate counts on bars
    for bar, val in zip(bars, counts.values):
        ax.text(
            val + counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontweight="bold", fontsize=10
        )

    # Mean line
    mean_count = counts.mean()
    ax.axvline(mean_count, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_count:.0f}")

    ax.set_xlabel("Number of Samples")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Random seed
# ─────────────────────────────────────────────────────────────────────────────

def set_random_seed(seed=42):
    """
    Set random seeds for Python, NumPy, and (if available) TensorFlow/PyTorch.

    Must be called at the top of every notebook and training script to
    guarantee full reproducibility. Seed 42 is used throughout this project.

    Parameters
    ----------
    seed : int
        Random seed. Default 42 (project-wide convention).

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    print(f"Random seed set to {seed} (Python, NumPy"
          + (", TensorFlow" if _tf_available() else "")
          + (", PyTorch" if _torch_available() else "")
          + ")")


def _tf_available():
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 6. Label encoder
# ─────────────────────────────────────────────────────────────────────────────

def create_label_encoder(labels):
    """
    Build deterministic string-to-integer and integer-to-string mappings.

    Classes are sorted alphabetically before encoding so the mapping is
    identical regardless of the order labels appear in the data. This
    matches the convention used in 03_feature_engineering.ipynb and
    label_encoder.json.

    Parameters
    ----------
    labels : array-like of str
        All class label strings in the dataset (duplicates allowed).

    Returns
    -------
    encoder_dict : dict
        {class_name (str): integer_index (int)}
    decoder_dict : dict
        {integer_index (int): class_name (str)}

    Example
    -------
    >>> enc, dec = create_label_encoder(df['category'])
    >>> y_int = np.array([enc[lbl] for lbl in df['category']])
    >>> y_str = [dec[i] for i in y_int]
    """
    classes      = sorted(set(labels))
    encoder_dict = {cls: idx for idx, cls in enumerate(classes)}
    decoder_dict = {idx: cls for cls, idx in encoder_dict.items()}
    return encoder_dict, decoder_dict


# ─────────────────────────────────────────────────────────────────────────────
# 7. Audio file health check
# ─────────────────────────────────────────────────────────────────────────────

def check_audio_file(filepath):
    """
    Load and inspect a single audio file, returning a health summary dict.

    Uses librosa with sr=None to preserve the native sample rate. Catches
    all exceptions so the function never raises — callers inspect the
    'error' key to detect failures.

    Parameters
    ----------
    filepath : str
        Path to the audio file (.wav, .mp3, .ogg, .3gp, etc.).

    Returns
    -------
    dict with keys:
        filepath   (str)   : as provided
        exists     (bool)  : whether the file exists on disk
        sr         (int)   : native sample rate in Hz (None on failure)
        duration   (float) : duration in seconds (None on failure)
        n_samples  (int)   : number of audio samples (None on failure)
        channels   (int)   : 1 for mono, 2 for stereo (None on failure)
        format     (str)   : file extension (lower-case)
        peak_amp   (float) : maximum absolute amplitude (None on failure)
        is_silent  (bool)  : True if peak_amp < 0.01
        error      (str)   : exception message if loading failed, else None

    Example
    -------
    >>> info = check_audio_file("data/raw/hungry/cry_001.wav")
    >>> if info['error']:
    ...     print(f"Bad file: {info['error']}")
    """
    result = {
        "filepath" : filepath,
        "exists"   : os.path.isfile(filepath),
        "sr"       : None,
        "duration" : None,
        "n_samples": None,
        "channels" : None,
        "format"   : os.path.splitext(filepath)[1].lower(),
        "peak_amp" : None,
        "is_silent": None,
        "error"    : None,
    }

    if not result["exists"]:
        result["error"] = "File not found"
        return result

    try:
        # Use soundfile for fast metadata on wav files
        info = sf.info(filepath)
        result["sr"]       = info.samplerate
        result["duration"] = info.duration
        result["n_samples"]= info.frames
        result["channels"] = info.channels
    except Exception:
        pass  # fall through to librosa for non-wav formats

    try:
        y, sr = librosa.load(filepath, sr=None, mono=True)
        result["sr"]        = sr
        result["duration"]  = round(float(librosa.get_duration(y=y, sr=sr)), 4)
        result["n_samples"] = len(y)
        result["channels"]  = 1   # librosa always returns mono
        peak                = float(np.max(np.abs(y)))
        result["peak_amp"]  = round(peak, 6)
        result["is_silent"] = peak < 0.01
    except Exception as e:
        result["error"] = str(e)

    return result
