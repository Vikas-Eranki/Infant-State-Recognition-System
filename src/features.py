"""
src/features.py
---------------
Reusable audio feature extraction functions for the Infant State Recognition System.

Used by:
    03_feature_engineering.ipynb  -- batch spectrogram generation
    04_baseline_ml.ipynb          -- SVM feature pipeline (via flattening)
    Phase 2 DS-CNN notebook       -- CNN input pipeline
    Phase 3 hybrid notebook       -- shared feature extraction stage

Pipeline per file:
    Raw .wav (22050 Hz, 7 s, peak-normalised)
        -> load_and_preprocess()
        -> generate_logmel_spectrogram()
        -> np.ndarray (128, 302)  float32, range ~[-80, 0] dB

Output shape derivation (centre-padding, librosa default):
    TARGET_SAMPLES = 154350
    padded         = 154350 + 2*(2048//2) = 156398
    T_frames       = floor((156398 - 2048) / 512) + 1 = 302
    Final shape    = (128, 302)
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Project-wide constants (must match 02_DataPreprocessing and 03_feature_engineering)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR       = 22050
TARGET_DURATION = 7.0
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)   # 154350

N_MELS          = 128
N_FFT           = 2048
HOP_LENGTH      = 512
FMAX            = 8000
EPSILON         = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and preprocess a single audio file
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(filepath, target_sr=TARGET_SR, target_duration=TARGET_DURATION):
    """
    Load one audio file and apply the full preprocessing pipeline.

    Steps applied in order:
    1. Load with librosa at target_sr (resamples if native sr differs).
    2. Convert to mono (librosa default).
    3. Pad with zeros at the end if shorter than target_duration.
       Post-padding (not pre-padding) preserves cry onset alignment.
    4. Trim to exactly target_samples if longer.
    5. Peak-normalise: y = y / (max|y| + epsilon).
       Removes gain variability while preventing division by zero.

    Parameters
    ----------
    filepath : str
        Path to a supported audio file (.wav, .mp3, .ogg, .3gp, etc.).
    target_sr : int
        Output sample rate in Hz. Default 22050 (project-wide standard).
        22050 Hz was chosen as a neutral resampling target between the two
        source populations (8000 Hz Donate-a-Cry recordings and 44100 Hz
        re-recorded samples) identified in the EDA sampling-rate analysis.
    target_duration : float
        Output duration in seconds. Default 7.0 s (95th percentile of the
        raw dataset duration distribution -- retains 94.8% of files intact).

    Returns
    -------
    np.ndarray, shape (target_samples,), dtype float32
        Peak-normalised mono waveform at target_sr Hz.
        target_samples = int(target_sr * target_duration) = 154350.

    Raises
    ------
    Exception
        Propagates any librosa loading error so callers can log failures.

    Example
    -------
    >>> audio = load_and_preprocess("data/processed/audio_clean/hungry/cry_001.wav")
    >>> audio.shape
    (154350,)
    """
    target_samples = int(target_sr * target_duration)

    # Load (resample to target_sr if needed; librosa handles all formats)
    y, _ = librosa.load(filepath, sr=target_sr, mono=True)

    # Duration normalisation
    if len(y) < target_samples:
        # Post-pad with zeros (silence) to preserve temporal alignment of onset
        y = np.pad(y, (0, target_samples - len(y)), mode="constant")
    elif len(y) > target_samples:
        # Trim from the end (trailing silence / repeated cry patterns)
        y = y[:target_samples]

    # Peak normalisation: maps to [-1, 1] regardless of microphone gain
    peak = np.max(np.abs(y))
    y    = y / (peak + EPSILON)

    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generate Log-Mel Spectrogram from a waveform
# ─────────────────────────────────────────────────────────────────────────────

def generate_logmel_spectrogram(audio, sr=TARGET_SR, n_mels=N_MELS,
                                n_fft=N_FFT, hop_length=HOP_LENGTH,
                                fmax=FMAX):
    """
    Convert a pre-loaded waveform to a Log-Mel Spectrogram.

    Mathematical pipeline
    ---------------------
    Step 1 -- STFT (Short-Time Fourier Transform):
        X[k, t] = sum_{n=0}^{N-1} x[n + t*H] * w[n] * exp(-j*2*pi*k*n/N)
        |X[k,t]|^2 gives the power spectrogram.
        n_fft=2048: window of 93 ms at 22050 Hz -- long enough to resolve
        the 400 Hz infant cry fundamental (needs >= 2.5 ms), short enough
        to track temporal changes in cry structure.
        hop_length=512: stride of 23 ms = 75% frame overlap -- standard for
        audio classification; balances temporal resolution and computation.

    Step 2 -- Mel Filterbank:
        m = 2595 * log10(1 + f/700)
        n_mels=128 triangular filters map the linear frequency axis to the
        Mel scale, approximating human auditory resolution.
        fmax=8000 Hz: covers full infant cry harmonic range (fundamental
        250-600 Hz, harmonics up to ~4 kHz); discards noise above 8 kHz
        from upsampled 8000 Hz source files.

    Step 3 -- Log compression (dB scale):
        S_dB[m, t] = 10 * log10(S[m,t] / S_max)
        ref=np.max normalises each clip's energy to the range [-80, 0] dB.
        Log compression mirrors human loudness perception (Weber-Fechner law)
        and makes quiet formants distinguishing cry types visible.

    Parameters
    ----------
    audio : np.ndarray, shape (target_samples,)
        Peak-normalised mono waveform at sr Hz.
        Produced by load_and_preprocess().
    sr : int
        Sample rate. Must equal TARGET_SR (22050 Hz).
    n_mels : int
        Number of Mel filterbank channels. Default 128.
    n_fft : int
        FFT window size in samples. Default 2048 (93 ms at 22050 Hz).
    hop_length : int
        STFT stride in samples. Default 512 (23 ms, 75% overlap).
    fmax : float
        Highest frequency included in Mel filterbank. Default 8000 Hz.

    Returns
    -------
    np.ndarray, shape (n_mels, T_frames), dtype float32
        Log-Mel spectrogram in dB.
        For default parameters: shape = (128, 302).
        Value range: approximately [-80, 0] dB.

    Example
    -------
    >>> audio = load_and_preprocess("cry_001.wav")
    >>> spec  = generate_logmel_spectrogram(audio)
    >>> spec.shape
    (128, 302)
    """
    mel_spec     = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build full dataset from manifest CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(manifest_path, audio_base_dir,
                  target_sr=TARGET_SR, target_duration=TARGET_DURATION,
                  n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                  fmax=FMAX, show_progress=True):
    """
    Generate Log-Mel Spectrograms for every file listed in a manifest CSV.

    Reads clean_manifest.csv produced by 02_DataPreprocessing.ipynb,
    calls load_and_preprocess() and generate_logmel_spectrogram() per file,
    and returns stacked NumPy arrays ready for model training.

    Audio paths are rebuilt from audio_base_dir + category + file_name so
    this function works both locally and on Kaggle (where the manifest's
    original absolute paths are invalid).

    Parameters
    ----------
    manifest_path : str
        Path to clean_manifest.csv.
        Required columns: file_name, category, is_augmented.
    audio_base_dir : str
        Root directory containing one sub-folder per category.
        e.g. "data/processed/audio_clean" or
             "/kaggle/input/.../audio_clean"
    target_sr : int
        Sample rate for loading. Default 22050 Hz.
    target_duration : float
        Clip duration in seconds. Default 7.0 s.
    n_mels : int
        Mel bins. Default 128.
    n_fft : int
        FFT window size. Default 2048.
    hop_length : int
        STFT stride. Default 512.
    fmax : float
        Max frequency for Mel filterbank. Default 8000 Hz.
    show_progress : bool
        Display tqdm progress bar. Default True.

    Returns
    -------
    X : np.ndarray, shape (N, n_mels, T_frames), dtype float32
        Spectrogram array. For default params: (N, 128, 302).
        Failed files are excluded; N may be less than len(manifest).
    y : np.ndarray, shape (N,), dtype int32
        Integer class labels aligned with X.
    label_encoder : dict
        {class_name (str): integer_index (int)}.
        Sorted alphabetically -- identical to label_encoder.json.
    failed : list of dict
        Records for files that could not be processed.
        Keys: index, file_name, category, error.

    Example
    -------
    >>> X, y, enc, failed = build_dataset(
    ...     manifest_path  = "data/processed/clean_manifest.csv",
    ...     audio_base_dir = "data/processed/audio_clean",
    ... )
    >>> X.shape
    (1334, 128, 302)
    """
    df = pd.read_csv(manifest_path)

    # Build label encoder from manifest (sorted = deterministic)
    classes       = sorted(df["category"].unique())
    label_encoder = {cls: idx for idx, cls in enumerate(classes)}

    # Pre-allocate arrays
    n_files        = len(df)
    target_samples = int(target_sr * target_duration)
    import math
    padded    = target_samples + 2 * (n_fft // 2)
    t_frames  = math.floor((padded - n_fft) / hop_length) + 1

    X       = np.zeros((n_files, n_mels, t_frames), dtype=np.float32)
    y       = np.full(n_files, -1, dtype=np.int32)
    failed  = []

    iterator = tqdm(df.iterrows(), total=n_files, desc="Building dataset") \
               if show_progress else df.iterrows()

    for i, (_, row) in enumerate(iterator):
        audio_path = os.path.join(audio_base_dir, row["category"], row["file_name"])
        try:
            audio    = load_and_preprocess(audio_path, target_sr, target_duration)
            X[i]     = generate_logmel_spectrogram(
                           audio, sr=target_sr, n_mels=n_mels,
                           n_fft=n_fft, hop_length=hop_length, fmax=fmax)
            y[i]     = label_encoder[row["category"]]
        except Exception as e:
            failed.append({
                "index"    : i,
                "file_name": row["file_name"],
                "category" : row["category"],
                "error"    : str(e),
            })

    # Remove failed entries
    if failed:
        valid = y != -1
        X     = X[valid]
        y     = y[valid]
        print(f"Warning: {len(failed)} files failed and were excluded.")
        for f in failed:
            print(f"  [{f['category']}] {f['file_name']}: {f['error']}")

    return X, y.astype(np.int32), label_encoder, failed
