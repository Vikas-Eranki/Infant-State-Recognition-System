"""
features.py — Spectrogram generation functions for infant cry audio.

This module provides utilities for extracting audio features
such as Mel spectrograms, MFCCs, and other spectral representations.
"""

import numpy as np
import librosa


def generate_spectrogram(audio_path, sr=22050, n_mels=128, hop_length=512):
    """
    Generate a Mel spectrogram from an audio file.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sampling rate.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for STFT.

    Returns:
        np.ndarray: Mel spectrogram in dB scale.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_mfcc(audio_path, sr=22050, n_mfcc=13, hop_length=512):
    """
    Extract MFCC features from an audio file.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sampling rate.
        n_mfcc (int): Number of MFCC coefficients.
        hop_length (int): Hop length for STFT.

    Returns:
        np.ndarray: MFCC feature matrix.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfccs
