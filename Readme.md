# Infant State Recognition System (Phase 1)

Translating infant cries into actionable needs using Audio Processing & Machine Learning

## Problem

Infants communicate primarily through crying, but interpreting these cries is:
- Subjective
- Inconsistent
- Dependent on caregiver experience

This project aims to build a system that can automatically classify infant cries into:
- Hunger
- Pain
- Sleep

Ultimately, the goal is to enable real-time, edge-based inference on IoT devices.

## Project Status

**Phase 1: Foundation Stage**

This is the initial phase of the project where we are:
- Understanding the problem domain
- Exploring the dataset
- Designing the pipeline
- Setting up preprocessing and baseline approaches

Detailed model results, optimizations, and deployment will be introduced in later phases.

## Dataset

We are using the Baby Cry Sense Dataset:
- **Link:** [Baby Cry Dataset](https://www.kaggle.com/datasets/mennaahmed23/baby-cry-dataset/data)

### Dataset Overview:
- Audio recordings of infant cries
- Labeled into categories such as:
  - Hunger
  - Pain
  - Sleep

### Initial Observations:
- Class imbalance may exist
- Audio lengths vary
- Presence of background noise

Further EDA and preprocessing are currently in progress.

## Proposed Approach

We follow an audio-to-image transformation pipeline:

`Raw Audio → DSP/FFT → Log-Mel Spectrogram → TF Lite Micro → Depthwise Separable CNN → Output`

### Key Idea

Convert audio signals into Log-Mel Spectrograms, allowing us to treat sound as an image and apply powerful learning techniques.

This forms the foundation for both:
- Traditional ML models (Phase 1)
- CNN-based Deep Learning models (Phase 2)

## Phase 1 Focus

This phase is dedicated to building a strong foundation:

### 1. Literature Review (In Progress)
- Audio classification techniques
- Infant cry analysis research
- Edge AI / TinyML systems

### 2. Data Understanding & EDA (In Progress)
- Waveform analysis
- Spectrogram visualization
- Class distribution analysis

### 3. Feature Engineering (Planned)
- Log-Mel Spectrogram extraction
- Normalization & padding
- Noise handling strategies

### 4. Baseline Models (Planned)
We will implement Advanced ML models such as:
- Support Vector Machine (SVM)
- Random Forest
- Other statistical models

These will serve as benchmarks before introducing deep learning.

### 5. Failure Analysis (Planned)
- Identify model weaknesses
- Analyze misclassifications
- Understand limitations of traditional ML

## Tech Stack (Planned)

- **Audio Processing:** Librosa
- **Machine Learning:** Scikit-learn
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

**Future:**
- TensorFlow Lite Micro
- Embedded deployment (ESP32)

## Repository Structure (Planned)

```text
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── results/
├── requirements.txt
└── README.md
```

## Usage (Work in Progress)

Setup instructions will be finalized as the implementation progresses.

```bash
git clone https://github.com/Vikas-Eranki/Infant-State-Recognition-System
cd infant-state-recognition
pip install -r requirements.txt
```

## Roadmap

### Phase 1 (Current)
- Problem understanding
- Dataset exploration
- Feature engineering setup
- Baseline ML models

### Phase 2
- Depthwise Separable CNN
- Spectrogram-based deep learning
- Regularization & augmentation

### Phase 3
- Hybrid ML + DL system
- Ablation studies
- Deployment on ESP32 (TinyML)

## Team

- Prashant Kumar - 230101
- Eranki Sai Vikas - 230121