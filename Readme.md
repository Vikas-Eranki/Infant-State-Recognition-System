# Infant State Recognition System

> Classifying infant cry states (hungry, belly pain, discomfort, tired, and more)
> from audio using Log-Mel Spectrograms on an edge-deployable pipeline.
> Built for the **Advanced ML and Deep Learning** course — 3-phase evaluation.

---

## Project Overview

This project bridges **probabilistic machine learning** and **deep learning** to
build a lightweight infant cry classifier suitable for deployment on embedded
microcontrollers (ESP32 via TensorFlow Lite Micro). The system translates raw
audio from an infant monitor into one of 8 actionable states in real time.

The architecture is evaluated across three phases:

| Model | Type | Phase |
|---|---|---|
| **Model A** — SVM Baseline | RBF-SVM on flattened Log-Mel features | Phase 1 ✅ |
| **Model B** — DS-CNN | Depthwise Separable CNN on spectrogram images | Phase 2 🔄 |
| **Model C** — Hybrid | SVM preprocessing stage + DS-CNN classification | Phase 3 🔄 |

Model C must definitively outperform both A and B on Macro F1-Score.

---

## Repository Structure

```
infant-state-recognition/
├── data/
│   └── processed/
│       ├── clean_manifest.csv           # one row per processed audio file
│       ├── preprocessing_params.json    # all preprocessing hyperparameters
│       └── augmentation_log.csv         # record of every synthetic sample
├── notebooks/
│   ├── 01_EDA.ipynb                     # exploratory data analysis (21 sections)
│   ├── 02_DataPreprocessing.ipynb       # full preprocessing pipeline (12 sections)
│   ├── 03_feature_engineering.ipynb     # Log-Mel spectrogram generation
│   └── 04_baseline_ml.ipynb             # SVM Model A training + failure analysis
├── src/
│   ├── __init__.py
│   ├── utils.py                         # shared utility functions
│   ├── features.py                      # spectrogram generation functions
│   └── model_ml.py                      # reusable SVM training and evaluation
├── docs/
│   ├── literature_review.md             # IEEE-format, 11 references
│   └── dataset_readme.md                # raw + processed dataset documentation
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** Large binary files (`audio_clean/`, `X.npy`, `y.npy`) are gitignored.
> All notebooks are designed to run on Kaggle using the preprocessed dataset at
> [kaggle.com/datasets/noxpie/baby-cry-preprocessed-dataset](https://www.kaggle.com/datasets/noxpie/baby-cry-preprocessed-dataset).

---

## Dataset

| Property | Value |
|---|---|
| **Raw dataset** | [Baby Cry Sense Dataset](https://www.kaggle.com/datasets/mennaahmed23/baby-cry-sense-dataset/data) |
| **Derived from** | Donate-a-Cry corpus (Veres, 2015) |
| **Classes** | 8 (belly pain, burping, cold_hot, discomfort, hungry, lonely, scared, tired) |
| **Raw files** | 1,126 audio recordings |
| **Processed files** | 1,334 (after QC, conversion, augmentation) |
| **License** | CC BY-SA 4.0 |

**Critical imbalance:** 15.9:1 ratio (hungry: 397 files vs lonely: 25 files).
Macro F1-Score is used as the primary metric throughout — not accuracy.

See [`docs/dataset_readme.md`](docs/dataset_readme.md) for full dataset documentation.

---

## Evaluation Structure

The project is graded on **5 Pillars** across 3 phases:

| Pillar | Requirement |
|---|---|
| 1 — Advanced ML Depth | Correct application of SVM/GMM with mathematical explanation |
| 2 — Deep Learning Rigor | DS-CNN with Flash Attention, Dropout, weight initialisation |
| 3 — Integration & Innovation | Hybrid must logically combine both domains and outperform both baselines |
| 4 — Technical Validation | Rigorous ablation studies and failure analysis with mathematical reasoning |
| 5 — Documentation & Reproducibility | Clean repo, requirements.txt, reproducible README |

| Phase | Weight | Due | Status |
|---|---|---|---|
| Phase 1 — Foundation & Advanced ML | 30% | Mar 23–27, 2026 | ✅ Complete |
| Phase 2 — Deep Learning Architecture | 30% | Apr 20–24, 2026 | 🔄 Upcoming |
| Phase 3 — Hybrid Integration + ESP32 | 40% | May 4–8, 2026 | 🔄 Upcoming |

---

## Phase 1 — Complete ✅

### What Was Delivered

Phase 1 establishes the academic foundation, dataset understanding, feature
engineering pipeline, and the Advanced ML baseline (Model A).

#### Literature Review

Located at [`docs/literature_review.md`](docs/literature_review.md).
IEEE-format review with 11 references across three anchor papers:

| Paper | Contribution | Role in This Project |
|---|---|---|
| Liu et al. (2019) — IEEE/CAA JAS | SVM + MFCC baseline, 5 classes | Model A baseline design |
| Abbaskhah et al. (2023) — Elsevier BSPC | CNN beats SVM on identical features | Justification for Phase 2 |
| Hammoud et al. (2024) — Frontiers in AI | SOTA 96.39% on 5 classes, random forest | Positions our 8-class contribution |

**Key argument:** prior work is ceiling-bounded by hand-crafted features and
flat classifiers. Our work is the first to control for the sampling-rate artifact
(undiscovered in all prior publications on this corpus) and the first to address
all 8 classes with an edge-deployable CNN.

#### Exploratory Data Analysis (`01_EDA.ipynb`)

21 sections covering full dataset characterisation.

**Key findings:**

| Finding | Detail |
|---|---|
| Class imbalance | 15.9:1 ratio (hungry: 397 vs lonely: 25) |
| Sampling rate artifact | `scared` is 81.8% at 44,100 Hz; `hungry` is 100% at 8,000 Hz — model could learn recording quality instead of cry patterns |
| Duration range | 4.13 s to 8.73 s; 95th percentile = 7.02 s → TARGET_DURATION = 7.0 s |
| Poor linear separability | PCA captures only 38.6% variance in 2 components; 23 components needed for 95% |
| Nearest-neighbour purity | 0.350 overall (random baseline = 0.125) — confirms classes overlap heavily |
| Top confused pairs | discomfort↔hungry (1074 overlaps), hungry↔tired (976), cold_hot↔hungry (899) |
| Data quality | 0 corrupted files |

#### Preprocessing Pipeline (`02_DataPreprocessing.ipynb`)

12 sections, fully executed. All decisions justified by EDA findings.

| Parameter | Value | Justification |
|---|---|---|
| `TARGET_SR` | 22,050 Hz | Neutral between 8 kHz and 44.1 kHz source populations |
| `TARGET_DURATION` | 7.0 s | 95th percentile of duration distribution |
| `TARGET_SAMPLES` | 154,350 | 7.0 × 22,050 |
| `AMPLITUDE_THRESHOLD` | 0.01 | Removes near-silent files |
| `MIN_SAMPLES_PER_CLASS` | 130 | Minimum class size after augmentation |

**Pipeline traceability:**

| Step | Count |
|---|---|
| Raw files loaded | 1,126 |
| Failed raw validation | 0 |
| Non-.wav files converted | 87 (72 `.3gp`, 8 `.ogg`, 7 `.mp3`) |
| Passed quality check | 1,125 (1 lonely file rejected — near-silent) |
| Augmented samples added | 209 (burping +6, lonely +106, scared +97) |
| **Final saved files** | **1,334** |
| Verification passed | 1,334 / 1,334 |

**Augmentation techniques:** time stretching (rate 0.8–1.2), pitch shifting
(±2 semitones), Gaussian noise injection, time shifting with wrapping.

**Important limitation:** `lonely` is 81.5% synthetic (106 from 24 originals)
and `scared` is 74.6% synthetic (97 from 33 originals). Per-class F1 scores for
these classes should be interpreted with caution.

#### Feature Engineering (`03_feature_engineering.ipynb`)

Converts all 1,334 processed audio files to Log-Mel Spectrograms.

**Mathematical pipeline:**

```
Raw audio (154,350 samples at 22,050 Hz)
  → STFT (n_fft=2048, hop=512, centre-padding)
  → Mel filterbank (n_mels=128, fmax=8,000 Hz)
  → Log compression (ref=np.max)
  → Log-Mel Spectrogram: shape (128, 302), range ~[−80, 0] dB
```

| Parameter | Value | Justification |
|---|---|---|
| `n_fft` | 2,048 | 93 ms window — resolves 400 Hz fundamental, tracks temporal changes |
| `hop_length` | 512 | 23 ms stride, 75% overlap — standard for audio classification |
| `n_mels` | 128 | Matches human critical-band resolution |
| `fmax` | 8,000 Hz | Covers full infant cry harmonic range; excludes noise from upsampled 8 kHz files |

**Output arrays saved to `/kaggle/working/`:**

| File | Shape | Size |
|---|---|---|
| `X.npy` | (1,334, 128, 302) float32 | 206.3 MB |
| `y.npy` | (1,334,) int32 | ~5 KB |
| `label_encoder.json` | 8 class mappings | <1 KB |

#### Baseline Model — SVM Model A (`04_baseline_ml.ipynb`)

**Configuration:**

```python
SVC(kernel='rbf', C=10, gamma='scale',
    class_weight='balanced',
    decision_function_shape='ovo',
    random_state=42)
```

- **Input:** 38,656-dim flattened spectrogram vectors (128 × 302)
- **Split:** augmentation-aware stratified 80/20 — augmented files only in train,
  zero leakage guaranteed
- **Scaling:** StandardScaler fit on train only

**Results (Phase 1 Ablation Table — Artifact 01):**

| Model | Configuration | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| **A — SVM Baseline** | RBF-SVM, C=10, flattened 38,656-dim | 21.78% | **0.4029** | 21.98% |
| B — DS-CNN | Depthwise Separable CNN (Phase 2) | — | — | — |
| C — Hybrid | SVM stage + DS-CNN (Phase 3) | — | — | — |

**Failure Analysis (Pillar 4):**

Three mathematically-explained failure modes were identified:

| Pair | EDA Overlap | Mathematical Root Cause |
|---|---|---|
| discomfort ↔ hungry | 1,074 | Translation sensitivity — onset offset inflates RBF distance in R^38656 |
| hungry ↔ tired | 976 | Temporal blindness — F₀ decay trajectory invisible to static kernel |
| cold_hot ↔ hungry | 899 | Silence amplification — StandardScaler inflates zero-padded frames (σ≈0 → 1/σ≫1) |

These limitations are architectural (not data-driven), directly justifying the
Depthwise Separable CNN in Phase 2.

#### Source Modules

| File | Functions | Purpose |
|---|---|---|
| `src/utils.py` | 7 functions | Dataset loading, class weights, label encoding, plotting, seed setting |
| `src/features.py` | 3 functions | Audio loading, spectrogram generation, batch dataset builder |
| `src/model_ml.py` | 4 functions | SVM training, evaluation, confusion matrix, failure case extraction |

---

## Reproduction Guide

### Requirements

```bash
pip install -r requirements.txt
```

### Run Order (on Kaggle)

All notebooks are designed to run sequentially in a single Kaggle session with
the preprocessed dataset mounted as input.

```
1. 03_feature_engineering.ipynb   → generates X.npy, y.npy, label_encoder.json
2. 04_baseline_ml.ipynb           → trains SVM, produces reports and ablation table
```

> Notebooks 01 and 02 (EDA and preprocessing) were run locally with the raw
> dataset. Their outputs (`clean_manifest.csv`, `preprocessing_params.json`,
> `augmentation_log.csv`, and the `audio_clean/` folder) are published at
> [kaggle.com/datasets/noxpie/baby-cry-preprocessed-dataset](https://www.kaggle.com/datasets/noxpie/baby-cry-preprocessed-dataset).

### Kaggle Path Configuration

Every notebook includes a Kaggle/local path switcher at the top. For Kaggle:

```python
KAGGLE_INPUT  = "/kaggle/input/datasets/noxpie/baby-cry-preprocessed-dataset"
DATASET_ROOT  = os.path.join(KAGGLE_INPUT, "BabyCryDataset_processed")
MANIFEST_PATH = os.path.join(DATASET_ROOT, "clean_manifest.csv")
AUDIO_BASE_DIR = os.path.join(DATASET_ROOT, "audio_clean")
OUTPUT_DIR    = "/kaggle/working"
```

### Random Seed

`random_state=42` is used everywhere. Call `set_random_seed(42)` from `src/utils.py`
at the top of any new notebook or script.

---

## Key Design Decisions (Viva-Ready)

| Question | Answer |
|---|---|
| Why 7.0 s target duration? | 95th percentile of duration distribution is 7.02 s — retains 94.8% of files intact, only 59 trimmed |
| Why resample to 22,050 Hz? | Neutral between the two source populations (8 kHz and 44.1 kHz); upsampling 8 kHz does not recover content above 4 kHz — acknowledged as dataset limitation |
| Why n_fft=2048, hop=512? | 93 ms window resolves 400 Hz fundamental; 23 ms stride = 75% overlap, standard for audio classification |
| Why Macro F1 as primary metric? | 15.9:1 imbalance makes accuracy misleading — a dummy classifier predicting hungry scores 35.3% |
| Why SVM with RBF kernel for Model A? | Maximum-margin classifier with non-linear boundary; consistent with Liu et al. (2019) baseline; `class_weight='balanced'` addresses imbalance |
| What does EDA reveal that prior work missed? | The sampling-rate artifact: scared is 81.8% at 44.1 kHz while hungry is 100% at 8 kHz — unreported in any prior publication on this corpus |

---

## References

| # | Citation |
|---|---|
| [1] | Liu et al. (2019). Infant Cry Language Analysis and Recognition. *IEEE/CAA JAS*. DOI: 10.1109/JAS.2019.1911435 |
| [2] | Abbaskhah et al. (2023). Infant cry classification by MFCC feature extraction with MLP and CNN structures. *Biomedical Signal Processing and Control*. DOI: 10.1016/j.bspc.2023.105261 |
| [3] | Hammoud et al. (2024). Machine learning-based infant crying interpretation. *Frontiers in Artificial Intelligence*. DOI: 10.3389/frai.2024.1337356 |

Full bibliography with all 11 references: [`docs/literature_review.md`](docs/literature_review.md)

---

## Phase 2 Preview (Due Apr 20–24, 2026)

- Depthwise Separable CNN operating on (128, 302) spectrogram images
- Grad-CAM interpretability to verify the CNN attends to correct frequency bands
- Audio augmentation strategies (pitch shifting, background noise injection)
- Regularisation: Dropout, Early Stopping
- TensorFlow Lite quantisation for edge deployment

---

*Project by: [Prashant Kumar - 230101] [Eranki Sai Vikas - 230121]| Advanced ML and Deep Learning | 2026*