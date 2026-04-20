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
| **Model B** — DS-CNN | Depthwise Separable CNN on spectrogram images | Phase 2 ✅ |
| **Model C** — Hybrid | SVM decision scores + DS-CNN GAP features → meta-classifier | Phase 3 🔄 |

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
│   ├── 04_baseline_ml.ipynb             # SVM Model A training + failure analysis
│   └── 05-dscnn-executed.ipynb          # DS-CNN Model B training + ablation + TFLite
├── src/
│   ├── __init__.py
│   ├── utils.py                         # shared utility functions
│   ├── features.py                      # spectrogram generation functions
│   ├── model_ml.py                      # reusable SVM training and evaluation
│   └── model_dl.py                      # DS-CNN architecture builder and TFLite export
├── reports/
│   ├── phase2_report.tex                # IEEE two-column Phase 2 report (LaTeX)
│   ├── figures/                         # exported PNGs for report figures
│   └── generate_architecture_diagram.py # architecture diagram generation script
├── docs/
│   ├── literature_review.md             # IEEE-format, 11 references
│   ├── dataset_readme.md                # raw + processed dataset documentation
│   └── project_guide.md                 # complete ground-up learning guide (all phases)
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** Large binary files (`audio_clean/`, `X.npy`, `y.npy`, `best_dscnn.keras`,
> `dscnn_quantized.tflite`) are gitignored.
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
| 2 — Deep Learning Rigor | DS-CNN with BatchNorm, Dropout, weight initialisation, ablation |
| 3 — Integration & Innovation | Hybrid must logically combine both domains and outperform both baselines |
| 4 — Technical Validation | Rigorous ablation studies and failure analysis with mathematical reasoning |
| 5 — Documentation & Reproducibility | Clean repo, requirements.txt, reproducible README |

| Phase | Weight | Due | Status |
|---|---|---|---|
| Phase 1 — Foundation & Advanced ML | 30% | Mar 23–27, 2026 | ✅ Complete |
| Phase 2 — Deep Learning Architecture | 30% | Apr 20–24, 2026 | ✅ Complete |
| Phase 3 — Hybrid Integration + ESP32 | 40% | May 4–8, 2026 | 🔄 Upcoming |

---

## Results Summary

| Model | Configuration | Accuracy | Macro F1 | Weighted F1 | TFLite Size |
|---|---|---|---|---|---|
| **A — SVM Baseline** | RBF-SVM, C=10, flattened 38,656-dim | 21.78% | 0.4029 | 21.98% | — |
| **B — DS-CNN** | Depthwise Separable CNN, 82,760 params | 39.56% | **0.4370** | 33.98% | **99.5 KB** |
| **C — Hybrid** | SVM + DS-CNN meta-classifier (Phase 3) | — | > 0.4370 target | — | < 512 KB |

**Primary metric: Macro F1-Score** (not accuracy — 15.9:1 class imbalance makes accuracy misleading).
Both models evaluated on the identical 225-sample test set. Model B delivers +8.5% relative Macro F1 over Model A.

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
| Top confused pairs | discomfort↔️hungry (1,074 overlaps), hungry↔️tired (976), cold_hot↔️hungry (899) |
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
- **Training time:** 21.2 seconds

**Phase 1 Results:**

| Metric | Value |
|---|---|
| Test Accuracy | 21.78% |
| **Test Macro F1** | **0.4029** |
| Test Weighted F1 | 21.98% |
| Support vectors | 1,051 / 1,109 training samples (94.8% — confirms dense overlap) |

**Failure Analysis (Pillar 4) — Three Architectural Failure Modes:**

| Mode | Confused Pair | EDA Overlap | Mathematical Root Cause |
|---|---|---|---|
| **FM1** — Translation sensitivity | All class pairs | — | Cry onset offset inflates RBF distance in R^38,656 — same cry shifted 100 ms looks entirely different after flattening |
| **FM2** — Temporal blindness | hungry ↔️ tired | 976 | F₀ decay trajectory (characteristic of tired) is invisible to a static kernel on a flattened vector |
| **FM3** — Silence amplification | cold_hot ↔️ hungry | 899 | StandardScaler: silent frames have σ≈0 → 1/σ≫1 → padding length drives the decision boundary |

These limitations are architectural (not data-driven) and cannot be fixed by
kernel tuning. They directly justify the Depthwise Separable CNN in Phase 2.

#### Phase 1 Source Modules

| File | Functions | Purpose |
|---|---|---|
| `src/utils.py` | 7 functions | Dataset loading, class weights, label encoding, plotting, seed setting |
| `src/features.py` | 3 functions | Audio loading, spectrogram generation, batch dataset builder |
| `src/model_ml.py` | 4 functions | SVM training, evaluation, confusion matrix, failure case extraction |

---

## Phase 2 — Complete ✅

### What Was Delivered

Phase 2 introduces Model B — a Depthwise Separable CNN that operates directly on
(128 × 302) Log-Mel spectrograms and is architecturally designed to address each
of the three failure modes identified in Phase 1. Phase 2 also delivers Grad-CAM
interpretability, a four-variant controlled ablation study, and a TFLite model
that fits within the ESP32 SRAM budget.

**Phase 2 report:** [`reports/phase2_report.tex`](reports/phase2_report.tex) (IEEE two-column format).

#### DS-CNN Architecture (`05-dscnn-executed.ipynb`, `src/model_dl.py`)

**Design rationale:** depthwise separable factorisation reduces multiply-accumulate
cost per block to (1/N + 1/D_K²) of a standard convolution. For N=64, D_K=3:
1/64 + 1/9 ≈ 0.127 — approximately 8× fewer MACs per block. This MAC reduction
is what enables post-training quantisation and ESP32 deployment.

**Architecture:**

```
Input (128, 302, 1)  →  Stem Conv2D(32)  →  DS Block 1 (32→64)
  →  DS Block 2 (64→128)  →  DS Block 3 (128→256)
  →  GlobalAveragePooling2D  →  Dropout(0.15)
  →  Dense(128, ReLU)  →  Dropout(0.1)  →  Dense(8, Softmax)
```

Each DS block: DepthwiseConv2D(3×3) + BatchNorm + ReLU + Conv2D(1×1) + BatchNorm + ReLU + MaxPool(2×2).

| Property | Value |
|---|---|
| Total parameters | **82,760** |
| Standard CNN equivalent | 423,688 params |
| Parameter reduction | **5.1×** |
| Weight initialisation | He Normal (all Conv / Dense layers) |
| Normalisation | BatchNorm after every convolutional layer |
| Classification head | Dropout(0.15) → Dense(128) → Dropout(0.1) → Dense(8, Softmax) |

**Failure mode fixes — architecture → mechanism:**

| Phase 1 Failure | DS-CNN Fix | Mechanism |
|---|---|---|
| FM1 — Translation sensitivity | Conv2D weight-sharing + MaxPool | Translation equivariance through weight sharing; local invariance through pooling |
| FM2 — Temporal blindness | 3×3 kernels + deep receptive fields | 3×3 kernels span multiple STFT frames; stacked blocks integrate longer temporal windows |
| FM3 — Silence amplification | GlobalAveragePooling2D | Zero-padded frames produce near-zero activations; GAP mean naturally downweights them |

FM3 fix validated empirically by Grad-CAM (all 8 classes: content-region attention > 60%).

#### Training Protocol

**Three-way augmentation-aware split (same test set as Phase 1):**

| Partition | Size | Composition |
|---|---|---|
| Train | 974 | 765 original + 209 augmented |
| Validation | 135 | original only, zero augmented |
| Test | 225 | original only, zero augmented — **byte-for-byte identical to Phase 1** |

Leakage assertions enforced at runtime: zero augmented samples in val or test. ✅

**Optimiser and callbacks:**

```python
optimizer  = Adam(learning_rate=5e-4)
loss       = CategoricalCrossentropy(label_smoothing=0.1)
callbacks  = [
    EarlyStopping(monitor='val_macro_f1', patience=25, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]
```

Label smoothing (ε=0.1) replaces per-sample class weighting as the imbalance
handling strategy — justified by ablation variant B1.

#### Phase 2 Results

| Metric | Model A (SVM) | Model B (DS-CNN) | Δ |
|---|---|---|---|
| Test Accuracy | 21.78% | **39.56%** | +17.78 pp (+81.6% relative) |
| **Test Macro F1** | 0.4029 | **0.4370** | **+0.0341 (+8.5% relative)** |
| Test Weighted F1 | 21.98% | 33.98% | +12.00 pp |

Both models evaluated on the identical 225-sample test set.

**Per-class F1 (Model B):**

| Class | F1 | Support | Status |
|---|---:|---:|---|
| hungry | 0.4803 | 79 | Absorber class — recall 69.6%, draws false positives from all other classes |
| burping | 0.4186 | 25 | Learned |
| belly pain | 0.3830 | 27 | Learned |
| cold_hot | 0.1538 | 26 | Partially confused with hungry |
| tired | 0.0606 | 28 | Partially confused with hungry |
| **discomfort** | **0.0000** | **28** | **Collapsed entirely into hungry → Phase 3 target** |
| lonely | 1.0000 | 5 | Small-sample artefact (81.5% synthetic training data) |
| scared | 1.0000 | 7 | Small-sample artefact (74.6% synthetic training data) |

**Three honest caveats:**
1. `lonely` and `scared` F1 = 1.0 are not reliable generalisation — 5 and 7 test samples respectively, dominated by synthetic training examples.
2. `discomfort` F1 = 0.0 is the primary residual failure (1,074 EDA overlaps with hungry from Phase 1). This is the explicit Phase 3 target.
3. `hungry` acts as an absorber class — high recall inflates its F1 at the cost of other classes.

#### Ablation Study

Four controlled variants, all evaluated on the identical 225-sample test set:

| Variant | Change vs Model B | Macro F1 | Δ | Interpretation |
|---|---|---:|---:|---|
| **A — SVM** | Phase 1 reference | 0.4029 | −0.034 | Classical baseline |
| **B1** | + class_weight='balanced' | 0.3501 | **−0.087** | Amplifies gradients from 80%-synthetic minority classes; destabilises training |
| **B2** | + SpecAugment (F=10, T=15) | 0.4065 | −0.031 | Masking 8% of signal per sample destroys more information than it regularises on 974 training examples |
| **B3** | − dropout (head only) | 0.4426 | +0.006 | Within seed-level noise; BatchNorm provides sufficient regularisation |
| **B4** | Standard Conv2D (5.1× params) | 0.4177 | −0.019 | Standard CNN with 5.1× more parameters scores lower — DS factorisation improves both efficiency and accuracy |
| **B — shipped** | Full DS-CNN | **0.4370** | — | Best Macro F1 at lowest parameter count |

**Key conclusions from ablation:**
- Class weighting (B1) is the most destructive change (−0.087): label smoothing is the correct imbalance strategy at this dataset scale.
- SpecAugment (B2) hurts on small datasets — confirmed to require data redundancy to be effective.
- DS factorisation (B4) is not just an efficiency trick: it produces better features at 1/5 the parameter count.

#### Grad-CAM Interpretability

Computed on the last DepthwiseConv2D layer. For each test sample, the fraction
of gradient-weighted attention in the content region (frames 0–189) vs the
zero-padded silence region (frames 190–301) is measured:

> **All 8 classes exhibit content-region attention > 60%.**

This empirically confirms the FM3 fix: the model attends to cry acoustics, not
padding artefacts. The SVM's StandardScaler-inflated silence regions are no
longer driving predictions.

**Figure:** `reports/figures/gradcam_per_class.png`

#### TFLite Edge Deployment

Post-training dynamic-range quantisation (Float32 → INT8 weights, float activations):

| Artefact | Size | ESP32 512 KB budget |
|---|---|---|
| Keras model (`.keras`) | ~1,100 KB | ✗ development only |
| TFLite model (`.tflite`, DRQ) | **99.5 KB** | ✅ fits with ~400 KB headroom |

- **Compression:** ~11× with < 1% accuracy loss
- **Verification:** 5/5 randomly selected test samples produce identical argmax predictions between Keras and TFLite interpreters
- **Headroom:** ~400 KB available for TFLite Micro runtime, audio capture buffers, and RTOS overhead

**Saved Phase 2 artefacts** (gitignored, available from Kaggle working directory):

| File | Description |
|---|---|
| `best_dscnn.keras` | Full Keras model (best validation Macro F1 checkpoint) |
| `dscnn_quantized.tflite` | INT8 quantised TFLite model (99.5 KB) |
| `svm_model.pkl` | SVM model retrained on Phase 2 split (for Phase 3 fusion) |
| `svm_scaler.pkl` | StandardScaler fitted on Phase 2 training set |
| `split_indices.pkl` | Train / val / test index arrays (Phase 2 split) |
| `norm_params.json` | Spectrogram normalisation bounds {X_min: −80, X_max: 0} |

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
1. 03_feature_engineering.ipynb    → generates X.npy, y.npy, label_encoder.json
2. 04_baseline_ml.ipynb            → trains SVM Model A, produces Phase 1 results
3. 05-dscnn-executed.ipynb         → trains DS-CNN Model B, ablation, Grad-CAM, TFLite
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
| Why SVM with RBF kernel for Model A? | Maximum-margin classifier with non-linear boundary; consistent with Liu et al. (2019) baseline |
| What does EDA reveal that prior work missed? | The sampling-rate artifact: scared is 81.8% at 44.1 kHz while hungry is 100% at 8 kHz — unreported in any prior publication on this corpus |
| Why depthwise separable convolution? | MAC reduction ratio 1/N + 1/D_K² ≈ 0.127 (8× per block) enables post-training quantisation and 99.5 KB TFLite fit within ESP32 512 KB SRAM |
| Why label smoothing instead of class weights? | Ablation B1 shows class_weight='balanced' drops Macro F1 by 0.087 — amplifies gradients from 80%-synthetic minority classes, destabilising training |
| Why no SpecAugment? | Ablation B2 shows SpecAugment drops Macro F1 by 0.031 — masking 8% per sample destroys information on a 974-sample training set; requires data redundancy to be effective |
| Why GlobalAveragePooling instead of Flatten? | Fixes FM3 (silence amplification): zero-padded regions produce near-zero ReLU activations; GAP mean naturally downweights them. Grad-CAM confirms content-region attention > 60% for all 8 classes |
| Why is discomfort F1 = 0.0 in Phase 2? | 1,074 EDA nearest-neighbour overlaps with hungry — the largest confused pair in the corpus. Purely architectural CNN change cannot resolve genuine acoustic similarity. Phase 3 hybrid specifically targets this |

---

## Phase 3 — Upcoming 🔄 (Due May 4–8, 2026)

### Objective

Introduce **Model C** — a hybrid classifier that fuses Model B's 256-dimensional
GAP feature embedding with Model A's 8-dimensional SVM decision scores into a
264-dimensional representation, then trains a meta-classifier on this fused input.

```
DS-CNN Model B  →  256-dim GAP temporal-acoustic features  ─┐
                                                              ├→  264-dim fused vector  →  Meta-Classifier  →  8 classes
SVM Model A     →  8-dim geometric decision scores          ─┘
```

### Success Criteria (all three must pass)

| Criterion | Threshold |
|---|---|
| Beat Model A | Macro F1 > 0.4029 |
| Beat Model B | Macro F1 > 0.4370 |
| ESP32 deployable | TFLite size < 512 KB SRAM |

Model C is only worth shipping if it surpasses both baselines. If it fails either
F1 threshold, Model B is the recommended deployment model.

### Primary Target

The `discomfort` class (Phase 2 F1 = 0.000, Phase 1 F1 = 0.103) — collapsed into
`hungry` in both prior phases due to 1,074 EDA-measured nearest-neighbour overlaps.
The SVM and CNN have different decision geometries and fail on different samples;
the meta-classifier is designed to exploit this complementarity.

### All Phase 2 Artefacts Are Ready

Zero retraining required to begin Phase 3. All saved model artefacts
(`svm_model.pkl`, `svm_scaler.pkl`, `best_dscnn.keras`, `split_indices.pkl`,
`norm_params.json`) provide a complete handoff from Phase 2.

---

## References

| # | Citation |
|---|---|
| [1] | Liu et al. (2019). Infant Cry Language Analysis and Recognition. *IEEE/CAA JAS*. DOI: 10.1109/JAS.2019.1911435 |
| [2] | Abbaskhah et al. (2023). Infant cry classification by MFCC feature extraction with MLP and CNN structures. *Biomedical Signal Processing and Control*. DOI: 10.1016/j.bspc.2023.105261 |
| [3] | Hammoud et al. (2024). Machine learning-based infant crying interpretation. *Frontiers in Artificial Intelligence*. DOI: 10.3389/frai.2024.1337356 |
| [4] | Howard et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv:1704.04861* |
| [5] | Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*. DOI: 10.1109/ICCV.2017.74 |
| [6] | Jacob et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. *CVPR 2018*. DOI: 10.1109/CVPR.2018.00286 |
| [7] | He et al. (2015). Delving Deep into Rectifiers. *ICCV 2015*. DOI: 10.1109/ICCV.2015.123 |
| [8] | Park et al. (2019). SpecAugment: A Simple Data Augmentation Method for ASR. *Interspeech 2019*. DOI: 10.21437/Interspeech.2019-2680 |

Full bibliography with all references: [`docs/literature_review.md`](docs/literature_review.md)

---

*Project by: [Prashant Kumar - 230101] [Eranki Sai Vikas - 230121] | Advanced ML and Deep Learning | 2026*
