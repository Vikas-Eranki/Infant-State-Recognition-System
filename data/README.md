# Baby Cry Sense Dataset

## Description
The **Baby Cry Sense Dataset** is designed to support research in automated recognition of infant cries using machine learning and deep learning techniques.  

This dataset enables researchers and developers to build intelligent systems capable of analyzing and interpreting acoustic signals corresponding to infants’ emotional and physical needs.

---

## Dataset Contents
The dataset contains **512 audio samples**, categorized into **8 distinct classes**:

| Class        | Number of Files |
|-------------|----------------|
| belly_Pain  | 16             |
| burping     | 18             |
| Cold/Hot    | 7              |
| discomfort  | 30             |
| hungry      | 382            |
| lonely      | 11             |
| scared      | 20             |
| tired       | 28             |

---

## Scientific Significance
- Enables modeling of distinctive cry patterns corresponding to different infant needs.
- Supports development of smart infant monitoring systems.
- Contributes to advancements in **Human-Computer Interaction (HCI)** for infant care.

---

## Research & Development Opportunities
- Feature extraction techniques:
  - Mel-Frequency Cepstral Coefficients (MFCCs)
  - Spectrogram analysis
- Model development using:
  - Convolutional Neural Networks (CNN)
  - Long Short-Term Memory (LSTM)
  - Transformer-based architectures
- Fine-grained analysis of cry patterns for improved classification and caregiver assistance.

---

## Dataset Sources
This dataset is compiled from multiple sources:

- https://github.com/skytells-research/DeepInfant.git  
- https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus  
- https://www.kaggle.com/datasets/bhoomikavalani/donateacrycorpusfeaturesdataset  
- https://github.com/hanagohar/baby_cry_classification  
- https://www.kaggle.com/datasets/shahdahmedfoud/baby-sound  

---

## ⚠️ Notes
- Class distribution is **imbalanced** (majority class: *hungry*).
- Preprocessing and augmentation may be required for optimal model performance.



## Dataset Setup
1. Download the Baby Cry Sense Dataset (1126 samples) from:
   https://www.kaggle.com/datasets/mennaahmed23/baby-cry-dataset/data
2. Extract and place the audio files in `data/raw/` following this structure:

data/raw/
├── belly_pain/
├── burping/
├── cold_hot/
├── discomfort/
├── hungry/
├── lonely/
├── scared/
└── tired/