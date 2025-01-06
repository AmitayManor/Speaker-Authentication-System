# Speaker Authentication System

A speaker authentication system using LPCC (Linear Predictive Cepstral Coefficients) features and Gaussian Naive Bayes classification. This project implements a lightweight, efficient speaker verification system suitable for mobile devices.

## Overview

The system performs speaker authentication through audio analysis using:
- LPCC feature extraction for voice characteristics
- Gaussian Naive Bayes classification for speaker verification
- Linear-time processing for efficient mobile deployment

## Features

- Text-independent speaker verification
- Pre-processing pipeline including silence removal and audio normalization
- LPCC-based feature extraction (20 coefficients)
- Gaussian Naive Bayes classification with configurable priors
- ~99.7% accuracy rate with near-zero false positive rate
- Linear time complexity for mobile efficiency

## Requirements

- Python 3.x
- Libraries:
  - librosa
  - spafe
  - numpy
  - scipy
  - scikit-learn
  - matplotlib

## Project Structure

```
.
├── app_logic.py       # Main application logic and configuration
├── db_process.py      # Database processing and model training
├── features_extraction.py  # LPCC feature extraction
├── file_ops.py        # File operations utilities
├── pre_process.py     # Audio pre-processing functions
└── main.py           # Entry point
```

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install librosa spafe numpy scipy scikit-learn matplotlib
```

## Usage

The system operates in three main phases:

1. **Training**:
```python
from app_logic import run_final_db_computation
run_final_db_computation()
```

2. **Threshold Computation**:
```python
from app_logic import run_threshold_computation
run_threshold_computation()
```

3. **User Authentication**:
```python
from app_logic import run_user_computation
run_user_computation()
```

## Configuration

Key parameters can be adjusted in `app_logic.py`:
- `FINAL_THRESHOLD`: Authentication confidence threshold (default: 0.99)
- `FINAL_PRIOR`: Prior probabilities for GNB classifier [imposter, authorized] (default: [0.2, 0.8])

## Performance

- Accuracy: 99.7%
- False Positive Rate: ~0%
- Processing Time: Linear complexity O(n)
- Suitable for mobile devices with limited processing power

## License

This project is inspired by the paper "A Text-Independent Speaker Authentication System for Mobile Devices" by Florentin Thullier, Bruno Bouchard & Bob-Antoine J. Menelas.
