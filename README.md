Author: Choi Joung Woo (2025011097)
Course: BRI519 - Brain Imaging Informatics (Fall 2025)
Institution: Korea University

## Project Overview
This project analyzes LFP signals recorded from the mouse auditory cortex during auditory tone stimulation experiments. The analysis includes:

- Preprocessing: Bandpass filtering, downsampling, and baseline correction
- Outlier Detection: MAD-based outlier trial rejection
- ERP Analysis: Event-Related Potential computation and peak analysis
- Frequency Analysis: Power Spectral Density and band power analysis


## Installation

### 1. Local Installation
```bash
git clone https://github.com/eprint523/BRI519-Final-Project-CJW.git
cd BRI519-Final-Project-CJW
# Create virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# install dependencies
pip install -r requirements.txt
```

### 2. Docker
See [Docker Setup](#docker-setup) section below.

## Usage

### Running the Analysis

```bash
python main.py
```

This will:
1. Load the LFP data from `data/mouseLFP.mat`
2. Preprocess the signals (filter, downsample, baseline correction)
3. Detect and remove outlier trials
4. Perform ERP analysis
5. Perform frequency band analysis
6. Save all figures to the `output/` directory

### Using Individual Modules

```python
from src.data_loader import load_lfp_data, separate_by_frequency
from src.preprocessing import preprocess_pipeline
from src.outlier_detection import find_outlier_indices, remove_outliers
from src.analysis import compute_erp, compute_band_power

# Load data
raw_sig, raw_freq = load_lfp_data('data/mouseLFP.mat')
sig_low, sig_high, freq_info = separate_by_frequency(raw_sig, raw_freq)

# Preprocess
processed_low = preprocess_pipeline(sig_low)
processed_high = preprocess_pipeline(sig_high)

# Remove outliers
outlier_idx, _ = find_outlier_indices(processed_low)
clean_low, _ = remove_outliers(processed_low, outlier_idx)

# Analyze
erp_results = compute_erp(clean_low)
```

## Docker Setup

### Building the Docker Image

```bash
docker build -t mouselfp-analysis .
```

### Running the Container

```bash
docker run -v $(pwd)/output:/app/output mouselfp-analysis
```

On Windows PowerShell:
```powershell
docker run -v ${PWD}/output:/app/output mouselfp-analysis
```

### Pulling from Docker Hub

```bash
docker pull eprint523/mouselfp-analysis:latest
docker run -v $(pwd)/output:/app/output eprint523/mouselfp-analysis:latest
```

## Analysis Methods

### 1. Preprocessing Pipeline
- **Bandpass Filter**: 0.1-500 Hz, 4th order Butterworth
- **Downsampling**: 10 kHz → 1 kHz
- **Baseline Correction**: Subtract mean of 0-100 ms pre-stimulus window

### 2. Outlier Detection
- **Method**: Median Absolute Deviation (MAD)
- **Criterion**: Trials with variance > median ± 3×MAD are rejected
- **Rationale**: MAD is robust to outliers unlike standard deviation

### 3. ERP Analysis
- Compute trial-averaged response (mean ± SEM)
- Identify N1 peak (negative deflection at ~110-130 ms)
- Compare amplitude and latency between tone conditions

### 4. Frequency Band Analysis
- **Bands analyzed**: Delta, Theta, Alpha, Beta, Low Gamma, High Gamma
- **Method**: Welch's periodogram
- **Window**: 100-300 ms post-stimulus
- **Statistics**: Independent t-test between conditions


## Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

## License

This project is for educational purposes as part of the BRI519 course at Korea University.

## Acknowledgments

- Course Instructor: Prof. Jonghwan Lee
- Teaching Assistant: neti2207@korea.ac.kr
