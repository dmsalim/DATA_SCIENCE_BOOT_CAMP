# Data Science Boot Camp — Final Project

Short-term weather radar nowcasting with the [SEVIR](https://registry.opendata.aws/sevir/) dataset, using **VIL** (Vertically Integrated Liquid) radar images.

Each SEVIR VIL event is a 384 × 384 × 49 sequence: 49 frames, 5 minutes apart. Each event is cut into windows of **13 input frames** (about the past hour) and **12 target frames** (the next hour). Two approaches are compared:

1. **Tabular models (scikit-learn):** a few summary features per window, used to predict the intensity of the 60-minutes-ahead frame (regression) and to classify it as normal, moderate or severe.
2. **Deep learning models (PyTorch):** all 12 future frames are forecast pixel by pixel with a CNN, a space-time Transformer, and a residual diffusion model built on the frozen Transformer. Each is scored with the Critical Success Index (CSI) against a persistence baseline.

> **Note:** The deep learning training script (`train_ml_models.py`) is not in this repository. The code here covers data processing, the tabular models, the diagnostic plots and anomaly detection.

## Repository layout

```
FINAL_PROJECT/
├── sevir_utils.py                  # HDF5 event loader, windowing, PyTorch Dataset
├── process_data.py                 # Explore catalog, plot a sample event, date-based train/test split
├── extract_tabular_features.py     # Windows -> tabular features + targets (CSV)
├── train_tabular_models.py         # Regression + severity classification (sklearn)
├── analysis_plots.py               # Diagnostic plots for the tabular models
├── classic_anomaly_detection.py    # Isolation Forest / One-Class SVM / LOF on tabular features
├── CATALOG.csv                     # SEVIR event catalog (all image types)
├── vil_train_catalog.csv           # VIL events before 2019-06-01 (15,693 events)
├── vil_test_catalog.csv            # VIL events from 2019-06-01 on (4,700 events)
├── vil_tabular_features_train.csv  # 47,079 windows
├── vil_tabular_features_test.csv   # 14,100 windows
├── ANALYSIS_PLOTS/                 # Regression, confusion matrix, feature importance, calibration plots
└── TESTS/                          # Sample VIL event: frame grid, per-frame PDFs, and movie
```

## Data not included

These files are too large for GitHub (over 100 MB each), so they are not in this repo:

- **Raw SEVIR VIL data** (`FINAL_PROJECT/data/vil/{2017,2018,2019}/*.h5`, about 137 GB in total)
- **`trained_tabular_models.pickle`** (about 700 MB), which `train_tabular_models.py` regenerates

To get the raw data, download it from the public SEVIR S3 bucket (no AWS account needed):

```bash
cd FINAL_PROJECT
aws s3 sync --no-sign-request s3://sevir/data/vil data/vil
```

The `file_name` column of the catalogs gives each file's path relative to `FINAL_PROJECT/data/`.

## Pipeline

Run everything from inside `FINAL_PROJECT/`:

```bash
cd FINAL_PROJECT

# 1. Explore the data, make sample plots in TESTS/, and write the train/test catalogs
python process_data.py

# 2. Extract tabular features (13-frame input, 12-frame target, stride 12)
python extract_tabular_features.py

# 3. Train the tabular regression/classification models -> trained_tabular_models.pickle
python train_tabular_models.py

# 4. Generate the diagnostic plots in ANALYSIS_PLOTS/
python analysis_plots.py

# 5. Run anomaly detection on the tabular features
python classic_anomaly_detection.py
```

The deep nowcasting models (CNN, Transformer, diffusion) were trained with a separate script that is not included here.

### Tabular features

From the last input frame: mean, max and standard deviation of intensity, and rain area (pixels > 0). The one-frame change in mean and max intensity. Mean and max over the whole 13-frame input window.

The targets come from the 60-minutes-ahead frame:
- `target_mean`: its mean VIL
- `target_exceeds`: whether any pixel is above VIL 74
- `target_severity`: computed at training time from the train-set quantiles of `target_mean`. At or above the 90th percentile is moderate; at or above the 99th is severe.

### Models

| Task | Models |
|---|---|
| Regression (`target_mean`) | Linear, Lasso, Random Forest, Gradient Boosting |
| Classification (`target_severity`) | Logistic Regression, Random Forest, Gradient Boosting |
| Frame forecasting (12 frames, training code not included) | Persistence baseline, SimpleConvBaseline (CNN), SpaceTimeTransformer, residual Gaussian diffusion |

The forecasting models were scored with CSI at VIL thresholds 16, 74, 133, 160, 181 and 219.

## Requirements

Python 3.11 with `numpy`, `pandas`, `matplotlib`, `h5py`, `scikit-learn` and `torch`:

```bash
pip install numpy pandas matplotlib h5py scikit-learn torch
```

## Data citation

Veillette, M., Samsi, S., & Mattioli, C. (2020). *SEVIR: A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology.* NeurIPS 2020.
