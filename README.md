# Human Activity Recognition (HAR) Project

This repository contains the source code for our Human Activity Recognition project using the MotionSense smartphone sensor dataset. We implement and compare three models:

* **Random Forest (RF)** on handcrafted time- and frequency-domain features
* **Multilayer Perceptron (MLP)** on the same handcrafted feature set
* **Long Short-Term Memory (LSTM)** network on raw windowed time-series


## 🚀 Quick Start

### 1. Clone & Install

```
git clone https://github.com/AnesuMugiya/MotionSenseProject.git
cd MotionSenseProject
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download & Organize Data

Place the MotionSense CSV folders under `data/A_DeviceMotion_data/` and ensure `data/data_subjects_info.csv` is present:

```
data/
├── data_subjects_info.csv
└── A_DeviceMotion_data/
    ├── dws_1/
    │   └── sub_1.csv
    ├── wlk_7/
    │   └── sub_1.csv
    └── …
```

### 3. Preprocess & Feature Extraction

Run the preprocessing notebook or script:

```
jupyter nbconvert --to notebook --execute notebooks/preprocess.ipynb
```

This generates:

* Windowed raw sequences for LSTM
* Handcrafted feature CSV for RF & MLP

### 4. Train Models

#### Random Forest

```
jupyter nbconvert --to notebook --execute notebooks/train_rf.ipynb
```

#### MLP

```
jupyter nbconvert --to notebook --execute notebooks/train_mlp.ipynb
```

#### LSTM

```
jupyter nbconvert --to notebook --execute notebooks/train_lstm.ipynb
```

### 5. Evaluate & Visualize

Generate confusion matrices and performance tables:

```
jupyter nbconvert --to notebook --execute notebooks/train_lstm.ipynb
```

Or run the evaluation script:

```
python src/evaluation.py --preds outputs/lstm_preds.npy --labels outputs/lstm_labels.npy --out images/LSTMoutput.png
```

---

## 📦 Dependencies

Key libraries in `requirements.txt`:

* Python 3.8+
* NumPy, Pandas
* scikit-learn
* PyTorch
* Matplotlib, Seaborn
* torchinfo (optional)
* Jupyter

Install with:

```
pip install -r requirements.txt
```

---

## 🔍 Design Decisions

* **Windowing (2 s / 50 % overlap):** captures a full gait cycle and boosts sample count
* **Feature Engineering (RF & MLP):** 18 statistical features (mean, std, skew, kurtosis) ± FFT for domain insights
* **Raw Sequence (LSTM):** end-to-end temporal learning
* **Normalization:** StandardScaler for handcrafted features; MinMaxScaler for raw channels
* **User-Wise Split:** 80 % train/val, 20 % test by subject to ensure generalization to new users

---

## 📄 Citation

If you use this code, please cite:

> Anesu Mugiya and Boikanyo Ntsimane, “Human Activity Recognition Using Smartphone Sensor Data,” EEE4114F Final Project Report, May 2025.

