
# 📡 WiFi-Pose-AI: Human Pose Estimation Using WiFi Signals

This project explores the frontier of **non-visual sensing** by using WiFi CSI (Channel State Information) to detect and estimate human body poses through walls, leveraging deep learning. By converting radio frequency signal variations into joint coordinate estimations, this system enables **vision without cameras**, offering a privacy-preserving alternative to traditional surveillance or pose tracking methods.

---

## 📘 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Limitations & Future Work](#limitations--future-work)
- [Ethical Considerations](#ethical-considerations)
- [Citations & References](#citations--references)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## 🚀 Overview

Human pose estimation has traditionally relied on optical systems like cameras, LiDAR, or infrared sensors. However, these methods can be intrusive and raise privacy concerns. WiFi-Pose-AI aims to:
- Utilize commodity WiFi hardware to sense human movement.
- Extract and process CSI data reflecting off the human body.
- Apply a neural network to estimate 2D body joint locations.
- Visualize these poses in real time, with potential extension to 3D.

---

## 🧠 How It Works

### System Pipeline:
```
WiFi Transmitter → CSI Collection Tool → Preprocessing → Deep Learning Model → Pose Keypoints
```

1. **CSI Extraction**: Using Intel 5300 NIC or Nexmon on a Raspberry Pi to extract signal variations.
2. **Preprocessing**: Apply bandpass filtering, denoising, normalization.
3. **Model Inference**: Deep CNN maps processed CSI to joint coordinates.
4. **Pose Visualization**: Output is rendered as 2D pose keypoints or stick-figure skeletons.

---

## 💻 Technologies Used

- **Python 3.8+**
- **PyTorch** – Model development
- **SciPy / NumPy** – Signal processing
- **Matplotlib / OpenCV** – Visualization
- **Intel 5300 NIC / Nexmon CSI Tool** – CSI extraction
- **OpenPose** – Ground-truth annotation (optional)

---

## 🔍 Applications

- 🏥 Fall detection for elderly people in smart homes
- 🔦 Vision through walls during emergency rescues
- 📡 Gesture control for home automation or VR
- 👁️ Privacy-conscious indoor monitoring

---

## 🧾 Project Structure

```
wifi-pose-ai/
├── data/
│   ├── raw_csi/               # Raw input CSI files (.npy, .csv, or .mat)
│   └── processed/             # Filtered & normalized files for training
├── notebooks/                 # Exploratory Jupyter notebooks
├── src/
│   ├── data_loader.py         # CSI file loading utilities
│   ├── preprocess.py          # Filtering and normalization
│   ├── model.py               # Neural network definition
│   ├── train.py               # Model training script (coming soon)
│   └── evaluate.py            # Evaluation functions (coming soon)
├── configs/                   # YAML-based hyperparameter configs
├── outputs/
│   ├── model_checkpoints/     # Saved model weights
│   └── plots/                 # Training and pose output plots
├── requirements.txt           # Dependency list
└── README.md                  # Project overview (this file)
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/wifi-pose-ai.git
cd wifi-pose-ai
pip install -r requirements.txt
```

> ⚠️ Note: You will need CSI data collected from a compatible NIC or simulator.

---

## 🧪 Usage

### 1. Load CSI Data
```python
from src.data_loader import load_dataset
X, y = load_dataset("data/raw_csi/")
```

### 2. Preprocess
```python
from src.preprocess import bandpass_filter, normalize
X_filtered = bandpass_filter(X)
X_norm = normalize(X_filtered)
```

### 3. Run Inference
```python
from src.model import CSIPoseNet
model = CSIPoseNet()
output = model(torch.tensor(X_norm).float())
```

---

## 🧠 Model Training

The `train.py` script (coming soon) will support:
- Cross-validation
- Batch training with GPU
- Model checkpoint saving
- TensorBoard support (optional)

---

## 📊 Evaluation

Metrics planned:
- Mean Squared Error (MSE) on joint locations
- Average keypoint detection accuracy
- Visualization-based qualitative analysis

---

## 🖼️ Visualizations

Results will be plotted as stick figures:
- 2D keypoints rendered using Matplotlib
- Real-time overlay (future)

---

## 🚧 Limitations & Future Work

- 🔁 Dataset augmentation tools in development
- 🔀 Transformer and RNN model versions under testing
- 🔊 Real-time streaming inference to be integrated
- 🔐 Federated training pipeline for multi-home use case

---

## 🔐 Ethical Considerations

WiFi-based sensing raises serious questions around surveillance, so we emphasize:
- Consent-based usage
- Indoor, privacy-preserving applications
- No raw imagery is collected
- Medical and rescue use prioritized

---

## 📚 Citations & References

- RF-Pose: [MIT CSAIL](http://rfpose.csail.mit.edu/)
- Dhalperi CSI Tool: https://dhalperi.github.io/linux-80211n-csitool/
- OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose

---

## 🙏 Acknowledgements

Thanks to:
- MIT’s RF-sensing community
- Contributors to CSI toolkits
- Open-source deep learning community

---

## 📩 Contact

Author: RAJESH KUMAR JOGI  
Email: rajeshkumarjogi.2098@gmail.com

---
