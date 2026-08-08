# 🩺 PulmoScan PACS — AI-Powered Chest X-Ray Analysis & PACS Decision Support System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced, end-to-end Deep Learning clinical decision support system (CDSS) built to **classify multi-class chest radiographs**, **localize radiological pathologies** using Grad-CAM heatmaps, and provide PACS-style diagnostic image controls. Designed with a Streamlit interface, this application streamlines radiologist workflows and generates printable, structured consultation PDF reports.

---

## 📸 Key Capabilities & System Features

### 1. Multi-Class Pathological Classification
* Powered by a custom **Convolutional Neural Network (CNN)** trained on chest radiograph data.
* Automatically categorizes images into **4 distinct clinical diagnoses**:
  * 🦠 **COVID-19**
  * 🫁 **Lung Opacity**
  * 🧫 **Viral Pneumonia**
  * ✅ **Normal**
* Provides confidence percentages alongside full softmax probability distribution outputs.

### 2. Grad-CAM Feature Localization & Explainable AI (XAI)
* Integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to highlight region-of-interest (ROI) visual features driving model prediction.
* Projects attention heatmaps onto the raw input radiograph to assist in identifying focal infiltrates and consolidations.

### 3. PACS-Style Diagnostic Image Windowing
* Interactive visual manipulation tools mimicking **Picture Archiving and Communication System (PACS)** workstations:
  * **Contrast & Brightness Adjustment:** Dynamic contrast stretching for improved soft tissue clarity.
  * **Grayscale Inversion:** Toggles between standard radiograph view and inverted density modes.
  * **Zoom & Region Pan:** Inspect fine pulmonary patterns and apical regions easily.

### 4. Built-in Clinical Knowledge Base
* Instant lookup for expected radiological findings, standard differential diagnoses, and diagnostic criteria for each detected class.
* Operates **100% locally** without external API dependencies or cloud lookups.

### 5. Automated PDF Consultation Reports
* Generates downloadable, multi-page structured medical reports containing:
  * Patient accession metadata and timestamps.
  * High-resolution original radiograph and Grad-CAM overlay snapshots.
  * Predicted diagnostic class probabilities and medical summary.

---

## 🛠️ System Architecture & Workflow

```text
[ Raw Chest Radiograph ]
          │
          ▼
┌──────────────────────────────────┐
│   PACS Preprocessing Engine      │ ◄── (Windowing, Resizing, Normalization)
└──────────────────────────────────┘
          │
          ├────────────────────────────────────────┐
          ▼                                        ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│    CNN Classifier Model          │    │    Grad-CAM Heatmap Generator    │
│    (model_1_CNN.keras)           │    │    (Layer Activation Extraction) │
└──────────────────────────────────┘    └──────────────────────────────────┘
          │                                        │
          ▼                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     Streamlit Clinical UI Dashboard                      │
│   • Class Probabilities  • Thermal Overlays  • Diagnostic Knowledge Base │
└──────────────────────────────────────────────────────────────────────────┘
          │
          ▼




pulmoscan-pacs/
├── chest_xray_dataset/                  # Sample test images & evaluation dataset
├── model/                               # Trained neural network artifacts
│   └── model_1_CNN.keras                # Core deep learning model weights
├── venv/                                # Local Python virtual environment
├── app.py                               # Primary Streamlit web platform
├── COVID_19_Radiography_with_CNN.ipynb  # Jupyter notebook for EDA, training & evaluation
├── covid19-radiography-database.zip     # Compressed dataset archive
├── LICENSE                              # Project license text
├── README.md                            # Comprehensive project documentation
└── requirements.txt                     # Python dependencies
┌──────────────────────────────────┐
│   Structured PDF Report Export   │
└──────────────────────────────────┘


## Installation

1. **Clone the repository:**

```bash
git clone <https://github.com/AhmedBushraAhmed/Chest-XRay-Vision-AI>
cd Chest-XRay-Vision-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
