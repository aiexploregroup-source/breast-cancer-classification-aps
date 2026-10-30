# 🧬 Breast Scan AI
### Deep Learning-Based Breast Cancer Histopathological Image Classification

<p align="center">
  <strong>Team Breast Scan AI</strong><br>
  An AI-powered academic project exploring deep learning for histopathological image classification.
</p>

<p align="center">
  <a href="https://breast-cancer-classification-aps.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Launch_Application-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="https://github.com/aiexploregroup-source/breast-cancer-classification-aps">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Deep_Learning-CNN%20%7C%20ResNet50-orange?style=flat-square">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Interface-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Project-2026--27-blue?style=flat-square">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

---

## 🔬 Overview

**Breast Scan AI** is a collaborative academic project that demonstrates the application of Deep Learning and Computer Vision in the classification of breast cancer histopathological images.

The project explores two neural network architectures:

- **Custom Convolutional Neural Network (CNN)**
- **ResNet50 with Transfer Learning**

Using the publicly available **BreaKHis dataset**, the models are trained to distinguish between two image classes:

- 🟢 **Benign:** Non-malignant tissue images.
- 🔴 **Malignant:** Malignant tissue images.

A Streamlit-based web application provides an interactive interface where users can upload histopathological images and view the corresponding model predictions.

> **Important Medical Disclaimer**
>
> This project is developed strictly for educational and research demonstration purposes. It is not a clinically validated medical device and must not be used for medical screening, diagnosis, treatment decisions, or reassurance about an individual's health. Model predictions are experimental outputs, not medical conclusions.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 Deep Learning | Implements CNN and ResNet50 architectures. |
| 🔬 Image Classification | Demonstrates Benign vs. Malignant image classification. |
| 🖼️ Image Upload | Allows users to upload histopathological images. |
| ⚡ Model Inference | Generates predictions through trained models. |
| 🌐 Web Application | Interactive interface built using Streamlit. |
| 📊 Model Evaluation | Explores model performance and classification accuracy. |
| ☁️ Online Deployment | Accessible through a browser without local installation. |

---

## 🧠 Deep Learning Architectures

The project explores two approaches to image classification.

### 1. Custom CNN

A custom Convolutional Neural Network designed to learn visual features from histopathological images.

- Multiple convolutional layers.
- Automated feature extraction.
- Image classification through learned representations.
- Training using the BreaKHis dataset.

### 2. ResNet50

A deep residual network that uses transfer learning to leverage previously learned visual representations.

- Pre-trained ResNet50 architecture.
- Residual connections.
- Transfer learning for image classification.
- Fine-tuning for the target classification task.

### 📈 Model Performance

The following are the reported project results. They should be interpreted as experimental results rather than independently validated clinical performance.

| Model | Reported Accuracy | Approach |
|---|---:|---|
| Custom CNN | ~92% | Custom deep learning architecture |
| ResNet50 | ~95% | Transfer learning and fine-tuning |

**Note:** Accuracy alone does not establish clinical reliability. Dataset separation, class-wise precision, recall, F1-score, sensitivity, specificity, and independent validation are important when evaluating classification models.

---

## 🗂️ Dataset

### BreaKHis: Breast Cancer Histopathological Image Dataset

The project uses the publicly available **BreaKHis dataset**, which contains histopathological images of breast tumor tissue.

| Dataset Information | Details |
|---|---|
| Dataset | BreaKHis |
| Total Images | 7,909 |
| Classification | Benign and Malignant |
| Magnification Levels | 40X, 100X, 200X, 400X |
| Image Type | Histopathological images |
| Source | Kaggle |

The dataset contains eight tumor subclasses grouped into two major categories: benign and malignant.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning and model development |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Web application development |
| PIL / Pillow | Image handling |

---

## 📁 Project Structure

```text
breast-cancer-classification-aps/
│
├── 📓 Detect_BreastCancer.ipynb
│   └── Dataset loading, preprocessing and evaluation
│
├── 📓 train_CustomModel_32_conv_20k.ipynb
│   └── Custom CNN model training
│
├── 📓 train_ResNet50_32_20k.ipynb
│   └── ResNet50 training and fine-tuning
│
├── 🐍 app.py
│   └── Streamlit application and model inference
│
├── 📄 requirements.txt
│   └── Project dependencies
│
└── 📄 README.md
    └── Project documentation
```

---

## 🚀 Live Application

Experience the project through its deployed Streamlit application.

<p align="center">
  <a href="https://breast-cancer-classification-aps.streamlit.app/">
    <img src="https://img.shields.io/badge/OPEN_BREAST_SCAN_AI-Visit_Live_Application-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Live Application">
  </a>
</p>

**Application capabilities:**
1. Open the deployed application.
2. Upload a histopathological image.
3. Allow the application to process the uploaded image.
4. View the model-generated classification output.

*The application is an educational demonstration, not a medical diagnostic service.*

---

## 💻 Installation & Local Setup

Want to explore the project on your own computer? Follow these steps.

### Prerequisites

- Python 3.10 or later
- Git
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/aiexploregroup-source/breast-cancer-classification-aps.git
```

### Step 2: Navigate to the Project Directory

```bash
cd breast-cancer-classification-aps
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Launch the Application

```bash
streamlit run app.py
```

The application should open in your default browser.

---

## 📦 Dependencies

The primary dependencies include:

```text
streamlit
tensorflow
opencv-python
numpy
pillow
matplotlib
seaborn
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🎓 Academic Information

**Project Type:** Collaborative Academic Capstone Project

**Project Year:** 2026–2027

**Institution:** Army Public School, LBS Marg, Lucknow

**Academic Batch:** AI Batch 2026–27

**Project Guide:** Mrs. Seema Agarwal

### 👨‍💻 Project Contributors

| Contributor | Role |
|---|---|
| Chowda Reddy N | Project Contributor |
| N Swaroopa | Project Contributor |

**Team:** Team Breast Scan AI

---

## 🙏 Acknowledgements

We sincerely acknowledge:

- **Kaggle and the BreaKHis dataset contributors** for making the histopathological image dataset publicly accessible.
- **Army Public School, LBS Marg, Lucknow** for providing the academic platform for this project.
- **Mrs. Seema Agarwal** for her valuable academic guidance and support.

---

## 📜 License

This project is distributed under the **MIT License**.

You are free to use, modify, and distribute the project according to the terms of the license.

---

## 📬 Connect With Us

<p align="center">
  <strong>TEAM BREAST SCAN AI</strong><br>
  Army Public School<br>
  LBS Marg, Lucknow
</p>

<p align="center">
  📧 <strong>Email:</strong>
  <a href="mailto:aiexploregroup@gmail.com">aiexploregroup@gmail.com</a>
</p>

---

<p align="center">
  <strong>🧬 Exploring the intersection of Artificial Intelligence and Healthcare.</strong>
  <br><br>
  Built with curiosity, collaboration and code.
  <br><br>
  ⭐ If you find this project interesting, consider starring the repository!
</p>
