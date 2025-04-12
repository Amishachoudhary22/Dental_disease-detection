# 🦷 Automated Dental and Gum Health Detection WebApp

This repository contains a Web Application that leverages a Convolutional Neural Network (CNN) model for automated detection of various dental and gum health issues from intraoral images. The project uses deep learning to analyze images and classify the dental condition, aiming to assist in early diagnosis and preventive care.

---

## 📑 Table of Contents
- [Technologies Used](#-technologies-used)
- [Dataset](#-dataset)
- [Features](#-features)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Results](#-results)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Technologies Used
- Python 3
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit (for Web App deployment)

---

## 📂 Dataset

The dataset consists of annotated images of human teeth and gums, categorized into multiple classes such as:
- Hypodentia
- Tooth Discoloration
- Calculus (Tartar)
- Caries (Cavities)
- Gingivitis
- Mouth Ulcers

Data preprocessing, augmentation, and labeling techniques were applied to enhance the dataset's diversity and model robustness.

---

## ✨ Features
- Data loading, preprocessing, and augmentation
- CNN-based model architecture
- Transfer learning with Efficient Convolutional Neural Networks (E-CNN)
- Model fine-tuning and optimization using Adamax optimizer
- Real-time prediction through a simple and user-friendly web interface
- Visualization of training metrics (Accuracy, Loss over epochs)
- Infected region detection using K-Means clustering and Grad-CAM heatmaps
- Deployment-ready Streamlit Web Application

---

## ⚙️ Getting Started

### Prerequisites
To run this project locally, you need:

- Python 3.x
- Jupyter Notebook
- Streamlit
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Automated-Dental-Gum-Health-Detection.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Automated-Dental-Gum-Health-Detection
    ```

3. **(Optional) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # For Windows: venv\Scripts\activate
    ```

4. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Or manually install main libraries:_
    ```bash
    pip install tensorflow keras numpy matplotlib opencv-python streamlit
    ```

---

## 🖥️ Usage

- **For training/testing the model:**
    Open the Jupyter Notebook:
    ```bash
    jupyter notebook Dental_Disease_Detection_Model.ipynb
    ```
    Follow the instructions in the notebook for training, evaluation, and visualization.

- **For running the Web App:**
    ```bash
    streamlit run app.py
    ```

Make sure the trained model files are correctly placed in the project directory or loaded dynamically.

---

## 📊 Results

The model's performance is evaluated based on:
- Accuracy
- Precision
- Recall
- F1-Score

Graphs of **training/validation loss** and **accuracy** over epochs are generated.  
The classification report and confusion matrix provide detailed insights into the model's effectiveness.

Grad-CAM visualizations highlight infected regions, making predictions more explainable for users.

---

## 🙏 Acknowledgments
- TensorFlow and Keras for providing powerful deep learning frameworks.
- Researchers and open-source contributors in the field of medical image analysis.
- Roboflow for facilitating dataset preprocessing and augmentation.
- Streamlit for simplifying Web App deployment.

---

## 📌 Project Status
✅ Completed model training and evaluation.  
✅ Developed a Streamlit-based interactive WebApp.  
✅ Future work: Expanding the dataset and model explainability features.

---

> ⚡ _Built with passion for combining healthcare and AI technologies!_
