Automated Dental and Gum Health Detection WebApp

Overview

This repository contains a Jupyter Notebook (Dental_Disease_Detection_ECNN.ipynb) and a web application built with Streamlit that implement a Convolutional Neural Network (CNN) for detecting dental and gum diseases.
The project uses deep learning to analyze oral images and classify various dental health conditions, aiding in early diagnosis and better oral care management.

Table of Contents

Technologies Used
Dataset
Features
Getting Started
Usage
Results
Acknowledgments
Technologies Used

Python
TensorFlow
Keras
OpenCV
NumPy
Matplotlib
scikit-image
Streamlit
Dataset

The dataset consists of intraoral images categorized into different classes such as:

Hypodontia (missing teeth)
Tooth discoloration
Calculus (tartar)
Caries (cavities)
Gingivitis
Mouth ulcers
Healthy teeth
The dataset is curated from multiple open-source dental datasets and manually annotated using tools like Roboflow. This labeled data is crucial for training and evaluating the CNN model.

Features

Image preprocessing and augmentation
Model architecture based on Efficient Convolutional Neural Networks (E-CNN)
Training with the Adamax optimizer
Real-time disease detection through a web interface
Grad-CAM visualization for explainable AI (XAI)
Infection area estimation using clustering techniques
Model evaluation with detailed classification metrics
Getting Started

Prerequisites
Make sure you have:

Python 3.x
Jupyter Notebook
Streamlit
Required Python packages (listed in requirements.txt)
Installation
Clone the repository:
git clone https://github.com/yourusername/Automated-Dental-Gum-Health-Detection.git
Navigate to the project directory:
cd Automated-Dental-Gum-Health-Detection
(Optional) Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:
pip install -r requirements.txt
(If requirements.txt is not available, manually install the libraries: TensorFlow, Keras, Streamlit, OpenCV, NumPy, Matplotlib, scikit-image.)
Usage

To run the model training:
jupyter notebook Dental_Disease_Detection_ECNN.ipynb
To launch the web application:
streamlit run app.py
Follow the instructions in the notebook and app to upload dental images, predict diseases, visualize infection areas, and view diagnostic explanations.

Ensure that the dataset is placed correctly as specified in the notebook and app files.

Results

The model's performance is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Visualizations for loss and accuracy curves over training epochs are provided. Grad-CAM heatmaps highlight the regions responsible for disease detection, ensuring transparency in predictions. The web app offers an easy-to-use interface for real-time dental health diagnostics.

Acknowledgments

TensorFlow and Keras for their deep learning frameworks.
Open-source contributors in the field of dental image analysis.
Roboflow for data annotation tools.
Researchers and datasets enabling advancements in dental health AI.
