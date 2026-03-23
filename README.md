# Handwritten Digit Recognition with MNIST

This project implements and compares a wide range of machine learning and deep learning models for handwritten digit classification using the **MNIST** dataset.

## 📌 Goal

Classify grayscale images of handwritten digits (0–9) into 10 classes with high accuracy, and understand the strengths and limitations of different modelling approaches.

## 📁 Dataset

- **MNIST** – 70,000 images of 28×28 pixels (60,000 training, 10,000 testing).
- Each pixel is normalised to a float in `[0,1]`.
- Labels are integers 0–9.

## 🧠 Models Explored

### Part 1: Traditional Machine Learning
- **k‑Nearest Neighbors** – simple instance‑based learning.
- **Logistic Regression** – linear classifier with softmax.
- **Decision Trees** – with pruning and visualisation.
- **Random Forest** – ensemble of trees.
- **Support Vector Machine (SVM)** – with RBF kernel.

### Part 2: Deep Learning (CNN & Regularisation)
- **Baseline Dense Network** – 2 hidden layers.
- **CNN (base)** – 2 convolutional + pooling layers.
- **CNN + L2 / L1 / ElasticNet regularisation**
- **CNN + Dropout**
- **CNN + Batch Normalisation**
- **CNN + Early Stopping**
- **CNN + Learning Rate Decay**
- **CNN + Data Augmentation**
- **Autoencoder (unsupervised) + Classifier**
- **Transfer Learning** from CIFAR‑10

## 🛠️ Requirements

Install the required libraries:

```bash
pip install numpy matplotlib seaborn pandas scikit-learn tensorflow
