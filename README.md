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

Model	Test Accuracy	Precision (macro)	Recall (macro)	F1-Score (macro)
Random Forest 	0.9694	0.969271	0.96911	0.969172
SVM (RBF, Tuned)	0.9688	0.96871	0.968404	0.968517
K-Nearest Neighbour	0.9589	0.959	0.958391	0.958518
Tuned Logistic Regression	0.9245	0.923627	0.923353	0.923355
Default Sklearn Logistic Regression	0.9244	0.864438	0.864082	0.864059
Scratch Logistic Regression	0.9042	0.864438	0.864082	0.864059
Decision Tree (high varience)	0.8661	0.864438	0.864082	0.864059
<img width="1292" height="230" alt="image" src="https://github.com/user-attachments/assets/3826db34-25fb-44d0-9c28-da7a7f49db0b" />
