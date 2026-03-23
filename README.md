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

## 📊 Results

### Traditional Machine Learning Models

| Model                           | Test Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|---------------------------------|---------------|-------------------|----------------|------------------|
| Random Forest                   | 0.9694        | 0.9693            | 0.9691         | 0.9692           |
| SVM (RBF, Tuned)                | 0.9688        | 0.9687            | 0.9684         | 0.9685           |
| K-Nearest Neighbour             | 0.9589        | 0.9590            | 0.9584         | 0.9585           |
| Tuned Logistic Regression       | 0.9245        | 0.9236            | 0.9234         | 0.9234           |
| Default Sklearn Logistic Reg.   | 0.9244        | 0.8644            | 0.8641         | 0.8641           |
| Scratch Logistic Regression     | 0.9042        | 0.8644            | 0.8641         | 0.8641           |
| Decision Tree (high variance)   | 0.8661        | 0.8644            | 0.8641         | 0.8641           |

### Deep Learning Models

| Model                           | Train Acc | Test Acc | Precision | Recall | F1   | Gen. Gap |
|---------------------------------|-----------|----------|-----------|--------|------|----------|
| CNN (base)                      | 0.9966    | **0.9908** | 0.9908    | 0.9907 | 0.9907 | 0.0058   |
| CNN + LR Decay                  | 0.9966    | 0.9899   | 0.9898    | 0.9897 | 0.9898 | 0.0067   |
| CNN + BatchNorm                 | 0.9959    | 0.9896   | 0.9897    | 0.9895 | 0.9895 | 0.0063   |
| CNN + Dropout                   | 0.9903    | 0.9886   | 0.9886    | 0.9885 | 0.9885 | 0.0017   |
| CNN + Early Stopping            | 0.9920    | 0.9870   | 0.9871    | 0.9869 | 0.9870 | 0.0050   |
| CNN + L2                        | 0.9886    | 0.9854   | 0.9855    | 0.9852 | 0.9852 | 0.0032   |
| CNN + Data Augmentation         | 0.9862    | 0.9831   | 0.9833    | 0.9830 | 0.9830 | 0.0031   |
| Baseline Dense                  | 0.9902    | 0.9765   | 0.9767    | 0.9764 | 0.9764 | 0.0137   |
| CNN + L1                        | 0.9736    | 0.9759   | 0.9758    | 0.9758 | 0.9757 | -0.0023  |
| CNN + ElasticNet                | 0.9710    | 0.9729   | 0.9729    | 0.9727 | 0.9727 | -0.0019  |
| Autoencoder + Classifier        | 0.8982    | 0.9073   | 0.9117    | 0.9063 | 0.9061 | -0.0091  |

> **Note**: Negative generalisation gap (test accuracy > training accuracy) can occur due to regularisation or random validation splits.

## 🔍 Key Observations

- **Traditional ML**: Random Forest and SVM achieve ~97% accuracy, outperforming linear models and k‑NN.
- **CNNs dominate**: The base CNN reaches **99.08%** test accuracy, significantly higher than any traditional model.
- **Regularisation trade‑off**: Dropout reduces the generalisation gap to just 0.17% while maintaining high accuracy; L1 and ElasticNet are too aggressive for MNIST.
- **Data augmentation** improves robustness but requires more training time to match the base CNN.
- **Autoencoder pre‑training** underperforms because the encoder was frozen; fine‑tuning would improve results.
- **Transfer learning** from CIFAR‑10 was not included in the final deep learning table but is available in the notebook; domain gap limits its effectiveness.

## 📈 Dimensionality Reduction (PCA / t‑SNE)

We applied **PCA** and **t‑SNE** to visualise the high‑dimensional pixel data in 2D:

- PCA captures global variance but may not separate digits well.
- t‑SNE produces distinct clusters corresponding to digit classes, confirming the separability of the data.

Run the provided notebook to see these visualisations.

## 🚀 How to Run

1. Clone the repository.
2. Install dependencies: `pip install numpy matplotlib seaborn pandas scikit-learn tensorflow`
3. Open and run the Jupyter notebooks:
   - `advanced_ml_models_mnist.ipynb` – for traditional ML models.
   - `deep_learning_mnist.ipynb` – for deep learning experiments.
4. The results tables and plots will be generated automatically.

## 📚 Learning Objectives Achieved

- ✅ **Image data representation** – understanding how images are flattened or used as 2D arrays.
- ✅ **Multiclass classification** – handling 10 digit classes with softmax.
- ✅ **Dimensionality reduction** – using PCA and t‑SNE for visualisation.
- ✅ **Model evaluation** – comparing accuracy, precision, recall, F1, and generalisation gap.
- ✅ **Regularisation techniques** – L1, L2, dropout, early stopping.
- ✅ **Architectural design** – building CNNs, autoencoders, and applying transfer learning.

## 📝 Conclusion

The **base CNN** is the best performer in terms of test accuracy, while **dropout** offers the best generalisation. Traditional ML models like Random Forest and SVM also achieve competitive results (~97%), but CNNs clearly dominate due to their ability to learn hierarchical spatial features.

This project provides a comprehensive hands‑on experience with both classical and deep learning methods for image classification.

## 📄 License

MIT
