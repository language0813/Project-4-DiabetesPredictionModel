# Diabetes Prediction Model Using Health Indicators

## Overview

This project aims to build a machine learning model to predict the likelihood of diabetes based on various health-related indicators. The dataset used in this project comes from the 2023 Behavioral Risk Factor Surveillance System (BRFSS) survey, which captures data on a wide range of health behaviors and conditions across various demographics. The target variable classifies individuals into two categories: diabetic and non-diabetic (including pre-diabetes).

The project follows a complete pipeline that involves:
- Data preprocessing.
- Addressing class imbalance.
- Training a robust Artificial Neural Network (ANN) for prediction.

---

## Dataset Source

- **Dataset**: [BRFSS 2023 Annual Data](https://www.cdc.gov/brfss/annual_data/annual_2023.html)
- **Referred Code**: [Diabetes Health Indicators Notebook](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook/notebook)
- **Health Indicator References**: [CDC - PCD Article (2019)](https://www.cdc.gov/pcd/issues/2019/19_0109.htm)


---

## Project Workflow

### 1. **Dataset Overview**

The dataset includes various health indicators and a binary target variable (`Diabetes_binary`) representing the presence or absence of diabetes. 

### 2. **Data Preprocessing**

- **Dataset Format**:
  - The 2023 BRFSS Data was initially available in XPT (SAS EXPORT) format. We used the Pandas `read_sas` method to load the dataset into a Pandas DataFrame and subsequently saved it as a CSV file for later use.

- **Feature and Target Separation**:
  - Features (`X`) were separated from the target variable (`y`).
  - The target variable is the `Diabetes_binary` column.

- **One-Hot Encoding**:
  - Categorical columns were identified and one-hot encoded to ensure compatibility with machine learning models.

- **Train-Test Split**:
  - The dataset was split into training (70%) and testing (30%) subsets using a stratified approach to maintain class distribution.


### 3. **Handling Class Imbalance**

The dataset was balanced using the **SMOTETomek** technique:

- **SMOTE (Synthetic Minority Oversampling Technique)**: Synthetic examples of the minority class were generated.
- **Tomek Links**: Instances causing class overlap were removed to enhance class separation.
This resulted in a balanced training dataset, addressing the class imbalance problem that is common in healthcare datasets.

### 4. **Feature Scaling**

All features were standardized using **StandardScaler** to normalize feature values. This step ensures that all features contribute equally to the model and aids in faster and more stable training.

### 5. **Neural Network Model**

We implemented a fully connected artificial neural network using TensorFlow and Keras. The architecture includes:

- **Input Layer**: Matches the number of features in the dataset.
- **Hidden Layers**:
  - Three dense layers with ReLU activation for non-linear transformations.
  - Batch normalization layers for stabilizing and accelerating training.
  - Dropout layers (50% rate) to prevent overfitting.
- **Output Layer**: A single neuron with a sigmoid activation function for binary classification.

### 6. **Compilation and Training**

The model was compiled with:

- **Loss Function**: `binary_crossentropy` for binary classification tasks.
- **Optimizer**: `Adam` with a learning rate of 0.001 for efficient gradient descent.
- **Metrics**: Accuracy was used as the primary performance metric.

Training Details:

- The model was trained for 20 epochs with a batch size of 32.
- Validation was performed on the test dataset to monitor generalization.

### 7. **Evaluation**

The model's performance was evaluated using:

- **Accuracy**: Measures overall predictive correctness.
- **Classification Report**: Includes precision, recall, F1-score, and support for both classes.

---

## Results

The model demonstrated high accuracy and robust performance on the test set, making it a reliable tool for diabetes prediction based on health indicators.

### Key Metrics:

#### **Accuracy**: 75%

![image](https://github.com/user-attachments/assets/6443be22-d434-4069-8cc8-8c3b7f707e8b)

---

## Methods Summary

- **Preprocessing**: Feature separation, one-hot encoding, SMOTETomek balancing, and feature scaling.
- **Model**: Fully connected neural network with dropout, batch normalization, and ReLU activations.
- **Evaluation**: Classification metrics and accuracy.

---

## Usage

To reproduce this project:
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn tensorflow
   ```
2. Place the dataset in the specified path: `Data/diabetes_binary_5050split_health_indicators_BRFSS2023.csv`.
3. Run the script.

## Conclusion

This project demonstrates how to process and model health-related data to predict diabetes using machine learning techniques. The dataset was preprocessed to handle missing values, reclassify variables, and balance the target classes. A neural network was trained, and its performance evaluated to predict diabetes effectively.
