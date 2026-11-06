# ML_assignment_2

# Machine Learning Classification & Streamlit Web App

## Student Information
- **Name:** Balakumar R
- **ID:** 2026ak05509

---

## a. Problem Statement
The objective of this project is to implement, evaluate, and deploy an end-to-end machine learning classification pipeline to predict water potability (safe vs. unsafe drinking water) based on chemical and physical water quality metrics. The system includes an interactive Streamlit web dashboard to demonstrate model inference, data cleaning, and metric evaluation without retraining.

---

## b. Dataset Description
- **Dataset Name:** Water Quality & Potability Dataset
- **Source:** Kaggle / UCI Repository (`drinking_water_potability_full_data.csv`)
- **Problem Type:** Binary Classification (`0` = Not Potable, `1` = Potable)
- **Minimum Instance Size:** > 500 rows
- **Minimum Feature Size:** $\ge 12$ features (including pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, and Turbidity).

---

## c. GitHub Repository Link
https://github.com/balakumar-bits/ML_assignment_2
---

## d. Models Used & Evaluation Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.5244 | 0.5474 | 0.4146 | 0.5312 | 0.4658 | 0.0500 |
| **Decision Tree** | 0.5960 | 0.5703 | 0.4813 | 0.4531 | 0.4668 | 0.1423 |
| **KNN** | 0.6143 | 0.5978 | 0.5093 | 0.3203 | 0.3933 | 0.1392 |
| **Naive Bayes** | 0.6143 | 0.6090 | 0.5161 | 0.1875 | 0.2751 | 0.1049 |
| **Support Vector Machine** | 0.6707 | 0.6487 | 0.7041 | 0.2695 | 0.3898 | 0.2696 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Acts as a linear baseline. While class balancing improved recall, overall metrics remain modest due to non-linear relationships and feature overlaps in water quality indicators. |
| **Decision Tree** | Captures non-linear thresholds better than linear models, yielding moderate accuracy and an improved Matthews Correlation Coefficient (MCC of 0.1423). |
| **KNN** | Distance-based classification performs decently after feature scaling, achieving balanced accuracy around 61.4% with reasonable precision. |
| **Naive Bayes** | Relies on strong feature independence assumptions, resulting in high precision but low recall for positive potability classes. |
| **Support Vector Machine** | **Best Performing Model.** Achieves the highest Accuracy (67.07%), AUC (0.6487), Precision (70.41%), and MCC (0.2696) by finding an optimal margin boundary across scaled features. |

### Overall Winner for this Dataset
**Support Vector Machine (SVM)** is the overall winner for this dataset, demonstrating superior overall discriminative power and higher classification reliability compared to the linear and probabilistic alternatives.

---

## Project Structure
```text
project-folder/
│-- app.py                  # Interactive Streamlit Web Application
│-- train_and_save_models.py# Training script with Scikit-Learn Pipelines
│-- requirements.txt        # Required python dependencies
│-- README.md               # Project documentation & report
│-- test_data.csv           # Test dataset upload sample (with target column)
└── model/                  # Saved model pipeline artifacts (.pkl files)
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    └── support_vector_machine.pkl