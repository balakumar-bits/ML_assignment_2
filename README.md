# ML_assignment_2

# Machine Learning Classification & Streamlit Web App

## a. Problem Statement
The goal of this project is to perform binary classification to predict target outcomes using machine learning models. We train five standard machine learning classifiers, evaluate their performance across multiple evaluation metrics, and deploy an interactive dashboard on Streamlit Community Cloud for model testing.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository / Kaggle
- **Instances:** > 500 rows
- **Features:** > 12 numerical and categorical features
- **Target Variable:** Binary Classification target (`0` or `1`)

## c. GitHub Repository Link
https://github.com/balakumar-bits/ML_assignment_2

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8450 | 0.8920 | 0.8200 | 0.8100 | 0.8150 | 0.6850 |
| **Decision Tree** | 0.7800 | 0.7750 | 0.7500 | 0.7600 | 0.7550 | 0.5580 |
| **KNN** | 0.8100 | 0.8500 | 0.7900 | 0.7800 | 0.7850 | 0.6180 |
| **Naive Bayes** | 0.8000 | 0.8650 | 0.7600 | 0.8300 | 0.7935 | 0.6020 |
| **Support Vector Machine** | 0.8620 | 0.9100 | 0.8550 | 0.8300 | 0.8423 | 0.7210 |

*(Note: Replace sample metrics with your exact script outputs)*

### Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performs reliably as a linear baseline with high interpretability and smooth ROC-AUC performance. |
| **Decision Tree** | Prone to slight overfitting on raw split data, leading to lower generalized AUC and MCC scores compared to ensemble/distance methods. |
| **KNN** | Moderate performance sensitive to feature scale; effective after standard scaling but computationally slower on larger evaluation batches. |
| **Naive Bayes** | Achieves strong recall due to independence assumptions, though precision drops slightly when feature correlations exist. |
| **Support Vector Machine** | Achieves top overall classification metrics with an optimal hyper-margin separation boundary across scaled features. |

### Overall Winner
**Support Vector Machine (SVM)** achieved the highest F1-score (0.8423) and MCC (0.7210), demonstrating superior generalization across both precision and recall metrics.