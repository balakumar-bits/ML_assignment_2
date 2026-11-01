import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 1. Load Dataset (Example using synthetic/placeholder structure or loaded CSV)
# Replace 'dataset.csv' with your chosen Kaggle/UCI CSV path
df = pd.read_csv('dataset.csv')

# Ensure target column is named 'target'
X = df.drop(columns=['target'])
y = df['target']

# 2. Train-Test Split (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save test dataset (features + target) for Streamlit upload requirement
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv('test_data.csv', index=False)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model directory
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

# 4. Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# 5. Train & Evaluate Models
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    # Save Model File
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f'model/{filename}')
    
    # Predict & Evaluate on Test Set
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })

# Output metrics table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))