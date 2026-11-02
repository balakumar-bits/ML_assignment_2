import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 1. Load Dataset
df = pd.read_csv('drinking_water_potability_full_data.csv')

# Separate Features (X) and Target (y)
X = df.drop(columns=['Potability'])
y = df['Potability']

# 2. Train-Test Split (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Save test dataset (features + target) for Streamlit upload requirement
# We keep raw X_test (with NaNs) so the Streamlit app can demonstrate cleaning/imputation!
test_df = X_test.copy()
test_df['target'] = y_test.values
test_df.to_csv('test_data.csv', index=False)
print("Saved 'test_data.csv' successfully.")

# 4. Define Classifiers
models = {
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# Create model directory
os.makedirs('model', exist_ok=True)

# 5. Train, Evaluate, and Build Pipelines
results = []

for name, model in models.items():
    # --- PIPELINE APPROACH ---
    # Step 1: Impute missing values with median
    # Step 2: Scale features with StandardScaler
    # Step 3: Train the Classifier
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Train pipeline on raw training data (which contains NaNs)
    pipeline.fit(X_train, y_train)
    
    # Save the full pipeline artifact
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(pipeline, f'model/{filename}')
    
    # Predict & Evaluate on Test Set (pipeline automatically handles NaN imputation and scaling!)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['classifier'], "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
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
print("\n--- Model Evaluation Summary ---")
print(results_df.to_string(index=False))