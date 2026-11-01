import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classification Dashboard", layout="wide")

st.title("📊 ML Classification Model Evaluation Dashboard")
st.write("Upload test data to evaluate pre-trained classification models without retraining.")

# --- Sidebar: Controls ---
st.sidebar.header("User Input Controls")

# Step 6a: Test Data Upload Option
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

# Step 6b: Model Selection Dropdown
model_option = st.sidebar.selectbox(
    "Select ML Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    )
)

# Model file mapping
model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl"
}

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    if 'target' not in test_df.columns:
        st.error("Error: Uploaded dataset must contain a column named 'target'.")
    else:
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        # Load scaler & transform features
        try:
            scaler = joblib.load('model/scaler.pkl')
            X_test_scaled = scaler.transform(X_test)
        except Exception:
            X_test_scaled = X_test # Fallback if scaling was done directly or not used

        # Load selected model
        model = joblib.load(model_files[model_option])
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Step 6c: Display Metrics
        st.subheader(f"📈 Evaluation Metrics for {model_option}")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")

        # Step 6d: Confusion Matrix & Classification Report
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col_right:
            st.subheader("📄 Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))

else:
    st.info("👈 Please upload a `test_data.csv` file from the sidebar to begin evaluation.")