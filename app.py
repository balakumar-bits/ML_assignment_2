import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(page_title="Water Potability ML Dashboard", layout="wide")

# Main Title Area
st.title("🌊 Water Quality & Potability ML Dashboard")
st.write("End-to-end Machine Learning pipeline featuring Data Cleaning, EDA, Model Evaluation, and Deployment.")

# --- Sidebar Controls ---
st.sidebar.header("Configuration Panel")

# Display Name and ID in the Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Student Info")
st.sidebar.text("Name: Balakumar R")
st.sidebar.text("ID: 2026ak05509")
st.sidebar.markdown("---")

# Step 6a: Dataset upload option (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

# Step 6b: Model selection dropdown
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

# Mapping model names to saved pipeline files
model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

if uploaded_file is not None:
    raw_test_df = pd.read_csv(uploaded_file)
    
    # Check if target column exists
    if 'target' not in raw_test_df.columns:
        st.error("Error: The uploaded CSV must contain a column named 'target'.")
    else:
        # Create tabs for structured navigation
        tab1, tab2 = st.tabs(["📊 Model Evaluation & Predictions", "🧹 Data Cleaning & EDA"])

        with tab2:
            st.subheader("1. Raw Data Preview (Showing Missing Values / NaNs)")
            st.dataframe(raw_test_df.head())
            
            # Missing value inspection
            missing_counts = raw_test_df.isnull().sum()
            st.write("**Missing Values Detected per Column in Uploaded Test Data:**")
            st.dataframe(missing_counts[missing_counts > 0])

            st.markdown("---")
            st.subheader("2. Automated Pipeline Cleaning Report")
            st.info("The saved Scikit-Learn Pipeline automatically cleans missing values (via median imputation) and scales features prior to prediction.")

            # Extract features and target for display
            X_test_raw = raw_test_df.drop(columns=['target'])
            y_test = raw_test_df['target']

            st.markdown("---")
            st.subheader("3. Exploratory Data Visualization (EDA)")
            
            col_eda1, col_eda2 = st.columns(2)
            
            with col_eda1:
                st.write("**Target Class Distribution (Potability)**")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.countplot(x=y_test, ax=ax, palette="Set2")
                ax.set_xticklabels(["Not Potable (0)", "Potable (1)"])
                st.pyplot(fig)
                
            with col_eda2:
                st.write("**Feature Correlation Heatmap**")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(X_test_raw.corr(), cmap="coolwarm", annot=False, ax=ax)
                st.pyplot(fig)

        with tab1:
            # --- Model Evaluation Tab ---
            X_test = raw_test_df.drop(columns=['target'])
            y_test = raw_test_df['target']
            
            # Load the selected pipeline model (handles imputation & scaling automatically!)
            model = joblib.load(model_files[model_option])
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Step 6c: Metrics Calculation
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.subheader(f"📈 Evaluation Metrics for: {model_option}")
            
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
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

            with col_right:
                st.subheader("📄 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0))

else:
    st.info("👈 Please upload your `test_data.csv` file using the sidebar to begin exploring and evaluating models.")