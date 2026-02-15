import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
)

 # 1. Page Configuration
st.set_page_config(page_title="ML Model Evaluator", layout="wide")
st.title("Classification Model Evaluator")
st.write("Upload a test dataset to evaluate the 6 implemented models.")
st.write("By Akhilesh Deepak Jichkar - 2025AA05613")


# 2. Sidebar: Model Selection and Data Upload
with st.sidebar:
    st.header("Upload & Configure")
    # (a): Dataset upload option (CSV)
    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])
    
    # (b): Model selection dropdown
    model_option = st.selectbox(
        "Select Model to Evaluate",
        ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
    )

# Map dropdown names
model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    model_features = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

    if 'ID' in test_data.columns:
        test_data = test_data.drop(columns=['ID'])
    elif test_data.columns[0].lower() == 'id':
        test_data = test_data.iloc[:, 1:]

    st.success("Test data uploaded successfully!")
    
    # assume the last column is the target
    X_test = test_data.iloc[:, :-1]
    y_true = test_data.iloc[:, -1]

    if X_test.shape[1] == 23:
        X_test.columns = model_features
    else:
        st.error(f"Error: Expected 23 feature columns, but got {X_test.shape[1]}. Please check your CSV.")
        st.stop()

    # Load Model and Scaler
    try:
        model = joblib.load(f"model/{model_map[model_option]}")
        
        # Load the scaler if the model is distance-based
        if model_option in ["Logistic Regression", "KNN", "Naive Bayes"]:
            scaler = joblib.load("model/scaler.pkl")
            X_test = scaler.transform(X_test)

        # Make Predictions
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # (c): Display of Evaluation Metrics
        st.header(f"Results for: {model_option}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        col2.metric("AUC Score", f"{roc_auc_score(y_true, y_probs):.4f}")
        col3.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
        col5.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
        col6.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")

        st.divider()

        # (d): Confusion Matrix & Classification Report
        st.subheader("Visualizations & Detailed Report")
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

        with v_col2:
            st.write("**Classification Report**")
            report = classification_report(y_true, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"Error loading model or processing data: {e}")
        st.info("Check if model files exist in the /model folder and the test CSV matches the training features.")

else:
    st.info("Please upload a test CSV file in the sidebar to begin.")



## How to get test data - random 100 rows from training data:

# df = pd.read_csv('default of credit card clients.csv')
# test_sample = df.sample(n=100, random_state=999)
# test_sample.to_csv('testing_data.csv', index=False)
# print("'testing_data.csv' created with 100 rows.")
