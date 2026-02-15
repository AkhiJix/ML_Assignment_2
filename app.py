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
