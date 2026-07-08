import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Credit Scoring Model", page_icon="💳")

st.title("💳 Credit Scoring Model")
st.write("Decision Tree and Random Forest Performance")

# Load trained Random Forest model
model = joblib.load("rf_model.pkl")

# Load feature names
feature_names = joblib.load("feature_names.pkl")

st.header("Model Information")

st.write("**Algorithm:** Random Forest Classifier")

st.write("**Task:** Credit Scoring Prediction")

st.success("Model trained successfully.")

# Feature Importance
st.header("Feature Importance")

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

st.dataframe(importance)

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(importance["Feature"], importance["Importance"])
plt.xticks(rotation=90)
plt.tight_layout()

st.pyplot(fig)

st.markdown("---")
st.write("Developed using Python, Scikit-learn and Streamlit.")
#Evaluation metrics
metrics = joblib.load("metrics.pkl")

st.header("Model Performance")

st.write("### Accuracy")
st.write(metrics["Accuracy"])

st.write("### ROC-AUC Score")
st.write(metrics["roc_auc_score"])

st.write("### Classification Report")
st.text(metrics["classification_report"])

st.write("### Confusion Matrix")
st.write(metrics["confusion_matrix"])
