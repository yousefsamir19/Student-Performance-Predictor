import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# Load trained models
# ===============================
with open("linear_model.pkl", "rb") as f:
    linear_model = pickle.load(f)

with open("poly_model.pkl", "rb") as f:
    poly_model = pickle.load(f)

with open("poly_transformer.pkl", "rb") as f:
    poly_transformer = pickle.load(f)

st.title("🎓 Student Performance Prediction")
st.write("Predict exam scores based on student factors.")

# ===============================
# User Inputs
# ===============================
hours = st.slider("Hours Studied per Day", 0, 12, 5)
tutoring = st.slider("Tutoring Sessions per Week", 0, 10, 2)
motivation = st.selectbox("Motivation Level", {"Low":0, "Medium":1, "High":2})
parent_involve = st.selectbox("Parental Involvement", {"Low":0, "Medium":1, "High":2})
resources = st.selectbox("Access to Resources", {"Low":0, "Medium":1, "High":2})

# Example input (⚠️ add more features if needed to match your dataset’s X_train columns!)
input_data = pd.DataFrame([{
    "Hours_Studied": hours,
    "Tutoring_Sessions": tutoring,
    "Motivation_Level": motivation,
    "Parental_Involvement": parent_involve,
    "Access_to_Resources": resources
}])

# ===============================
# Predictions
# ===============================
# Linear Regression
linear_pred = linear_model.predict(input_data)[0]

# Polynomial Regression
input_poly = poly_transformer.transform(input_data)
poly_pred = poly_model.predict(input_poly)[0]

st.subheader("📊 Predicted Exam Scores:")
st.write(f"**Linear Regression:** {linear_pred:.2f}")
st.write(f"**Polynomial Regression (Degree=2):** {poly_pred:.2f}")
