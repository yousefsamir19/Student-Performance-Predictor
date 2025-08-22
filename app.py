import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# -------------------------------
# Load or Train Models
# -------------------------------
def load_or_train_models():
    if (
        os.path.exists("linear_model.pkl")
        and os.path.exists("poly_model.pkl")
        and os.path.exists("poly_transformer.pkl")
    ):
        with open("linear_model.pkl", "rb") as f:
            linear_model = pickle.load(f)
        with open("poly_model.pkl", "rb") as f:
            poly_model = pickle.load(f)
        with open("poly_transformer.pkl", "rb") as f:
            poly_transformer = pickle.load(f)
        return linear_model, poly_model, poly_transformer, None
    else:
        df = pd.read_csv("StudentPerformanceFactors.csv")
        target_column = "Exam_Score"  # Change if different
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Encode categorical columns
        for col in X.select_dtypes(include="object").columns:
            X[col] = pd.factorize(X[col])[0]

        linear_model = LinearRegression().fit(X, y)
        poly_transformer = PolynomialFeatures(degree=2)
        X_poly = poly_transformer.fit_transform(X)
        poly_model = LinearRegression().fit(X_poly, y)

        # Save models
        with open("linear_model.pkl", "wb") as f:
            pickle.dump(linear_model, f)
        with open("poly_model.pkl", "wb") as f:
            pickle.dump(poly_model, f)
        with open("poly_transformer.pkl", "wb") as f:
            pickle.dump(poly_transformer, f)

        return linear_model, poly_model, poly_transformer, list(X.columns)

# -------------------------------
# Load Models
# -------------------------------
linear_model, poly_model, poly_transformer, columns = load_or_train_models()

if columns is None:
    df = pd.read_csv("StudentPerformanceFactors.csv")
    target_column = "Exam_Score"
    columns = list(df.drop(target_column, axis=1).columns)

df = pd.read_csv("StudentPerformanceFactors.csv")
X = df.drop(target_column, axis=1)

st.set_page_config(page_title="Scholar Score Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Scholar Score Predictor")
st.write("Predict student exam scores based on various performance factors.")

# -------------------------------
# Input Form in Columns
# -------------------------------
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
categorical_cols = X.select_dtypes(include="object").columns.tolist()

with st.form(key="input_form"):
    st.subheader("📥 Enter Student Details")
    col1, col2 = st.columns(2)

    user_input = {}
    for i, col in enumerate(numeric_cols):
        slider_col = col1 if i % 2 == 0 else col2
        min_val, max_val = float(X[col].min()), float(X[col].max())
        mean_val = float(X[col].mean())
        user_input[col] = slider_col.slider(f"{col}", min_val, max_val, mean_val)

    for i, col in enumerate(categorical_cols):
        dropdown_col = col1 if i % 2 == 0 else col2
        options = X[col].unique().tolist()
        user_input[col] = dropdown_col.selectbox(f"{col}", options)

    submit_button = st.form_submit_button(label="Predict")

# -------------------------------
# Make Predictions
# -------------------------------
if submit_button:
    input_df = pd.DataFrame([user_input])

    # Encode categorical inputs
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = pd.factorize(input_df[col])[0]

    linear_pred = linear_model.predict(input_df)[0]
    poly_pred = poly_model.predict(poly_transformer.transform(input_df))[0]

    st.subheader("📊 Predictions")
    st.metric("Prediction", f"{linear_pred:.2f}")
    #st.metric("Polynomial Regression Prediction", f"{poly_pred:.2f}")
