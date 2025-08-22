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
        # Load dataset
        df = pd.read_csv("StudentPerformanceFactors.csv")

        # -----------------------
        # Set your target column
        # -----------------------
        target_column = "Exam_Score"  # <-- change if your CSV target column is different

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Encode categorical columns automatically
        for col in X.select_dtypes(include="object").columns:
            X[col] = pd.factorize(X[col])[0]

        # Train Linear Regression
        linear_model = LinearRegression().fit(X, y)

        # Train Polynomial Regression
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

# If columns are not returned, get them from CSV
if columns is None:
    df = pd.read_csv("StudentPerformanceFactors.csv")
    target_column = "Exam_Score"
    columns = list(df.drop(target_column, axis=1).columns)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🎓 Student Performance Predictor")
st.write("Enter student details (like in the dataset) to predict their exam score.")

# Create dynamic input fields for all features
user_input = {}
for col in columns:
    user_input[col] = st.text_input(f"{col}", "0")  # default as string

# Convert inputs to numeric if possible
input_df = pd.DataFrame([user_input])
for col in input_df.columns:
    try:
        input_df[col] = pd.to_numeric(input_df[col])
    except:
        # Convert categorical text to factor codes as in training
        input_df[col] = pd.factorize(input_df[col])[0]

# Predict on button click
if st.button("Predict"):
    linear_pred = linear_model.predict(input_df)[0]
    poly_pred = poly_model.predict(poly_transformer.transform(input_df))[0]

    st.success(f"📊 Linear Regression Prediction: {linear_pred:.2f}")
    st.success(f"📈 Polynomial Regression Prediction: {poly_pred:.2f}")
