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

        target_column = "Exam_Score"  # change if different
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Encode categorical columns automatically
        categorical_cols = X.select_dtypes(include="object").columns
        for col in categorical_cols:
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

# Get feature columns if not returned
if columns is None:
    df = pd.read_csv("StudentPerformanceFactors.csv")
    target_column = "Exam_Score"  # change if different
    columns = list(df.drop(target_column, axis=1).columns)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict their exam score.")

# Load dataset to get unique values for categorical features
df = pd.read_csv("StudentPerformanceFactors.csv")
X = df.drop(target_column, axis=1)

user_input = {}
for col in columns:
    if X[col].dtype == "object":
        # Categorical → dropdown
        options = X[col].unique().tolist()
        user_input[col] = st.selectbox(f"{col}", options)
    else:
        # Numeric → slider
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        user_input[col] = st.slider(f"{col}", min_val, max_val, mean_val)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical columns like training
for col in input_df.select_dtypes(include="object").columns:
    input_df[col] = pd.factorize(input_df[col])[0]

# Predict on button click
if st.button("Predict"):
    linear_pred = linear_model.predict(input_df)[0]
    poly_pred = poly_model.predict(poly_transformer.transform(input_df))[0]

    st.success(f"Prediction: {linear_pred:.2f}")
    #st.success(f"📈 Polynomial Regression Prediction: {poly_pred:.2f}")

