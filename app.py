import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ====================================================
# Load or Train Models
# ====================================================
def load_or_train_models():
    if os.path.exists("linear_model.pkl") and os.path.exists("poly_model.pkl") and os.path.exists("poly_transformer.pkl"):
        with open("linear_model.pkl", "rb") as f:
            linear_model = pickle.load(f)
        with open("poly_model.pkl", "rb") as f:
            poly_model = pickle.load(f)
        with open("poly_transformer.pkl", "rb") as f:
            poly_transformer = pickle.load(f)
        return linear_model, poly_model, poly_transformer, None  # no need to return columns
    else:
        # Load dataset
        df = pd.read_csv("StudentPerformanceFactors.csv")

        
        target_column = "Exam_Score"

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Train linear regression
        linear_model = LinearRegression().fit(X, y)

        # Train polynomial regression
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression().fit(X_poly, y)

        # Save models
        with open("linear_model.pkl", "wb") as f:
            pickle.dump(linear_model, f)
        with open("poly_model.pkl", "wb") as f:
            pickle.dump(poly_model, f)
        with open("poly_transformer.pkl", "wb") as f:
            pickle.dump(poly, f)

        return linear_model, poly_model, poly, list(X.columns)


# ====================================================
# Streamlit App
# ====================================================
st.title("🎓 Student Performance Predictor")

st.write("Enter student details (like in the dataset) to predict their performance.")

# Load or train models
linear_model, poly_model, poly_transformer, columns = load_or_train_models()

# If columns weren’t returned (because models already exist), reload dataset to get column names
if columns is None:
    df = pd.read_csv("StudentPerformanceFactors.csv")
    target_column = "PerformanceIndex"  
    columns = list(df.drop(target_column, axis=1).columns)

# Create input fields dynamically
user_data = {}
for col in columns:
    user_data[col] = st.number_input(f"{col}", value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_data])

# Predictions
if st.button("Predict Performance"):
    linear_pred = linear_model.predict(input_df)[0]
    poly_pred = poly_model.predict(poly_transformer.transform(input_df))[0]

    st.success(f"📊 Linear Regression Prediction: {linear_pred:.2f}")
    st.success(f"📈 Polynomial Regression Prediction: {poly_pred:.2f}")
