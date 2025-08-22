import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

# -------------------------------
# Function to load or train models
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
    else:
        df = pd.read_csv("StudentPerformanceFactors.csv")

        # Encode categorical features
        label_encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        X = df.drop("PerformanceIndex", axis=1)
        y = df["PerformanceIndex"]

        # Train linear regression
        linear_model = LinearRegression().fit(X, y)

        # Train polynomial regression
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

    return linear_model, poly_model, poly_transformer


# -------------------------------
# Load models
# -------------------------------
linear_model, poly_model, poly_transformer = load_or_train_models()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎓 Scholar Score Predictor")
st.write("Predict student performance index based on input factors.")

# Example input fields (you can adjust based on dataset features)
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=75)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=12, value=7)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
motivation = st.slider("Motivation Level (1-10)", min_value=1, max_value=10, value=5)

# Create input DataFrame
input_data = pd.DataFrame(
    {
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Sleep_Hours": [sleep_hours],
        "Previous_Scores": [previous_scores],
        "Motivation_Level": [motivation],
    }
)

# Predictions
if st.button("Predict Performance"):
    linear_pred = linear_model.predict(input_data)[0]
    poly_pred = poly_model.predict(poly_transformer.transform(input_data))[0]

    st.subheader("📊 Predictions")
    st.write(f"**Linear Regression Prediction:** {linear_pred:.2f}")
    st.write(f"**Polynomial Regression Prediction:** {poly_pred:.2f}")
