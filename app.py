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
        and os.path.exists("columns.pkl")
    ):
        with open("linear_model.pkl", "rb") as f:
            linear_model = pickle.load(f)
        with open("poly_model.pkl", "rb") as f:
            poly_model = pickle.load(f)
        with open("poly_transformer.pkl", "rb") as f:
            poly_transformer = pickle.load(f)
        with open("columns.pkl", "rb") as f:
            columns = pickle.load(f)
        return linear_model, poly_model, poly_transformer, columns
    else:
        df = pd.read_csv("StudentPerformanceFactors.csv")
        target_column = "Exam_Score"
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Encode categorical columns
        for col in X.select_dtypes(include="object").columns:
            X[col] = pd.factorize(X[col])[0]

        columns = list(X.columns)

        linear_model = LinearRegression().fit(X, y)
        poly_transformer = PolynomialFeatures(degree=2)
        X_poly = poly_transformer.fit_transform(X)
        poly_model = LinearRegression().fit(X_poly, y)

        # Save models and columns
        with open("linear_model.pkl", "wb") as f:
            pickle.dump(linear_model, f)
        with open("poly_model.pkl", "wb") as f:
            pickle.dump(poly_model, f)
        with open("poly_transformer.pkl", "wb") as f:
            pickle.dump(poly_transformer, f)
        with open("columns.pkl", "wb") as f:
            pickle.dump(columns, f)

        return linear_model, poly_model, poly_transformer, columns

# -------------------------------
# Load Models
# -------------------------------
linear_model, poly_model, poly_transformer, columns = load_or_train_models()
df = pd.read_csv("StudentPerformanceFactors.csv")
target_column = "Exam_Score"
X = df.drop(target_column, axis=1)

st.set_page_config(page_title="Scholar Score Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Scholar Score Predictor")
st.write("Predict student exam scores based on various performance factors.")

# -------------------------------
# Input Form (Centered & Narrow)
# -------------------------------
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
categorical_cols = X.select_dtypes(include="object").columns.tolist()

# Create layout with 3 columns (left spacer, center form, right spacer)
left, center, right = st.columns([1,2,1])

with center:
    with st.form(key="input_form"):
        st.subheader("📥 Enter Student Details")
        
        user_input = {}
        for i, col in enumerate(columns, start=1):
            if col in numeric_cols:
                min_val, max_val = int(X[col].min()), int(X[col].max())
                mean_val = int(X[col].mean())
                user_input[col] = st.slider(f"{i}. {col}", min_val, max_val, mean_val)
            elif col in categorical_cols:
                options = X[col].unique().tolist()
                user_input[col] = st.selectbox(f"{i}. {col}", options)

        submit_button = st.form_submit_button(label="Predict")

# -------------------------------
# Make Predictions
# -------------------------------
if submit_button:
    input_df = pd.DataFrame([user_input])

    # Encode categorical inputs
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = pd.factorize(input_df[col])[0]

    # Reorder columns to match training
    input_df = input_df[columns]

    linear_pred = linear_model.predict(input_df)[0]
    poly_pred = poly_model.predict(poly_transformer.transform(input_df))[0]

    # Center the prediction results too
    left, center, right = st.columns([1,2,1])
    with center:
        #st.subheader("You will get:")
        #st.metric("",f"{linear_pred:.2f}")
        st.write(f"✅ In the end, you will get the score around *{linear_pred:.2f}*")
        #st.metric("Polynomial Regression Prediction", f"{poly_pred:.2f}")






