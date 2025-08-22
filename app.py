import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------
# Load or Train Models
# -----------------------------
def load_or_train_models():
    if os.path.exists("linear_model.pkl") and os.path.exists("poly_model.pkl") and os.path.exists("poly_transformer.pkl"):
        # Load existing models
        with open("linear_model.pkl", "rb") as f:
            linear_model = pickle.load(f)
        with open("poly_model.pkl", "rb") as f:
            poly_model = pickle.load(f)
        with open("poly_transformer.pkl", "rb") as f:
            poly_transformer = pickle.load(f)
    else:
        # Retrain if files not found
        df = pd.read_csv("StudentPerformanceFactors.csv")
        X = df.drop(columns="Exam_Score")
        y = df["Exam_Score"]

        # Linear regression
        linear_model = LinearRegression().fit(X, y)

        # Polynomial regression
        poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
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

linear_model, poly_model, poly_transformer = load_or_train_models()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎓 Scholar Score Predictor")
st.write("Predict student exam performance using Linear or Polynomial Regression.")

# Collect user input
st.sidebar.header("Input Features")
hours = st.sidebar.slider("Study Hours", 0, 20, 5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
parental = st.sidebar.selectbox("Parental Involvement", [0, 1])
resources = st.sidebar.selectbox("Access to Resources", [0, 1])
extracurricular = st.sidebar.selectbox("Extracurricular Activities", [0, 1])
sleep = st.sidebar.slider("Sleep Hours", 0, 12, 7)
prev_scores = st.sidebar.slider("Previous Scores", 0, 100, 75)
motivation = st.sidebar.slider("Motivation Level", 0, 10, 5)
internet = st.sidebar.selectbox("Internet Access", [0, 1])

# Create dataframe for prediction
features = pd.DataFrame([[
    hours, attendance, parental, resources, extracurricular,
    sleep, prev_scores, motivation, internet
]], columns=[
    "Hours_Studied", "Attendance", "Parental_Involvement",
    "Access_to_Resources", "Extracurricular_Activities",
    "Sleep_Hours", "Previous_Scores", "Motivation_Level", "Internet_Access"
])

# Model selection
model_choice = st.radio("Choose a model:", ["Linear Regression", "Polynomial Regression"])

if model_choice == "Linear Regression":
    prediction = linear_model.predict(features)[0]
else:
    features_poly = poly_transformer.transform(features)
    prediction = poly_model.predict(features_poly)[0]

st.subheader("Predicted Exam Score")
st.write(f"📊 {prediction:.2f}")
