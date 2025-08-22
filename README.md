# 🎓 Student Performance Prediction

This project predicts student exam scores based on various academic, social, and economic factors.  
It explores the relationships between features such as **study hours, tutoring sessions, parental involvement, teacher quality, and more** to understand their influence on exam performance.  

## 📌 Project Overview
- Performed **data cleaning & preprocessing** (handling missing values, encoding categorical data, outlier treatment).  
- Conducted **Exploratory Data Analysis (EDA)** with boxplots, histograms, scatterplots, and correlation heatmaps.  
- Built and evaluated **Linear Regression** and **Polynomial Regression** models.  
- Compared model performance using **MAE, RMSE, and R² score**.  

## 🛠️ Tech Stack
- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  

## 📂 Dataset
The dataset used: **StudentPerformanceFactors.csv**  
It contains features like:  
- `Hours_Studied`  
- `Tutoring_Sessions`  
- `Parental_Involvement`  
- `Access_to_Resources`  
- `Teacher_Quality`  
- `Motivation_Level`  
- `Family_Income`  
- `Exam_Score` (Target Variable)  
... and more.  

## 📊 Exploratory Data Analysis
- Checked for missing values and duplicates.  
- Handled outliers using **IQR capping**.  
- Encoded categorical variables (ordinal & one-hot encoding).  
- Visualized feature relationships and correlations.  

## 🤖 Models Implemented
1. **Linear Regression**  
   - Baseline model to predict exam scores.  
2. **Polynomial Regression (Degree = 2)**  
   - Captures non-linear relationships for improved accuracy.  

## 📈 Model Evaluation
Metrics used:  
- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)  
- **R² Score** (Coefficient of Determination)  

## 📷 Visualizations
- Distribution of exam scores.  
- Exam scores vs study hours.  
- Actual vs Predicted exam scores.  
- Residual error distribution.

## 🚀 Check It Out
Try the live app here 👉 [🎓 Scholar Score Predictor](https://your-app-link-here.streamlit.app)

## 🚀 Live Demo
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen?logo=streamlit)](https://your-app-link-here.streamlit.app)

Click the badge above to try the app!
