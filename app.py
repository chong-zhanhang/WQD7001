import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Define list of countries for UI convenience
countries = [
    'Australia', 'Brazil', 'Canada', 'China', 'Colombia', 'France', 'Germany',
    'India', 'Italy', 'Japan', 'New Zealand', 'Nigeria', 'South Africa',
    'South Korea', 'Spain', 'Thailand', 'United Kingdom', 'United States', 'Vietnam'
]

# Load selected model
@st.cache_resource
def load_model(model_name):
    return joblib.load(f"{model_name}.pkl")

# Generate user inputs
def user_input_features():
    st.sidebar.header("Patient Features")

    data = {
        'age': st.sidebar.slider("Age", 18, 100, 50),
        'sex': st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1]),
        'cholesterol': st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 200),
        'heart rate': st.sidebar.slider("Heart Rate (bpm)", 40, 200, 80),
        'diabetes': st.sidebar.selectbox("Diabetes (0/1)", [0, 1]),
        'family history': st.sidebar.selectbox("Family History (0/1)", [0, 1]),
        'smoking': st.sidebar.selectbox("Smoking (0/1)", [0, 1]),
        'obesity': st.sidebar.selectbox("Obesity (0/1)", [0, 1]),
        'alcohol consumption': st.sidebar.selectbox("Alcohol Consumption (0/1)", [0, 1]),
        'exercise hours per week': st.sidebar.slider("Exercise Hours/Week", 0.0, 20.0, 3.0),
        'diet': st.sidebar.selectbox("Healthy Diet (0/1)", [0, 1]),
        'previous heart problems': st.sidebar.selectbox("Previous Heart Problems (0/1)", [0, 1]),
        'medication use': st.sidebar.selectbox("Medication Use (0/1)", [0, 1]),
        'stress level': st.sidebar.slider("Stress Level (0-10)", 0, 10, 5),
        'sedentary hours per day': st.sidebar.slider("Sedentary Hours/Day", 0.0, 20.0, 8.0),
        'income': st.sidebar.slider("Income (USD/month)", 0, 20000, 3000),
        'bmi': st.sidebar.slider("BMI", 10.0, 50.0, 24.0),
        'triglycerides': st.sidebar.slider("Triglycerides (mg/dL)", 50, 600, 150),
        'physical activity days per week': st.sidebar.slider("Active Days/Week", 0.0, 7.0, 3.0),
        'sleep hours per day': st.sidebar.slider("Sleep Hours/Day", 0, 12, 7),
        'systolic pressure': st.sidebar.slider("Systolic BP", 90, 200, 120),
        'diastolic pressure': st.sidebar.slider("Diastolic BP", 60, 120, 80),
        'latitude': st.sidebar.slider("Latitude", -90.0, 90.0, 0.0),
        'longitude': st.sidebar.slider("Longitude", -180.0, 180.0, 0.0)
    }

    # One-hot encode countries
    selected_country = st.sidebar.selectbox("Country", countries)
    for country in countries:
        data[f'country_{country}'] = (country == selected_country)

    return pd.DataFrame([data])

# App UI
st.title("Early Heart Attack Detection")

# Load user input
input_df = user_input_features()

# Select model
model_options = {
    "Logistic Regression": "logistic_regression_model",
    "Decision Tree": "decision_tree_model",
    "Random Forest": "random_forest_model",
    "XGBoost": "xgboost_model"
}
model_choice = st.selectbox("Select Model", list(model_options.keys()))
model = load_model(model_options[model_choice])

# Prediction logic
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"**Heart Attack Risk Probability:** {prob:.4f}")
    st.write(f"**Predicted Class (1 = Risk):** {pred}")

    if pred == 1:
        st.error("⚠️ High Risk Detected – Recommend Medical Evaluation.")
    else:
        st.success("✅ Low Risk Detected – Continue Healthy Habits.")
