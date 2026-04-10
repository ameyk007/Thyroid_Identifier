import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL
model = joblib.load("thyroid.joblib")

st.title("🧠 Thyroid Prediction")

age = st.number_input("Age", 1, 120)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
adenopathy = st.selectbox("Adenopathy", ["Yes", "No"])
focality = st.selectbox("Focality", ["Uni-Focal", "Multi-Focal"])
stage = st.selectbox("Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])

if st.button("Predict"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Smoking": [smoking],
        "Adenopathy": [adenopathy],
        "Focality": [focality],
        "Stage": [stage]
    })

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Recurrence (Probability: {prob:.2f})")
    else:
        st.success(f"✅ No Recurrence (Probability: {prob:.2f})")
