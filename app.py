import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("thyroid.pkl", "rb"))

# -------------------------------
# Title
# -------------------------------
st.set_page_config(page_title="Thyroid Recurrence Prediction", layout="wide")

st.title("🧠 Thyroid Cancer Recurrence Prediction System")
st.markdown("Predict recurrence risk using clinical inputs")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"])
adenopathy = st.sidebar.selectbox("Adenopathy", ["Yes", "No"])
focality = st.sidebar.selectbox("Focality", ["Uni-Focal", "Multi-Focal"])
stage = st.sidebar.selectbox("Stage", ["I", "II", "III", "IV"])

# -------------------------------
# Encoding Function
# -------------------------------
def encode_input():
    return pd.DataFrame({
        "Age": [age],
        "Gender": [1 if gender == "Male" else 0],
        "Smoking": [1 if smoking == "Yes" else 0],
        "Adenopathy": [1 if adenopathy == "Yes" else 0],
        "Focality": [1 if focality == "Multi-Focal" else 0],
        "Stage": [
            {"I": 1, "II": 2, "III": 3, "IV": 4}[stage]
        ]
    })

# -------------------------------
# Prediction Button
# -------------------------------
if st.sidebar.button("Predict"):

    input_df = encode_input()

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # -------------------------------
    # Main Output
    # -------------------------------
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("Prediction: HIGH RISK (Recurrence Likely)")
    else:
        st.success("Prediction: LOW RISK (No Recurrence)")

    st.metric("Recurrence Probability", f"{round(probability*100,2)}%")

    # -------------------------------
    # Probability Bar Chart
    # -------------------------------
    st.subheader("📈 Probability Distribution")

    probs = [1 - probability, probability]
    labels = ["No Recurrence", "Recurrence"]

    fig, ax = plt.subplots()
    ax.bar(labels, probs)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # -------------------------------
    # Feature Importance (if available)
    # -------------------------------
    st.subheader("🔍 Key Factors (Feature Importance)")

    try:
        importance = model.feature_importances_
        features = input_df.columns

        fig2, ax2 = plt.subplots()
        ax2.barh(features, importance)
        ax2.set_xlabel("Importance")
        st.pyplot(fig2)

        top_features = features[np.argsort(importance)[-2:]]
        st.write("Top Influencing Factors:", list(top_features))

    except:
        st.info("Feature importance not available for this model")

    # -------------------------------
    # Risk Gauge (Simple)
    # -------------------------------
    st.subheader("🎯 Risk Gauge")

    if probability > 0.7:
        st.error("🔴 High Risk")
    elif probability > 0.4:
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    # -------------------------------
    # Counterfactual Suggestions
    # -------------------------------
    st.subheader("💡 Suggestions")

    suggestions = []

    if stage in ["III", "IV"]:
        suggestions.append("If stage was lower, recurrence risk may decrease")

    if adenopathy == "Yes":
        suggestions.append("Absence of adenopathy reduces risk")

    if smoking == "Yes":
        suggestions.append("Avoid smoking to improve prognosis")

    if len(suggestions) == 0:
        suggestions.append("Maintain current health monitoring")

    for s in suggestions:
        st.write("👉", s)

# -------------------------------
# Accuracy Graph (Dummy Example)
# -------------------------------
st.subheader("📊 Model Accuracy Comparison")

models = ["Logistic", "RandomForest", "DecisionTree", "XGBoost"]
accuracy = [0.82, 0.89, 0.85, 0.91]  # replace with real scores

fig3, ax3 = plt.subplots()
ax3.bar(models, accuracy)
ax3.set_ylabel("Accuracy")
st.pyplot(fig3)
