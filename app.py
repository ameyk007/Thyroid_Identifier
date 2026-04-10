import streamlit as st
import pandas as pd
import pickle

# Load model
with open("thyroid.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Thyroid Recurrence Predictor", layout="centered")

st.title("🧠 Thyroid Cancer Recurrence Prediction")

st.markdown("Enter patient details below:")

# ========================
# INPUTS
# ========================
age = st.text_input("Age")

gender = st.selectbox("Gender", ["Male", "Female"])

smoking = st.selectbox("Smoking", ["Yes", "No"])

adenopathy = st.selectbox("Adenopathy", ["Yes", "No"])

focality = st.selectbox("Focality", ["Uni-Focal", "Multi-Focal"])

stage = st.selectbox("Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])


# ========================
# PREPROCESSING FUNCTION
# ========================
def preprocess(age, gender, smoking, adenopathy, focality, stage):
    # Convert inputs into numeric (must match training encoding)
    
    age = int(age)

    gender = 1 if gender == "Male" else 0
    smoking = 1 if smoking == "Yes" else 0
    adenopathy = 1 if adenopathy == "Yes" else 0
    focality = 1 if focality == "Multi-Focal" else 0

    stage_map = {
        "Stage I": 1,
        "Stage II": 2,
        "Stage III": 3,
        "Stage IV": 4
    }

    stage = stage_map[stage]

    data = pd.DataFrame([[age, gender, smoking, adenopathy, focality, stage]],
                        columns=["Age", "Gender", "Smoking", "Adenopathy", "Focality", "Stage"])

    return data


# ========================
# PREDICTION
# ========================
if st.button("Predict"):

    try:
        input_data = preprocess(age, gender, smoking, adenopathy, focality, stage)

        prediction = model.predict(input_data)[0]

        # Probability (if supported)
        try:
            prob = model.predict_proba(input_data)[0][1]
        except:
            prob = None

        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.error("⚠️ Recurrence: YES")
        else:
            st.success("✅ Recurrence: NO")

        if prob is not None:
            st.info(f"📊 Probability of Recurrence: {prob:.2f}")

    except Exception as e:
        st.warning(f"⚠️ Error: {e}")
