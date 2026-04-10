import streamlit as st
import pandas as pd
import pickle

# ========================
# LOAD MODELS
# ========================
with open("thyroid.pkl", "rb") as file:
    models = pickle.load(file)

# Dummy accuracy (replace with your real values)
model_accuracy = {
    "Logistic Regression": 0.85,
    "Random Forest": 0.92,
    "Decision Tree": 0.88,
    "XGBoost": 0.94
}

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="Thyroid Predictor", page_icon="🧠", layout="wide")

# ========================
# SIDEBAR
# ========================
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

st.sidebar.markdown(f"📊 Accuracy: **{model_accuracy[model_choice]*100:.2f}%**")

# ========================
# MAIN TITLE
# ========================
st.title("🧠 Thyroid Cancer Recurrence Predictor")
st.markdown("### Enter Patient Clinical Details")

# ========================
# INPUT LAYOUT (COLUMNS)
# ========================
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120)

    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    smoking = st.selectbox("Smoking", ["Yes", "No"])

    adenopathy = st.selectbox("Adenopathy", ["Yes", "No"])

with col3:
    focality = st.selectbox("Focality", ["Uni-Focal", "Multi-Focal"])

    stage = st.selectbox("Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])


# ========================
# PREPROCESS FUNCTION
# ========================
def preprocess(age, gender, smoking, adenopathy, focality, stage):
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

    return pd.DataFrame([[age, gender, smoking, adenopathy, focality, stage]],
                        columns=["Age", "Gender", "Smoking", "Adenopathy", "Focality", "Stage"])


# ========================
# PREDICT BUTTON
# ========================
st.markdown("---")

if st.button("🔍 Predict", use_container_width=True):

    input_data = preprocess(age, gender, smoking, adenopathy, focality, stage)

    model = models[model_choice]

    prediction = model.predict(input_data)[0]

    try:
        prob = model.predict_proba(input_data)[0][1]
    except:
        prob = None

    st.markdown("## 📊 Prediction Result")

    # Stylish Output
    if prediction == 1:
        st.markdown(
            "<h2 style='color:red;'>⚠️ High Risk of Recurrence</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h2 style='color:green;'>✅ Low Risk (No Recurrence)</h2>",
            unsafe_allow_html=True
        )

    if prob is not None:
        st.progress(float(prob))
        st.write(f"**Probability:** {prob:.2f}")

