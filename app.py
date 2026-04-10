import streamlit as st
import pandas as pd
import pickle

# ========================
# LOAD MODELS
# ========================
with open("thyroid_01.pkl", "rb") as f:
    models = pickle.load(f)

# Load accuracy if available
try:
    with open("accuracy.pkl", "rb") as f:
        accuracy = pickle.load(f)
except:
    accuracy = {name: 0.0 for name in models.keys()}

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="Thyroid Recurrence Predictor",
    page_icon="🧠",
    layout="wide"
)

# ========================
# CUSTOM CSS (UI IMPROVEMENT)
# ========================
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:bold;
        text-align:center;
        color:#4CAF50;
    }
    .sub-text {
        text-align:center;
        color:gray;
    }
    .result-box {
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# TITLE
# ========================
st.markdown('<p class="main-title">🧠 Thyroid Cancer Recurrence Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-powered clinical prediction system</p>', unsafe_allow_html=True)

# ========================
# SIDEBAR
# ========================
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))

st.sidebar.markdown(f"📊 Accuracy: **{accuracy.get(model_choice,0)*100:.2f}%**")

# Best model highlight
best_model = max(accuracy, key=accuracy.get)
st.sidebar.markdown(f"🏆 Best Model: **{best_model}**")

# ========================
# INPUT SECTION
# ========================
st.markdown("### 📝 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    adenopathy = st.selectbox("Adenopathy", ["Yes", "No"])

with col3:
    focality = st.selectbox("Focality", ["Uni-Focal", "Multi-Focal"])
    stage = st.selectbox("Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])

# ========================
# CREATE INPUT DATAFRAME
# ========================
def create_input():
    return pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Smoking": [smoking],
        "Adenopathy": [adenopathy],
        "Focality": [focality],
        "Stage": [stage]
    })

# ========================
# PREDICTION
# ========================
st.markdown("---")

if st.button("🔍 Predict", use_container_width=True):

    try:
        input_data = create_input()
        model = models[model_choice]

        prediction = model.predict(input_data)[0]

        try:
            prob = model.predict_proba(input_data)[0][1]
        except:
            prob = None

        st.markdown("## 📊 Prediction Result")

        # RESULT DISPLAY
        if prediction == 1:
            st.markdown(
                "<div class='result-box' style='background-color:#ff4d4d;color:white;'>⚠️ High Risk of Recurrence</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box' style='background-color:#4CAF50;color:white;'>✅ Low Risk (No Recurrence)</div>",
                unsafe_allow_html=True
            )

        # PROBABILITY
        if prob is not None:
            st.markdown("### 📈 Confidence Score")
            st.progress(float(prob))
            st.write(f"**Probability of Recurrence:** {prob:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.markdown("💡 *This tool assists clinical decision-making. Not a substitute for medical advice.*")
