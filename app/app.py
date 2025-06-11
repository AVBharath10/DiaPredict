import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="DiaPredict",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load the trained model
model = joblib.load("../model/diapredict_model.pkl")

# Title and description
st.title("DiaPredict 🩸")
st.write("**No labs? No problem.** Predict diabetes risk in 30s.")

# Input form
with st.form("user_inputs"):
    st.write("### 📝 Enter Health Metrics")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=100)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=30, max_value=180, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    submitted = st.form_submit_button("🔍 Predict Risk")

# Prediction
if submitted:
    input_data = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]],
                              columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                       "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    
    risk = model.predict_proba(input_data)[0][1]
    risk_percent = int(risk * 100)

    st.write("---")
    st.subheader("🧪 Prediction Result")
    st.write(f"### 🧬 Risk Score: **{risk_percent}%**")

    if risk_percent > 50:
        st.error("🚨 **High risk!** Please consult a doctor immediately.")
    else:
        st.success("✅ **Low risk.** Keep maintaining your healthy lifestyle!")
