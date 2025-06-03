import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model\diapredict_model.pkl")

st.title("DiaPredict 🩸")
st.write("**No labs? No problem.** Predict diabetes risk in 30s.")

# Input form
with st.form("user_inputs"):
    st.write("### Enter Health Metrics")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=100)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=30, max_value=180, value=70)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    # Format input for the model
    input_data = pd.DataFrame([[pregnancies, glucose, bp, bmi]], 
                             columns=["Pregnancies", "Glucose", "BloodPressure", "BMI"])
    
    # Predict
    risk = model.predict_proba(input_data)[0][1]  # Probability of diabetes (0-1)
    risk_percent = int(risk * 100)
    
    # Show result
    st.write(f"### Risk Score: {risk_percent}%")
    if risk_percent > 50:
        st.error("🚨 High risk! Consult a doctor.")
    else:
        st.success("✅ Low risk. Maintain healthy habits!")