import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/sleep_disorder_pipeline.joblib")

st.title("AI Sleep Disorder Classifier")

# User input form
age = st.number_input("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
occupation = st.selectbox("Occupation", ["Doctor", "Engineer", "Sales Representative", "Nurse", "Teacher", "Software Engineer"])
sleep_duration = st.number_input("Sleep Duration (hours)", 3.0, 12.0, 6.0)
stress = st.slider("Stress Level", 1, 10, 5)
activity = st.slider("Physical Activity Level", 0, 100, 50)
heart_rate = st.number_input("Heart Rate", 40, 120, 75)
steps = st.number_input("Daily Steps", 1000, 20000, 5000)
bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])

# Encode input (simple manual mapping)
gender_map = {"Male": 1, "Female": 0}
bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
occupation_map = {name: i for i, name in enumerate(["Doctor", "Engineer", "Sales Representative", "Nurse", "Teacher", "Software Engineer"])}

input_df = pd.DataFrame([[
    age, gender_map[gender], occupation_map[occupation], sleep_duration,
    0,  # placeholder for Quality of Sleep (unused)
    activity, stress, bmi_map[bmi], 
    0,  # placeholder for BP (unused)
    heart_rate, steps
]], columns=[
    "Age", "Gender", "Occupation", "Sleep Duration", 
    "Quality of Sleep", "Physical Activity Level", "Stress Level", 
    "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps"
])

# Prediction
pred = model.predict(input_df)[0]
st.write("### Predicted Sleep Disorder:", pred)
