import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path

st.set_page_config(page_title="Sleep Disorder Classifier", layout="centered")
st.title("Sleep Disorder Classifier")
st.caption("Predict Insomnia / Sleep Apnea / None")

MODEL_PATH = Path("models/sleep_disorder_pipeline.joblib")
COLS_PATH = Path("models/feature_columns.json")

# Hard stop if files missing
if not MODEL_PATH.exists():
    st.error("Missing model file: models/sleep_disorder_pipeline.joblib")
    st.stop()
if not COLS_PATH.exists():
    st.error("Missing columns file: models/feature_columns.json")
    st.stop()

pipe = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = json.loads(COLS_PATH.read_text())

# Simple choices (handle_unknown='ignore' allows unseen categories)
gender_choices = ["Male", "Female"]
bmi_choices = ["Underweight", "Normal", "Overweight", "Obese"]

# Occupation as free text to avoid mismatch; unseen OK due to handle_unknown='ignore'
with st.form("form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", gender_choices)
        occupation = st.text_input("Occupation", value="Software Engineer")
        sleep_duration = st.number_input("Sleep Duration (hours)", 3.0, 12.0, 6.5, 0.1)
        quality_of_sleep = st.slider("Quality of Sleep (1–10)", 1, 10, 7)
        physical_activity = st.slider("Physical Activity Level (0–100)", 0, 100, 50)
    with col2:
        stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
        bmi_category = st.selectbox("BMI Category", bmi_choices)
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 75)
        daily_steps = st.number_input("Daily Steps", 0, 50000, 6000, 100)
        bp_sys = st.number_input("Systoli
