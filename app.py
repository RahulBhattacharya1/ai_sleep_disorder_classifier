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
        bp_sys = st.number_input("Systolic BP", 70, 250, 120)
        bp_dia = st.number_input("Diastolic BP", 40, 150, 80)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build input using TRAINING FEATURE NAMES
    row = {
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": float(sleep_duration),
        "Quality of Sleep": int(quality_of_sleep),
        "Physical Activity Level": int(physical_activity),
        "Stress Level": int(stress_level),
        "BMI Category": bmi_category,
        "Heart Rate": int(heart_rate),
        "Daily Steps": int(daily_steps),
        "BP_Systolic": float(bp_sys),
        "BP_Diastolic": float(bp_dia),
    }

    input_df = pd.DataFrame([row])

    # Reindex to exact training columns (add missing as NaN, drop extras)
    input_df = input_df.reindex(columns=FEATURE_COLUMNS)

    # Optional: show diagnostic if names differ
    missing = [c for c in FEATURE_COLUMNS if c not in row]
    extra = [c for c in row if c not in FEATURE_COLUMNS]
    if missing or extra:
        st.warning(f"Adjusted columns. Missing: {missing} | Extra: {extra}")

    try:
        pred = pipe.predict(input_df)[0]
        st.success(f"Predicted Sleep Disorder: {pred}")

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            classes = pipe.named_steps["model"].classes_
            probs = pipe.predict_proba(input_df)[0]
            st.write("Probabilities:")
            st.dataframe(pd.DataFrame({"Class": classes, "Probability": probs}).set_index("Class"))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
