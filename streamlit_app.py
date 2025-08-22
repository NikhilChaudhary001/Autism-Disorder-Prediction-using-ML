import streamlit as st
import pickle
import numpy as np

# Load trained model and encoders
model = pickle.load(open("autism_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))  # dict of label encoders

st.title("🧩 Autism Spectrum Disorder (ASD) Prediction")
st.write("Answer the following questions to predict ASD likelihood.")

# === Screening Questions (A1–A10) ===
st.subheader("Screening Questions (0 = No, 1 = Yes)")
qs = []
questions = [
    "Does the child look at you when you call their name?",
    "Does the child make eye contact?",
    "Does the child point to indicate interest in objects?",
    "Does the child play pretend games (e.g., house, doctor)?",
    "Does the child use gestures (e.g., waving goodbye)?",
    "Does the child respond to your facial expressions/emotions?",
    "Does the child enjoy playing with other children?",
    "Does the child imitate your actions (e.g., clapping)?",
    "Does the child show interest in other children?",
    "Does the child try to get you to watch them?"
]
for i, q in enumerate(questions, start=1):
    qs.append(st.selectbox(f"A{i}_Score: {q}", [0, 1]))

# === Demographics & categorical ===
age = st.slider("Age", 1, 100, 18)
gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_)
jaundice = st.selectbox("Was the child jaundiced at birth?", ["Yes", "No"])
austim = st.selectbox("Has a family member been diagnosed with autism?", ["Yes", "No"])
country = st.selectbox("Country of Residence", encoders["contry_of_res"].classes_)
used_app_before = st.selectbox("Have you used this screening app before?", ["Yes", "No"])
result = st.selectbox("Screening test result (0 = Negative, 1 = Positive)", [0, 1])
relation = st.selectbox("Relation of respondent", encoders["relation"].classes_)

# === Encoding categorical values ===
gender = 1 if gender == "Male" else 0
jaundice = 1 if jaundice == "Yes" else 0
austim = 1 if austim == "Yes" else 0
used_app_before = 1 if used_app_before == "Yes" else 0
ethnicity = encoders["ethnicity"].transform([ethnicity])[0]
country = encoders["contry_of_res"].transform([country])[0]
relation = encoders["relation"].transform([relation])[0]

# === Feature vector in EXACT order ===
features = np.array([[ 
    qs[0], qs[1], qs[2], qs[3], qs[4],
    qs[5], qs[6], qs[7], qs[8], qs[9],
    age, gender, ethnicity, jaundice, austim,
    country, used_app_before, result, relation
]])

# === Threshold selection ===
threshold = st.slider("Select decision threshold (default = 0.5)", 0.0, 1.0, 0.5)

# === Prediction ===
if st.button("🔍 Predict"):
    prob = model.predict_proba(features)[0][1]   # probability of ASD = class 1
    st.write(f"🔎 Model Probability of ASD: **{prob:.2%}**")

    if prob >= threshold:
        st.error(f"⚠️ Likely ASD (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Unlikely ASD (Confidence: {prob:.2%})")
