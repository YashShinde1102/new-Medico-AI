import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure disease_app and root directories are in Python path for reliable imports
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(APP_DIR)

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database import create_table, insert_patient
from report_generator import generate_report

st.set_page_config(page_title="Patient Registration", layout="centered")

st.title("📋 Smart Patient Registration and Disease Prediction")

create_table()

def find_file(filename):
    if os.path.exists(filename):
        return filename
    fallback = os.path.join(ROOT_DIR, filename)
    if os.path.exists(fallback):
        return fallback
    return filename

try:
    df = pd.read_csv(find_file("Training.csv"))
    tr = pd.read_csv(find_file("Testing (2).csv"))
except FileNotFoundError:
    st.error("⚠️ Dataset files not found. Please ensure 'Training.csv' and 'Testing (2).csv' are present.")
    st.stop()

# Clean dataset
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
tr = tr.loc[:, ~tr.columns.str.contains('^Unnamed')]

df["prognosis"] = df["prognosis"].astype(str).str.strip()
tr["prognosis"] = tr["prognosis"].astype(str).str.strip()
symptoms = [col for col in df.columns if col != "prognosis"]
diseases = sorted(df["prognosis"].unique())

mapping = {d: i for i, d in enumerate(diseases)}
df.replace({"prognosis": mapping}, inplace=True)
tr.replace({"prognosis": mapping}, inplace=True)

X, y = df[symptoms], np.ravel(df[["prognosis"]])
X_test, y_test = tr[symptoms], np.ravel(tr[["prognosis"]])

# Train model
model = RandomForestClassifier().fit(X, y)
accuracy = accuracy_score(y_test, model.predict(X_test))

with st.form("report_form"):
    st.subheader("👤 Patient Details")
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    phone = st.text_input("Phone Number")
    email = st.text_input("Email Address")
    address = st.text_area("Home Address")
    blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    symptoms_input = st.text_area("Describe Symptoms (comma separated)")

    submitted = st.form_submit_button("Submit & Generate Report")

if submitted:
    if not (name and age and phone and email and symptoms_input):
        st.error("⚠️ Please fill in all mandatory fields.")
        st.stop()

    # One-hot encode symptom input
    symptom_list = [s.strip().lower().replace(" ", "_") for s in symptoms_input.split(",")]
    input_vector = [1 if s in symptom_list else 0 for s in symptoms]
    input_data = [input_vector]

    # Predict
    pred_index = model.predict(input_data)[0]
    predicted_disease = diseases[int(pred_index)]

    # Load precaution data from CSV
    precaution_dict = {}
    try:
        precaution_df = pd.read_csv(find_file("cleaned_precautions.csv"))
        for _, row in precaution_df.iterrows():
            precaution_dict[row["Disease"].strip()] = row["Precaution"].strip()
    except FileNotFoundError:
        st.warning("⚠️ 'cleaned_precautions.csv' not found. Using default precaution message.")

    precautions = precaution_dict.get(predicted_disease, "Consult a doctor for detailed advice.")

    # Display results
    st.success(f"🎯 Predicted Disease: **{predicted_disease}**")
    st.info(f"📊 Model Accuracy: {accuracy*100:.2f}%")

    # Store report
    report_path = generate_report(
        name, age, phone, email, address, blood_group,
        ", ".join(symptom_list), predicted_disease, precautions
    )
    insert_patient((name, age, phone, email, address, blood_group,
                    ", ".join(symptom_list), predicted_disease, precautions, report_path))

    st.info(f"Report generated: {os.path.basename(report_path)}")
    with open(report_path, "rb") as f:
        st.download_button("📥 Download Report", f, file_name=os.path.basename(report_path))
