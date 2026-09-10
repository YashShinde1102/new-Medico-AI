import streamlit as st
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from collections import Counter

st.set_page_config(page_title="Smart Disease Predictor", layout="centered")
st.title(" Symptoms Based Disease Predictor")
st.caption("A hybrid model combining machine learning and symptom similarity analysis")

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_file(filename):
    if os.path.exists(filename):
        return filename
    fallback = os.path.join(ROOT_DIR, filename)
    if os.path.exists(fallback):
        return fallback
    return filename

# Load datasets
try:
    df = pd.read_csv(find_file("Training.csv"))
    tr = pd.read_csv(find_file("Testing (2).csv"))
    disease_symptom_df = pd.read_csv(find_file("cleaned_symptoms_new.csv"))
except FileNotFoundError:
    st.error("Error: One or more dataset files not found. Ensure all CSVs are present.")
    st.stop()

# Preprocess data
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
tr = tr.loc[:, ~tr.columns.str.contains('^Unnamed')]
df["prognosis"] = df["prognosis"].astype(str).str.strip()
tr["prognosis"] = tr["prognosis"].astype(str).str.strip()

# Features and labels
symptoms = [col for col in df.columns if col != "prognosis"]
diseases = sorted(df["prognosis"].unique())
mapping = {d: i for i, d in enumerate(diseases)}

df["prognosis"] = df["prognosis"].map(mapping).astype(int)
tr["prognosis"] = tr["prognosis"].map(mapping).astype(int)

X = df[symptoms]
y = df["prognosis"].to_numpy(dtype=np.int32)
X_test = tr[symptoms]
y_test = tr["prognosis"].to_numpy(dtype=np.int32)

@st.cache_resource
def get_models():
    dt = tree.DecisionTreeClassifier().fit(X, y)
    rf = RandomForestClassifier().fit(X, y)
    nb = GaussianNB().fit(X, y)

    acc_dt = accuracy_score(y_test, dt.predict(X_test))
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_nb = accuracy_score(y_test, nb.predict(X_test))
    return dt, rf, nb, acc_dt, acc_rf, acc_nb

def prepare_input(symptoms_selected):
    temp = [0] * len(symptoms)
    for s in symptoms_selected:
        if s and s in symptoms:
            temp[symptoms.index(s)] = 1
    return [temp]

def predict_model(model, input_data):
    pred = model.predict(input_data)[0]
    try:
        return diseases[int(pred)]
    except (IndexError, TypeError):
        return "Unknown"

# Sidebar selections
st.sidebar.header("Select Symptoms")
symptom_inputs = [st.sidebar.selectbox(f"Symptom {i+1}", [""] + symptoms) for i in range(5)]

if st.sidebar.button("Predict Disease"):
    user_input = [s for s in symptom_inputs if s]
    
    if not user_input:
        st.warning("Please select at least one symptom.")
        st.stop()
    
    st.subheader("Model-Based Predictions")
    with st.spinner("Analyzing symptoms..."):
        dt, rf, nb, acc_dt, acc_rf, acc_nb = get_models()

        pred_dt = predict_model(dt, prepare_input(user_input))
        pred_rf = predict_model(rf, prepare_input(user_input))
        pred_nb = predict_model(nb, prepare_input(user_input))

        st.info(f" Decision Tree → {pred_dt} ({acc_dt*100:.2f}% accurate)")
        st.info(f" Random Forest → {pred_rf} ({acc_rf*100:.2f}% accurate)")
        st.info(f" Naive Bayes → {pred_nb} ({acc_nb*100:.2f}% accurate)")
    
    st.subheader(" Weighted Ensemble Result")

    # Combine model predictions with their accuracies as weights
    model_preds = [
        (pred_dt, acc_dt),
        (pred_rf, acc_rf),
        (pred_nb, acc_nb)
    ]

    # Weighted voting logic
    vote_counter = Counter()
    for pred, weight in model_preds:
        vote_counter[pred] += weight

    # Find the disease with the highest total weight
    final_disease, final_weight = vote_counter.most_common(1)[0]

    # Display result
    st.success(f" Final Predicted Disease: **{final_disease}** (Weighted Ensemble)")

    st.markdown("---")
