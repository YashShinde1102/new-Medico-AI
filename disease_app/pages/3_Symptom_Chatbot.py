import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Symptom Chatbot", layout="centered")

st.markdown("## Symptom-Based Disease Chatbot")

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_file(filename):
    if os.path.exists(filename):
        return filename
    fallback = os.path.join(ROOT_DIR, filename)
    if os.path.exists(fallback):
        return fallback
    return filename

@st.cache_resource
def load_assets():
    with open(find_file("df_symptoms.pkl"), "rb") as f1: df = pickle.load(f1)
    with open(find_file("vectorizer.pkl"), "rb") as f2: vectorizer = pickle.load(f2)
    with open(find_file("X.pkl"), "rb") as f3: X = pickle.load(f3)
    return df, vectorizer, X

df_symptoms, vectorizer, X = load_assets()
desc = pd.read_csv(find_file("symptom_Description.csv"))
prec = pd.read_csv(find_file("cleaned_precautions.csv"))

# Normalize disease names in lookup tables
desc["Disease"] = desc["Disease"].astype(str).str.lower().str.strip()
prec["Disease"] = prec["Disease"].astype(str).str.lower().str.strip()

prec_group = prec.groupby("Disease")["Precaution"].apply(
    lambda x: "; ".join(sorted(set(x.dropna().astype(str))))
).reset_index()
prec_group["Disease"] = prec_group["Disease"].astype(str).str.lower().str.strip()


def predict_diseases(query, top_k=3):
    if not query.strip():
        return "Please describe your symptoms."

    vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(vec, X)[0]
    # Get top_k * 3 indices to allow for filtering of duplicates/low scores
    idxs = np.argsort(sims)[::-1]

    seen, results = set(), []
    for i in idxs:
        if len(results) == top_k: 
            break
        
        disease = str(df_symptoms.iloc[i]["Disease"]).strip().lower()
        
        if not disease or disease in seen:
            continue
        
        seen.add(disease)
        score = sims[i]

        # Skip if score is too low
        if score <= 0:
             continue 

        #Description lookup
        desc_text = desc.loc[desc["Disease"]==disease, "Description"]
        description = desc_text.iloc[0] if not desc_text.empty else "No description."
        
        #Precautions lookup
        pre_text = prec_group.loc[prec_group["Disease"]==disease, "Precaution"]
        precautions = pre_text.iloc[0] if not pre_text.empty else "No precautions."

      
        
        results.append(f"Disease: {disease.title()} (Score: {score:.3f})\n"
                        f"Description: {description}\n"
                        f"Precautions: {precautions}\n")
        

    return "\n\n".join(results) if results else "No matches found."

query = st.text_area("Describe your symptoms (e.g., 'fever, cough, fatigue'):")
top_k = st.slider("Top K results", 1, 5, 3)

if st.button(" Predict Disease"):
    with st.spinner("Analyzing..."):
        output = predict_diseases(query, top_k)
        st.text_area("Predicted Diseases", value=output, height=300)