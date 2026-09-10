import streamlit as st
import os

st.set_page_config(page_title="Medical AI System", layout="centered")

st.title("🏥 AI-Medico: Disease Diagnosis System")
st.markdown("""
Welcome to the **AI Medical Assistant** — choose a feature from the sidebar:
- 🩺 **Disease Predictor** using symptoms and ML  
- 🔬 **Cancer Image Detection** using Deep Learning & GradCAM  
- 💬 **Symptom-Based Chatbot** using NLP similarity  
- 📋 **Report Generation** of the patient             
""")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
cover_path = os.path.join(CURRENT_DIR, "cover_image.jpg")
if os.path.exists(cover_path):
    st.image(cover_path, caption="Your Health Companion", use_container_width=True)
