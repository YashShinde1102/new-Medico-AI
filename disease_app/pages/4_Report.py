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

from database import create_table, insert_patient, get_all_patients, delete_patient
from report_generator import generate_report

st.set_page_config(page_title="Patient Management & Reports", layout="wide")

st.title("📋 Patient Registration, Diagnosis & Clinical Records")

# Ensure database is initialized
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
df["prognosis"] = df["prognosis"].map(mapping).astype(int)
tr["prognosis"] = tr["prognosis"].map(mapping).astype(int)

X = df[symptoms]
y = df["prognosis"].to_numpy(dtype=np.int32)
X_test = tr[symptoms]
y_test = tr["prognosis"].to_numpy(dtype=np.int32)

# Train model (cached for performance)
@st.cache_resource
def get_report_model():
    m = RandomForestClassifier().fit(X, y)
    acc = accuracy_score(y_test, m.predict(X_test))
    return m, acc

model, accuracy = get_report_model()

# ==============================================================================
# Two-Tab Navigation: 1. Registration & Diagnosis | 2. Patient Database History
# ==============================================================================
tab_register, tab_database = st.tabs([
    "📝 New Patient Registration & Diagnosis",
    "📊 Database Records & Patient History"
])

# ------------------------------------------------------------------------------
# TAB 1: New Patient Registration & Diagnosis
# ------------------------------------------------------------------------------
with tab_register:
    st.markdown("### Register Patient & Generate Aesthetic Clinical Report")
    st.caption("Fill in the patient's demographics and observed symptoms to generate a diagnosis and formatted medical PDF report.")

    with st.form("report_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 Personal Details")
            name = st.text_input("Full Name *", placeholder="e.g. Yash Shinde")
            age = st.number_input("Age *", min_value=1, max_value=120, value=25, step=1)
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            phone = st.text_input("Phone Number *", placeholder="+91 9876543210")

        with col2:
            st.subheader("🏥 Contact & Symptoms")
            email = st.text_input("Email Address *", placeholder="patient@example.com")
            address = st.text_input("Address *", placeholder="City, State, Country")
            symptoms_input = st.text_area(
                "Observed Symptoms (comma separated) *",
                placeholder="e.g. itching, skin_rash, nodal_skin_eruptions, fatigue"
            )

        submitted = st.form_submit_button("🚀 Submit, Predict & Generate Clinical Report", use_container_width=True)

    if submitted:
        if not (name.strip() and phone.strip() and email.strip() and address.strip() and symptoms_input.strip()):
            st.error("⚠️ Please fill in all mandatory fields before submitting.")
        else:
            # One-hot encode symptom input
            symptom_list = [s.strip().lower().replace(" ", "_") for s in symptoms_input.split(",") if s.strip()]
            input_vector = [1 if s in symptom_list else 0 for s in symptoms]
            input_data = [input_vector]

            # Predict disease
            pred_index = model.predict(input_data)[0]
            predicted_disease = diseases[int(pred_index)]

            # Load precaution data
            precaution_dict = {}
            try:
                precaution_df = pd.read_csv(find_file("cleaned_precautions.csv"))
                for _, row in precaution_df.iterrows():
                    precaution_dict[row["Disease"].strip()] = row["Precaution"].strip()
            except FileNotFoundError:
                st.warning("⚠️ 'cleaned_precautions.csv' not found. Using default precaution advice.")

            precautions = precaution_dict.get(predicted_disease, "Consult a registered medical specialist for dedicated follow-up care.")

            # Diagnostic summary card
            st.success(f"### 🎯 Predicted Diagnosis: **{predicted_disease}**")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.info(f"**Model Diagnostic Accuracy:** {accuracy*100:.2f}%")
            with c2:
                st.markdown(f"**Recommended Action:** {precautions}")

            with st.spinner("Generating aesthetic clinical PDF report..."):
                report_path = generate_report(
                    name=name,
                    age=age,
                    phone=phone,
                    email=email,
                    address=address,
                    blood_group=blood_group,
                    symptoms=", ".join(symptom_list),
                    predicted_disease=predicted_disease,
                    precautions=precautions
                )

                # Insert patient into SQLite database
                patient_id = insert_patient((
                    name.strip(), age, phone.strip(), email.strip(), address.strip(), blood_group,
                    ", ".join(symptom_list), predicted_disease, precautions, report_path
                ))

            st.balloons()
            st.success(f"✅ Record saved in database (Patient ID: **#{patient_id}**)! Medical Report generated successfully.")

            # Download button
            if os.path.exists(report_path):
                with open(report_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Clinical PDF Report",
                        data=pdf_file,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                        use_container_width=True
                    )


# ------------------------------------------------------------------------------
# TAB 2: Database Records & Patient History
# ------------------------------------------------------------------------------
with tab_database:
    st.markdown("### 🗄️ Registered Patient Database & Historical Reports")
    st.caption("All consultation records stored securely in SQLite (`patients.db`). Search, filter, or re-download any medical report.")

    patients_df = get_all_patients()

    if patients_df.empty:
        st.info("No patient records registered yet. Fill out the registration form in the first tab to add records!")
    else:
        # Metrics summary
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Patients Registered", len(patients_df))
        with m2:
            top_disease = patients_df["predicted_disease"].mode()[0] if not patients_df["predicted_disease"].empty else "N/A"
            st.metric("Most Frequent Diagnosis", top_disease)
        with m3:
            latest_name = patients_df.iloc[0]["name"] if not patients_df.empty else "N/A"
            st.metric("Latest Registered Patient", latest_name)

        st.markdown("---")

        # Search & Filter Controls
        fcol1, fcol2 = st.columns([2, 1])
        with fcol1:
            search_query = st.text_input("🔍 Search by Patient Name or Phone:", placeholder="Type to filter...")
        with fcol2:
            disease_filter = st.selectbox(
                "Filter by Diagnosis:",
                ["All Conditions"] + sorted(patients_df["predicted_disease"].unique().tolist())
            )

        # Apply filters
        filtered_df = patients_df.copy()
        if search_query.strip():
            query = search_query.strip().lower()
            filtered_df = filtered_df[
                filtered_df["name"].str.lower().str.contains(query, na=False) |
                filtered_df["phone"].str.lower().str.contains(query, na=False)
            ]
        if disease_filter != "All Conditions":
            filtered_df = filtered_df[filtered_df["predicted_disease"] == disease_filter]

        st.markdown(f"**Showing {len(filtered_df)} of {len(patients_df)} patient records:**")

        # Interactive Data Table
        display_columns = ["id", "name", "age", "blood_group", "predicted_disease", "phone", "email", "symptoms"]
        st.dataframe(
            filtered_df[display_columns].rename(columns={
                "id": "ID",
                "name": "Patient Name",
                "age": "Age",
                "blood_group": "Blood Group",
                "predicted_disease": "Diagnosed Condition",
                "phone": "Phone",
                "email": "Email",
                "symptoms": "Reported Symptoms"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Patient Details & Report Re-Download Section
        st.subheader("📄 View Details & Download Patient Report")
        selected_id = st.selectbox(
            "Select Patient to View / Download Report:",
            options=filtered_df["id"].tolist(),
            format_func=lambda pid: f"ID #{pid} - {filtered_df.loc[filtered_df['id'] == pid, 'name'].values[0]} ({filtered_df.loc[filtered_df['id'] == pid, 'predicted_disease'].values[0]})"
        )

        if selected_id:
            patient_row = filtered_df[filtered_df["id"] == selected_id].iloc[0]

            with st.container():
                st.markdown(f"""
                <div style="background-color: #0E1626; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                    <h4 style="margin: 0; color: #38BDF8;">👤 {patient_row['name']} (Age: {patient_row['age']} | Blood Group: {patient_row['blood_group']})</h4>
                    <p style="margin: 4px 0; color: #94A3B8;"><b>Phone:</b> {patient_row['phone']} | <b>Email:</b> {patient_row['email']} | <b>Address:</b> {patient_row['address']}</p>
                    <hr style="border-color: #1E293B; margin: 8px 0;"/>
                    <p style="margin: 4px 0;"><b>🎯 Diagnosed Disease:</b> <span style="color: #4ADE80; font-weight: bold;">{patient_row['predicted_disease']}</span></p>
                    <p style="margin: 4px 0;"><b>🩺 Symptoms:</b> {patient_row['symptoms']}</p>
                    <p style="margin: 4px 0;"><b>💊 Clinical Precautions:</b> {patient_row['precautions']}</p>
                </div>
                """, unsafe_allow_html=True)

                col_dl, col_del = st.columns([3, 1])

                with col_dl:
                    # Check if report exists or re-generate on the fly
                    pdf_file_path = patient_row["report_path"]
                    if not (pdf_file_path and os.path.exists(pdf_file_path)):
                        # Regenerate if missing
                        pdf_file_path = generate_report(
                            name=patient_row["name"],
                            age=patient_row["age"],
                            phone=patient_row["phone"],
                            email=patient_row["email"],
                            address=patient_row["address"],
                            blood_group=patient_row["blood_group"],
                            symptoms=patient_row["symptoms"],
                            predicted_disease=patient_row["predicted_disease"],
                            precautions=patient_row["precautions"]
                        )

                    with open(pdf_file_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download Medical Report for {patient_row['name']}",
                            data=f,
                            file_name=os.path.basename(pdf_file_path),
                            mime="application/pdf",
                            key=f"dl_{selected_id}",
                            use_container_width=True
                        )

                with col_del:
                    if st.button("🗑️ Delete Record", key=f"del_{selected_id}", use_container_width=True):
                        delete_patient(selected_id)
                        st.toast(f"Record #{selected_id} deleted.")
                        st.rerun()

        # CSV Database Export
        st.markdown("---")
        csv_data = patients_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Export Entire Database to CSV",
            data=csv_data,
            file_name="ai_medico_patient_database.csv",
            mime="text/csv"
        )
