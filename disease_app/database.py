import sqlite3
import os
import pandas as pd

# Persistent database path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "patients.db")

def create_connection():
    """Create or connect to SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_table():
    """Create table for storing patient data"""
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            phone TEXT,
            email TEXT,
            address TEXT,
            blood_group TEXT,
            symptoms TEXT,
            predicted_disease TEXT,
            precautions TEXT,
            report_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_patient(data):
    """Insert new patient record into database"""
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients (
            name, age, phone, email, address, blood_group,
            symptoms, predicted_disease, precautions, report_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()
    patient_id = cursor.lastrowid
    conn.close()
    return patient_id

def get_all_patients():
    """Fetch all patient records as a pandas DataFrame"""
    create_table()
    conn = create_connection()
    query = """
        SELECT id, name, age, phone, email, address, blood_group,
               symptoms, predicted_disease, precautions, report_path
        FROM patients
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def delete_patient(patient_id):
    """Delete a patient record by ID"""
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()
