import sqlite3

def create_connection():
    """Create or connect to SQLite database"""
    conn = sqlite3.connect("patients.db")
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
            report_path TEXT
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
    conn.close()
