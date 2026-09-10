from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def generate_report(name, age, phone, email, address, blood_group,
                    symptoms, predicted_disease, precautions):
    """Generate PDF report for a patient"""
    folder = "patient_reports"
    os.makedirs(folder, exist_ok=True)

    filename = f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(folder, filename)

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 80, " AI-Powered Medical Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 120, f"Name: {name}")
    c.drawString(50, height - 140, f"Age: {age}")
    c.drawString(50, height - 160, f"Phone: {phone}")
    c.drawString(50, height - 180, f"Email: {email}")
    c.drawString(50, height - 200, f"Address: {address}")
    c.drawString(50, height - 220, f"Blood Group: {blood_group}")

    # Symptoms
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 260, "Symptoms:")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 280, symptoms)

    # Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 320, "Predicted Disease:")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 340, predicted_disease)

    # Precautions
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 380, "Recommended Precautions:")
    c.setFont("Helvetica", 12)
    text_obj = c.beginText(70, height - 400)
    for line in precautions.split('. '):
        text_obj.textLine(line.strip() + '.')
    c.drawText(text_obj)

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawRightString(width - 50, 50, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.save()
    return path
