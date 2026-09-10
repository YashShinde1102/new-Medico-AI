# Use an official lightweight Python image
FROM python:3.10-bookworm

# Set working directory inside container
WORKDIR /app

# Copy all project files
COPY . /app

# Install system dependencies (needed for TensorFlow, OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8080

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Run Streamlit app
CMD ["streamlit", "run", "disease_app/Home.py", "--server.port=8080"]
