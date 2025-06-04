# Use an official lightweight Python image
FROM python:3.13.1-slim

# Add OS-level dependencies for OpenCV and others
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR /app

# Copy and install dependencies separately to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your files into the container
COPY . .

# Ensure model weights file exists
RUN ls -la *.pt || echo "Warning: No .pt model files found"

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Define the command to run your app
CMD ["python", "main.py"]