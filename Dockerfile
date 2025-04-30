# Use an official lightweight Python image
FROM python:3.13.1-slim

# Set working directory in container
WORKDIR /app

# Copy your files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Define the command to run your app
CMD ["python", "main.py"]