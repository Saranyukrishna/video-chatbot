# Use an official Python runtime as a base image
FROM python:3.11-slim

# Install FFmpeg and system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . /app/

# Expose the port the app runs on (default port for Streamlit is 8501)
EXPOSE 8501

# Command to run your app using Streamlit
CMD ["streamlit", "run", "youtube.py"]
