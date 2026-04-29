# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements and setup files
COPY setup.py .
COPY environment.yml .

# Install dependencies using pip (simpler for Docker than conda usually)
# We extract the pip dependencies or just use the setup.py
RUN pip install --no-cache-dir .

# Copy the rest of the application code
COPY . .

# Set the entry point
ENTRYPOINT ["python", "main.py"]
