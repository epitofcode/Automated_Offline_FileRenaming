# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some python packages like chromadb/hnswlib)
RUN apt-get update && apt-get install -y 
    build-essential 
    curl 
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# WARNING FOR RENDER DEPLOYMENT:
# This Dockerfile runs the Python API. It expects Ollama to be running and accessible.
# If deploying to Render, you will need to point MODEL_URL or setup Ollama appropriately,
# as running a 5GB LLM locally inside a free-tier Render container will cause an Out-Of-Memory error.

# Start the FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
