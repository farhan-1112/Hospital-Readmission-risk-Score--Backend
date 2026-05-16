# Use official Python 3.11 runtime as base image
FROM python:3.11-slim

# Set environment variables for Python performance and Docker stability
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=10000

# Set working directory
WORKDIR /app

# Install system dependencies required for ML libraries (like XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the ML models explicitly (documented for clarity)
# This includes model.pkl, scaler.pkl, and encoder.pkl
COPY models/ ./models/

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 10000

# Health check to ensure the service is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Start the FastAPI application using uvicorn
# Using --host 0.0.0.0 is critical for container accessibility
# Using the PORT environment variable for Render compatibility
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
