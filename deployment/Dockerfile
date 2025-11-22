# Multi-stage Dockerfile for Air Quality Prediction Platform

FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ../src ./src
COPY ../models ./models
COPY ../data ./data

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 5000 8501

# Default command (can be overridden)
CMD ["python", "src/api/app.py"]
