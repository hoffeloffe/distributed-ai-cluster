# Multi-stage Dockerfile for Distributed AI Worker Node
FROM python:3.9-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r aiuser && useradd -r -g aiuser aiuser

FROM base as dependencies

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM dependencies as runtime

# Copy application code
WORKDIR /app
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache && \
    chown -R aiuser:aiuser /app

# Switch to non-root user
USER aiuser

# Expose ports
EXPOSE 8888 8889 8890 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Start the worker node
CMD ["python", "src/kubernetes_worker.py"]
