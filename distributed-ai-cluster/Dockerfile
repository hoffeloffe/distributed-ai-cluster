# Multi-stage Dockerfile for Distributed AI Cluster components
FROM python:3.9-slim AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for AI workloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r aiuser && useradd -r -g aiuser aiuser

FROM base AS dependencies

WORKDIR /deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS runtime

# Copy Python dependencies installed in the dependencies stage
COPY --from=dependencies /usr/local /usr/local

WORKDIR /app

# Copy project files
COPY src/ ./src/
COPY config/ ./config/

# Optional models directory (empty in repo) to avoid COPY failure
COPY models/ ./models/

# Ensure directories exist and set permissions
RUN mkdir -p /app/logs /app/cache && \
    chown -R aiuser:aiuser /app

USER aiuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -m src.healthcheck || exit 1

CMD ["python3", "-m", "src.entrypoint"]
