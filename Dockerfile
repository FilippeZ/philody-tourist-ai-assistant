# Multi-Stage Production Dockerfile for Philody AI Tourist Assistant
FROM python:3.11-slim as base

# Set environment variables for production performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create non-root user for security compliance (EU AI Act & SOC2)
RUN groupadd -r appgroup && useradd -r -g appgroup appuser \
    && chown -R appuser:appgroup /app

USER appuser

# Expose ports: 8000 for FastAPI REST Gateway, 8501 for Streamlit UI
EXPOSE 8000 8501

# Healthcheck for container orchestration (Kubernetes / Docker Swarm)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint: Production FastAPI REST Gateway via Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
