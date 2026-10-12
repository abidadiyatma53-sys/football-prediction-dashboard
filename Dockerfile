# ==============================================================================
# Production Multi-Stage Dockerfile for Football Analytics & Prediction Engine
# Supports FastAPI Backend (Port 8000) and Streamlit Dashboard (Port 8501)
# ==============================================================================

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install required system runtime libraries (libgomp1 is required for XGBoost OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure data directory exists with write permissions for SQLite persistence
RUN mkdir -p /app/data && chmod 777 /app/data

# Create non-root system user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default health check on FastAPI docs
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Default command launches FastAPI backend
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
