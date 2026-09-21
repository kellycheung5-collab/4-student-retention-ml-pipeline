# Use an official lightweight Python runtime
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files, buffer stdout/stderr, and set default PORT
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (curl for HEALTHCHECK, build-essential for C extensions like SHAP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and model artifacts into the container
COPY api/ ./api/
COPY models/ ./models/

# Expose port (defaults to 8000, configurable by cloud host)
EXPOSE 8000

# Healthcheck to monitor container status dynamically using the PORT variable
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Command to run FastAPI via Uvicorn with dynamic port evaluation for Azure/cloud hosting
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]