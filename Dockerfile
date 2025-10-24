# Anant Enterprise Ray Server - Dockerfile
# Multi-stage build for production-ready container
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r anant && useradd -r -g anant -u 1001 anant

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY . .

# Set proper ownership
RUN chown -R anant:anant /app

# Switch to non-root user
USER anant

# Create directories for data persistence
RUN mkdir -p /app/data /app/logs /app/config

# Expose ports
# 8000: Anant Enterprise API
# 8265: Ray Dashboard
# 8080: Additional services
EXPOSE 8000 8265 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "ray_anant_cluster.py"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-asyncio black flake8 mypy

# Copy application code
COPY . .

# Set ownership for development
RUN chown -R anant:anant /app
USER anant

# Development command
CMD ["python", "ray_anant_cluster.py", "--dev"]

# Testing stage
FROM development as testing

# Run tests by default
CMD ["pytest", "tests/", "-v"]