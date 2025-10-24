# FastAPI Application Dockerfile
# ==============================

FROM rayproject/ray:2.31.0-py39

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY ../requirements.txt ./ray_requirements.txt

# Install dependencies with latest versions
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ray_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs

# Set environment variables for latest versions
ENV PYTHONPATH=/app
ENV RAY_ADDRESS=ray-head:6379
ENV RAY_DISABLE_IMPORT_WARNING=1
ENV RAY_LOG_LEVEL=INFO

# Expose the FastAPI port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command with modern FastAPI features
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]