# Dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Set environment variables
ENV CACHE_DIR=/app/model_cache
ENV HF_HUB_CACHE=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create cache directory
RUN mkdir -p ${CACHE_DIR}

# Expose port
EXPOSE 8000

# Run with auto-reload in dev, remove --reload for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "300"]
