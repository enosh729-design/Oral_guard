FROM python:3.10-slim

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Copy dependency manifest first (layer caching) ---
COPY requirements.txt /app/requirements.txt

# --- Install Python dependencies ---
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy application source ---
COPY src/ /app/src/
COPY api/ /app/api/

# --- Create output directories ---
RUN mkdir -p /app/outputs/gradcam \
             /app/src/detector/weights \
             /app/src/classifier/checkpoints \
             /app/mlflow

# --- Environment variables ---
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# --- Expose FastAPI port ---
EXPOSE 8000

# --- Health check ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --- Launch ---
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
