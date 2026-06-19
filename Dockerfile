FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV
# NOTE: libgl1-mesa-glx was removed in Debian Trixie — use libgl1 instead
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .
COPY src/ ./src/

# Copy entrypoint and make executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Streamlit port (HF Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Use entrypoint to write secrets.toml from env vars then start Streamlit
ENTRYPOINT ["/entrypoint.sh"]
