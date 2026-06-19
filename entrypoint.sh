#!/bin/bash
# Write Streamlit secrets from environment variables before app starts
mkdir -p /app/.streamlit

cat > /app/.streamlit/secrets.toml << EOF
APP_PASSWORD = "${APP_PASSWORD}"
EOF

echo "secrets.toml written to /app/.streamlit/secrets.toml"

# Start Streamlit
exec streamlit run app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
