FROM python:3.9-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Remove or comment out these lines since healthcheck.sh doesn't exist
# COPY healthcheck.sh /healthcheck.sh
# RUN chmod +x /healthcheck.sh

# Replace the script-based healthcheck with a direct curl command
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005", "--log-level", "info"]







