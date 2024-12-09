FROM python:3.9-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Create upload directory with proper permissions
RUN mkdir -p uploaded_docs && \
    chmod 777 uploaded_docs

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Ensure the upload directory has proper permissions
RUN chmod 777 uploaded_docs

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]