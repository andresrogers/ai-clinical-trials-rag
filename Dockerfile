FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy source code
COPY src/ ./src/
COPY app/ ./app/

# Expose ports
EXPOSE 8000 8501

# Default command (can override)
CMD ["uvicorn", "biotech_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
