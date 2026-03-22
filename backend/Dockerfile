# --- Builder Stage ---
FROM python:3.10-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# --- Final Stage ---
FROM python:3.10-slim

WORKDIR /app

# Copy the pre-built wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install packages using the pre-built wheels
RUN pip install --no-cache-dir /wheels/* --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# Ensure non-root execution for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
