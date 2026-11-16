# Dockerfile for DS108 Rainfall Forecasting Serving Engine
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal runtime system dependencies (OpenMP for LightGBM/XGBoost and curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned pure-serving Python dependencies
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Create non-root runtime user
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

# Copy minimal runtime source code, model artifacts, and test benchmark leaderboard
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser artifacts/models/ /app/artifacts/models/
COPY --chown=appuser:appuser reports/test_benchmark/ /app/reports/test_benchmark/

USER appuser

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI app with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

