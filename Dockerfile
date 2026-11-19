# Stage 1: Build Vite + React SPA static bundle
FROM node:24-alpine AS frontend-builder

WORKDIR /frontend

ARG VITE_API_BASE_URL=""
ARG VITE_COLD_START_THRESHOLD_MS="3000"
ARG VITE_REPO_URL="https://github.com/BGKhanh/Precipitation-EDA"
ARG VITE_CI_URL="https://github.com/BGKhanh/Precipitation-EDA/actions"
ARG VITE_MLFLOW_URL="https://dagshub.com/BGKhanh/Precipitation-EDA.mlflow"

ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_COLD_START_THRESHOLD_MS=$VITE_COLD_START_THRESHOLD_MS \
    VITE_REPO_URL=$VITE_REPO_URL \
    VITE_CI_URL=$VITE_CI_URL \
    VITE_MLFLOW_URL=$VITE_MLFLOW_URL

COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Minimal Python 3.12 CPU runtime (no Node.js in final image)
FROM python:3.12-slim AS runtime

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

# Copy minimal runtime source code, model artifacts, benchmark leaderboard, and built SPA static files
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser artifacts/models/ /app/artifacts/models/
RUN test "$(wc -c < /app/artifacts/models/demo_rainfall_vintage.db)" -gt 102400 || \
    (echo "ERROR: demo_rainfall_vintage.db appears to be an unhydrated Git LFS pointer; pull LFS or run ensure_demo_db.py before docker build." >&2 && exit 1)
COPY --chown=appuser:appuser reports/test_benchmark/ /app/reports/test_benchmark/
COPY --from=frontend-builder --chown=appuser:appuser /frontend/dist /app/frontend/dist

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
