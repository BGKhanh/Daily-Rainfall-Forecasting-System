# Pure Python 3.12 CPU serving image (decoupled from Cloudflare Workers frontend SPA)
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

# Create non-root runtime user and writable volume directory for Turso SQLite replica
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /var/lib/rainfall && \
    chown appuser:appuser /var/lib/rainfall

# Copy minimal runtime source code, configuration, model artifacts, and benchmark leaderboard
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser artifacts/models/ /app/artifacts/models/
RUN test "$(wc -c < /app/artifacts/models/demo_rainfall_vintage.db)" -gt 102400 || \
    (echo "ERROR: demo_rainfall_vintage.db appears to be an unhydrated Git LFS pointer; pull LFS or run ensure_demo_db.py before docker build." >&2 && exit 1)
COPY --chown=appuser:appuser reports/test_benchmark/ /app/reports/test_benchmark/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
