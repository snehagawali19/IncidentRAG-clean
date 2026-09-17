FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY data/runbooks/ data/runbooks/
RUN pip install --no-cache-dir uv && uv pip install --system .

RUN useradd --create-home --uid 10001 incidentrag && chown -R incidentrag:incidentrag /app
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1
CMD ["uvicorn", "incidentrag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
