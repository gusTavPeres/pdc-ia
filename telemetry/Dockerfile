FROM python:3.11-slim

WORKDIR /app

COPY shared/__init__.py /app/shared/__init__.py
COPY shared/device_telemetry.py /app/shared/device_telemetry.py
COPY telemetry/ /app/telemetry/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PDC_HOST_SYS_ROOT=/host/sys \
    PDC_HOST_PROC_ROOT=/host/proc \
    PDC_HOST_TELEMETRY_PATH=/app/data/telemetry/host.json \
    PDC_TELEMETRY_PROVIDER_DIR=/app/data/telemetry/providers

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=4 \
    CMD ["python", "-m", "telemetry.agent", "--check", "/app/data/telemetry/host.json"]

CMD ["python", "-m", "telemetry.agent"]
