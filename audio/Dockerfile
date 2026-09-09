FROM python:3.11-slim

RUN set -eux; \
    packages="curl ffmpeg libsndfile1"; \
    for attempt in 1 2 3; do \
        rm -rf /var/lib/apt/lists/*; \
        if apt-get -o Acquire::Retries=3 update \
            && apt-get -o Acquire::Retries=3 install -y --no-install-recommends $packages; then \
            break; \
        fi; \
        if [ "$attempt" -eq 3 ]; then exit 1; fi; \
        sleep 5; \
    done; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY shared/requirements.txt /app/shared_req.txt
RUN pip install --no-cache-dir -r /app/shared_req.txt

RUN pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    piper-tts==1.5.0 \
    soundfile \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

COPY audio/app/ /app/app/

ENV PYTHONPATH=/app \
    PDC_FRONT=audio \
    PDC_GPU_MODE=cpu

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=10 \
    CMD curl -sf http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
