FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY shared/requirements.txt /app/shared_req.txt
RUN pip install --no-cache-dir -r /app/shared_req.txt

RUN pip install --no-cache-dir \
    faster-whisper==1.0.3 \
    soundfile \
    numpy \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

RUN pip install --no-cache-dir piper-tts pathvalidate

COPY audio/app/ /app/app/

ENV PYTHONPATH=/app
ENV PDC_FRONT=audio
ENV PDC_GPU_MODE=auto

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=10 \
    CMD curl -sf http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
