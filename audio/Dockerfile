FROM python:3.11-slim

# Dependências do sistema para áudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependências Python compartilhadas
COPY shared/requirements.txt /app/shared_req.txt
RUN pip install --no-cache-dir -r /app/shared_req.txt

# Dependências de áudio
RUN pip install --no-cache-dir \
    faster-whisper==1.0.3 \
    soundfile \
    numpy \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

# piper-tts e sua dependência pathvalidate
RUN pip install --no-cache-dir piper-tts pathvalidate

# Código da aplicação
COPY audio/app/ /app/app/

ENV PYTHONPATH=/app
ENV PDC_FRONT=audio
ENV PDC_GPU_MODE=auto

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=10 \
    CMD curl -sf http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
