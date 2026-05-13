FROM python:3.11-slim

# Dependências do sistema para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependências Python compartilhadas
COPY shared/requirements.txt /app/shared_req.txt
RUN pip install --no-cache-dir -r /app/shared_req.txt

# Pina numpy a versão compatível com PyTorch 2.2 (numpy 2.x quebra torch.from_numpy)
RUN pip install --no-cache-dir "numpy<2.0"

# PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# Dependências de visão (depois do torch para não sobrescrever numpy)
RUN pip install --no-cache-dir \
    "ultralytics==8.2.0" \
    opencv-python-headless \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

# Garante numpy compatível após ultralytics instalar suas dependências
RUN pip install --no-cache-dir "numpy<2.0"

# Código da aplicação
COPY vision/app/ /app/app/

ENV PYTHONPATH=/app
ENV PDC_FRONT=vision
ENV PDC_GPU_MODE=auto

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=8 \
    CMD curl -sf http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
