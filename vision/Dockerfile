FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN set -eux; \
    packages="curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender1"; \
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
    torch==2.11.0 torchvision==0.26.0 --index-url "${TORCH_INDEX_URL}"

RUN pip install --no-cache-dir \
    ultralytics==8.4.102 \
    opencv-python-headless \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

COPY vision/app/ /app/app/

ENV PYTHONPATH=/app \
    PDC_FRONT=vision \
    PDC_GPU_MODE=cpu \
    YOLO_CONFIG_DIR=/app/data/ultralytics

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=8 \
    CMD curl -sf http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
