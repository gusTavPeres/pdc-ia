FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY shared/requirements.txt /app/shared_req.txt
RUN pip install --no-cache-dir -r /app/shared_req.txt
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

ENV CMAKE_ARGS="-DGGML_BLAS=ON;-DGGML_BLAS_VENDOR=OpenBLAS"
RUN pip install --no-cache-dir llama-cpp-python==0.3.34 \
    --no-binary llama-cpp-python

COPY slm/app/ /app/app/

ENV PYTHONPATH=/app \
    PDC_FRONT=slm \
    PDC_GPU_MODE=cpu

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=10 \
    CMD curl -sf http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
