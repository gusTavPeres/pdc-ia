FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-dev python3-venv     build-essential cmake curl git     && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel scikit-build

# Create libcuda stub symlink for build-time linking
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

WORKDIR /app

COPY shared/requirements.txt /app/shared_req.txt
RUN pip install --no-cache-dir -r /app/shared_req.txt

RUN pip install --no-cache-dir fastapi 'uvicorn[standard]'

# llama-cpp-python WITH CUDA
ENV CMAKE_ARGS='-DGGML_CUDA=on'
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV FORCE_CMAKE=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:$LIBRARY_PATH
RUN pip install --no-cache-dir 'llama-cpp-python==0.3.21' --no-binary llama-cpp-python

# Remove stub symlink so runtime uses real libcuda from nvidia driver
RUN rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1

COPY slm/app/ /app/app/

ENV PYTHONPATH=/app
ENV PDC_FRONT=slm
ENV PDC_GPU_MODE=auto
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=10     CMD curl -sf http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
