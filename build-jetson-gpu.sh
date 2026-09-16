#!/usr/bin/env bash
# Monta as imagens GPU do Jetson para as tres frentes, via docker run+exec+commit.
#
# Por que run+commit e nao `docker build`: o buildkit no Jetson tem DNS instavel
# ("Name or service not known" intermitente) e a base aponta o pip para o indice
# do Jetson (sem os pacotes do PyPI). O run+commit usa a rede do `docker run`
# (que resolve) e deixa fixar PIP_INDEX_URL. Nada e instalado no host — tudo fica
# nas imagens pdc-<front>:jetson-gpu.
#
# Rodar NO Jetson (a partir de ~/benchmarks_gustavo, onde estao shared/ e <front>/app):
#   bash build-jetson-gpu.sh [vision|slm|audio|all]
set -uo pipefail
cd "$(dirname "$0")"
WHAT="${1:-all}"
PIP="https://pypi.org/simple"

# instala pacotes preservando o que a base ja traz (torch/llama/ct2 com CUDA).
_commit() {  # <container> <imagem-final> <front>
  local c="$1" img="$2" front="$3"
  docker exec "$c" mkdir -p /app/app /app/shared
  docker cp shared/. "$c":/app/shared/
  docker cp "$front"/app/. "$c":/app/app/
  docker commit \
    --change "ENV PYTHONPATH=/app PDC_FRONT=$front PDC_GPU_MODE=gpu" \
    --change 'WORKDIR /app' \
    --change 'CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]' \
    "$c" "$img" >/dev/null
  docker rm -f "$c" >/dev/null
  echo "  commitada: $img"
}

build_vision() {
  echo "== vision (torch CUDA + ultralytics) =="
  docker rm -f b-vis 2>/dev/null
  docker run -d --name b-vis --network host dustynv/l4t-pytorch:r36.2.0 sleep infinity >/dev/null
  # torch/torchvision CUDA ja na base; ultralytics --no-deps p/ nao trocar o torch;
  # numpy<2 (o torch da base foi compilado com numpy 1.x).
  docker exec -e PIP_INDEX_URL=$PIP b-vis pip3 install --no-cache-dir --no-deps \
    ultralytics==8.2.103 ultralytics-thop==2.0.0 2>&1 | tail -1
  docker exec -e PIP_INDEX_URL=$PIP b-vis pip3 install --no-cache-dir \
    numpy==1.26.4 opencv-python-headless pillow pyyaml requests scipy tqdm pandas \
    psutil py-cpuinfo matplotlib fastapi "uvicorn[standard]" python-multipart 2>&1 | tail -1
  _commit b-vis pdc-vision:jetson-gpu vision
}

build_slm() {
  echo "== slm (llama.cpp CUDA) =="
  docker rm -f b-slm 2>/dev/null
  docker run -d --name b-slm --network host dustynv/llama_cpp:r36.2.0 sleep infinity >/dev/null
  # llama-cpp-python com CUDA ja vem na base. So o servidor da frente.
  docker exec -e PIP_INDEX_URL=$PIP b-slm pip3 install --no-cache-dir \
    numpy==1.26.4 psutil pyyaml requests fastapi "uvicorn[standard]" 2>&1 | tail -1
  _commit b-slm pdc-slm:jetson-gpu slm
}

build_audio() {
  echo "== audio (faster-whisper/CTranslate2 CUDA; Piper/TTS em CPU) =="
  docker rm -f b-aud 2>/dev/null
  docker run -d --name b-aud --network host dustynv/whisperx:r36.2.0 sleep infinity >/dev/null
  # CTranslate2 CUDA ja na base. faster-whisper + piper + servidor.
  # faster-whisper compativel com o ctranslate2 3.24 (CUDA) da base, com
  # --no-deps para NAO trocar o ctranslate2 por um de CPU. Depois as deps dele
  # (menos ctranslate2) + piper + servidor.
  docker exec -e PIP_INDEX_URL=$PIP b-aud pip3 install --no-cache-dir --no-deps \
    faster-whisper==0.10.1 2>&1 | tail -1
  docker exec -e PIP_INDEX_URL=$PIP b-aud pip3 install --no-cache-dir \
    av huggingface-hub tokenizers onnxruntime piper-tts==1.5.0 soundfile \
    numpy==1.26.4 psutil pyyaml requests fastapi "uvicorn[standard]" python-multipart 2>&1 | tail -1
  _commit b-aud pdc-audio:jetson-gpu audio
}

[ "$WHAT" = all ] || [ "$WHAT" = vision ] && build_vision
[ "$WHAT" = all ] || [ "$WHAT" = slm ]    && build_slm
[ "$WHAT" = all ] || [ "$WHAT" = audio ]  && build_audio
echo "== imagens GPU do Jetson prontas =="
docker images | grep "jetson-gpu"
