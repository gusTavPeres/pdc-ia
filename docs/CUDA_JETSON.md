# GPU/CUDA no Jetson — investigação e habilitação

> Documento de entrega. Registra por que a variante GPU estava bloqueada em ARM,
> o que foi verificado no device, a que categoria de mudança pertence, e o
> caminho seguro para habilitar a inferência na GPU do Jetson Orin.

## Pergunta

Por que `./pdc build <front> --gpu` recusa em ARM ("A variante CUDA atual é
destinada a NVIDIA dGPU x86_64; em ARM use --cpu") e o CUDA "não funciona" no
Jetson?

## Resposta curta

O CUDA **funciona** no Jetson. O bloqueio é uma **trava de código** que impede
construir a imagem GPU do repositório em ARM — porque essa imagem é feita para
**GPU dedicada x86_64**, não para o Jetson (L4T/JetPack). Não há nada quebrado no
device nem no kernel: CUDA, driver, GPU e runtime Docker já vêm no JetPack.

## O que foi verificado no device (15-16/09/2026)

Device: `jetson-nano` (na verdade um **Jetson Orin Nano**, aarch64).

| Item | Estado no device |
| --- | --- |
| L4T / JetPack | **R36.5.0** (JetPack 6.x), `nvidia-l4t-core 36.5.0` |
| CUDA no host | **12.6** instalado (`/usr/local/cuda-12.6`, `nvcc V12.6.68`) |
| GPU | **Orin, compute capability 8.7 (Ampere), 8 GB** (memória unificada) |
| Dispositivos | `/dev/nvhost-*`, `/dev/nvgpu*` presentes |
| Runtime Docker NVIDIA | **presente** (`/etc/docker/daemon.json` tem `runtimes.nvidia`) |
| nvidia-container-toolkit | **instalado** (1.16.2) |
| Disco livre | 771 GB |

**Prova de compute (host, sem instalar nada):** um programa CUDA mínimo compilado
com `nvcc` e executado em `/tmp` reportou:

```
GPUs CUDA visiveis: 1
Device0: Orin | capability 8.7 | 8.0 GB
```

Ou seja, a pilha CUDA do host está íntegra e a GPU computa.

## Por que a imagem GPU do repositório não serve no Jetson

Os Dockerfiles de GPU usam bases de **x86_64**:

```
slm/Dockerfile.gpu:   FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
audio/Dockerfile.gpu: FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
```

As imagens `nvidia/cuda:12.x` do Docker Hub trazem a userspace CUDA de **x86_64**
e não rodam em ARM/Jetson. No Jetson a CUDA vem do JetPack no host e é injetada no
container pelo `nvidia-container-runtime`; a base precisa ser uma imagem **L4T**
(`nvcr.io/nvidia/l4t-*` ou equivalente da comunidade) casada com o R36, e os
runtimes (PyTorch, CTranslate2, TensorRT) precisam ser builds do JetPack.

A trava no código:

```python
def _supports_current_cuda_images():
    return os.uname().machine.lower() in {"x86_64", "amd64"}
```

Ela existe para **não** produzir uma imagem quebrada (CUDA de x86 numa placa ARM).
É correta enquanto a variante Jetson não existir.

## Categoria da mudança (regra do projeto)

- **Limitação proposital?** A trava é proposital, mas protege contra uma imagem
  ausente, não é uma política de "não usar GPU". Trocá-la é mudança de código.
- **Precisa mexer em kernel / imagem do sistema?** **Não.** CUDA, driver, GPU e
  runtime Docker já estão no device (JetPack). Nada de kernel/sistema é tocado.
- **É instalável/container sem risco de quebrar o sistema?** **Sim.** Habilitar
  exige: (a) uma imagem Docker com PyTorch/CTranslate2 do Jetson (isolada, não
  toca o host), (b) rodar com `--runtime nvidia`, (c) ajustar o flag de código.
  Tudo containerizado e reversível.

Conclusão: **pode ser feito com segurança** — não há mudança de kernel nem de
imagem do sistema.

## Caminho de habilitação (planejado)

1. Base L4T/JetPack 6 (r36) com PyTorch do Jetson para a frente de visão (YOLO).
2. Compose GPU do Jetson usando `runtime: nvidia` e device requests da GPU.
3. Detecção de Jetson no CLI (ler L4T) para liberar `--gpu` só quando a imagem
   Jetson existir — sem afetar o caminho x86 nem o CPU.
4. Medir YOLO na GPU e comparar com a CPU (o ganho esperado é de várias vezes).

## Registro de progresso

- (será preenchido conforme a habilitação avança)

### 16/09 — passthrough de GPU para container: OK

Com o runtime nvidia e as variáveis certas, o container **enxerga a GPU** (imagem
`pdc-vision:cpu` só para o teste de passthrough):

```
docker run --rm --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all <img>
```

Resultado dentro do container:
```
nvhost: 12 devs | nvgpu: 1
libcuda: /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1
```

O `nvidia-container-runtime` está em modo `auto` (CSV) e injeta os dispositivos
`/dev/nvhost*`, `/dev/nvgpu*` e o driver `libcuda.so.1`. Ou seja: **o caminho de
container para a GPU está pronto** — falta uma imagem cujo PyTorch/CTranslate2
seja compilado para o Jetson (CUDA), já que a `pdc-vision:cpu` traz Torch de CPU.

### 16/09 — PyTorch + CUDA no container: OK (compute real)

Imagem pública `dustynv/l4t-pytorch:r36.2.0` (PyTorch para JetPack 6, mantida pela
NVIDIA; containerizada, não instala nada no host — 21 GB, disco livre 771 GB):

```
torch 2.2.0 | cuda True | device Orin
20x matmul 4000^3 na GPU: 2.125s (1205 GFLOP/s)  # ~1.2 TFLOP/s FP32
VRAM usada: 139 MB de 8.0 GB
```

**Fica provado, ponta a ponta, que a GPU do Orin roda PyTorch com CUDA dentro de
um container.** O bloqueio era só a ausência de uma imagem ARM/Jetson + a trava de
código — nada no device.

### 16/09 — RESULTADO: YOLO na GPU é 24× mais rápido que na CPU

Imagem de visão do Jetson montada sobre `dustynv/l4t-pytorch:r36.2.0` (torch 2.2.0
CUDA) + ultralytics 8.2.103. Mesmo `yolov8n`, mesma imagem 640×640, 50 inferências:

```
cuda:0 (modelo em cuda:0): 22.3 fps ( 44.8 ms/frame)
cpu    (modelo em cpu):     0.9 fps (1091.5 ms/frame)
>>> ganho GPU/CPU: 24.4x
```

A GPU do Orin entrega **~24× o desempenho da CPU** em detecção YOLO — a vantagem
do Jetson que o caminho CPU-only não media. (Contra os 2,7 fps do benchmark CPU do
`pdc`, a GPU a 22 fps é ~8×; a diferença de baseline vem do harness — aqui é
`predict()` cru, uma imagem por vez.)

### Percalços do empacotamento (e correções)

1. **Índice pip da imagem base** aponta para `jetson.webredirect.org` (wheels do
   Jetson), sem `ultralytics`. Correção: `PIP_INDEX_URL=https://pypi.org/simple`.
2. **ultralytics puxaria torch de CPU** e sobrescreveria o torch CUDA da base.
   Correção: `pip install --no-deps ultralytics` + as demais dependências à parte.
3. **DNS instável no buildkit** (`Name or service not known` intermitente).
   Correção: montar a imagem via `docker run`+`exec`+`commit` (a rede do `run`
   resolve), em vez de `docker build`.
4. **NumPy 2.x quebra o torch da base** (compilado com NumPy 1.x: "Numpy is not
   available"). Correção: fixar `numpy==1.26.4`.

### Receita reproduzível (sem tocar no host)

```bash
docker run -d --name b --network host dustynv/l4t-pytorch:r36.2.0 sleep infinity
docker exec -e PIP_INDEX_URL=https://pypi.org/simple b \
  pip3 install --no-cache-dir --no-deps ultralytics==8.2.103 ultralytics-thop==2.0.0
docker exec -e PIP_INDEX_URL=https://pypi.org/simple b pip3 install --no-cache-dir \
  numpy==1.26.4 opencv-python-headless pillow pyyaml requests scipy tqdm pandas \
  psutil py-cpuinfo matplotlib fastapi "uvicorn[standard]" python-multipart
docker cp shared/. b:/app/shared/ ; docker cp vision/app/. b:/app/app/
docker commit --change 'ENV PYTHONPATH=/app PDC_FRONT=vision' b pdc-vision:jetson-gpu
docker rm -f b
# rodar com a GPU:
docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all -v <models>:/app/data pdc-vision:jetson-gpu ...
```

## Conclusão da investigação

- **O CUDA não estava quebrado.** A GPU do Orin funciona e entrega **~24× a CPU**
  em YOLO. O bloqueio era só a ausência de uma imagem ARM/Jetson + a trava de
  código `_supports_current_cuda_images()`.
- **Habilitar foi seguro** (regra do projeto): tudo em container, com componentes
  do JetPack já instalados. **Nada de kernel, driver ou imagem do sistema** foi
  tocado. A imagem `pdc-vision:jetson-gpu` fica no device; nada no host mudou.
- **Próximo passo (requer decisão):** integrar como variante `--gpu` do Jetson no
  `./pdc` — detectar L4T, usar o compose com `runtime: nvidia`, e medir SLM/áudio
  na GPU também. Isso mexe no fluxo do CLI, então fica para aprovação.

## Integração no ./pdc (variante GPU do Jetson) e resultados

Variante commitada: `_is_jetson()` detecta o L4T e `--gpu` no Jetson usa
`docker-compose.jetson.yml` (imagem `pdc-<front>:jetson-gpu` + `runtime: nvidia`),
em vez da imagem CUDA x86. As imagens são montadas por `build-jetson-gpu.sh`.

Medido pelo pipeline real do `./pdc` (não `predict()` cru):

| Frente | CPU | GPU (Jetson) | Ganho |
| --- | --- | --- | --- |
| visão (YOLOv8n) | 2,7 fps | **20,4 fps** | **7,5×** |
| SLM (tinyllama-1.1B) | ~20 tok/s | **54,4 tok/s** | **2,7×** |
| áudio (whisper) | — | ver abaixo | — |

Bases usadas (públicas, mantidas pela NVIDIA/comunidade; nada instalado no host):
- visão: `dustynv/l4t-pytorch:r36.2.0` (torch 2.2.0 CUDA) + ultralytics.
- SLM: `dustynv/llama_cpp:r36.2.0` (llama.cpp com offload CUDA).
- áudio: `dustynv/whisperx:r36.2.0` (CTranslate2 3.24 com CUDA).

### Percalço do áudio (CTranslate2)

A base whisperx traz `ctranslate2 3.24.0` **com CUDA (1 device)**. Instalar
`faster-whisper==1.2.1` puxava `ctranslate2 4.8.2` **de CPU** por cima, zerando a
GPU. Correção: `faster-whisper==0.10.1 --no-deps` (compatível com o ct2 3.24 da
base) + as demais dependências, preservando o CTranslate2 CUDA.

### 20-21/09 — SLM: gemma3 e qwen3 não carregavam na GPU (base velha)

**Sintoma.** Na campanha longa, 3 modelos SLM falhavam toda rodada só no Jetson
(RC=1): `gemma3-1b-q4`, `gemma3-1b-q8`, `qwen3-4b-q4`. Os demais (tinyllama,
qwen2.5-0.5b), além de vision e audio, rodavam normalmente. Erro real no
container:

```
[slm] Carregando: gemma-3-1b-it-Q8_0.gguf (formato: gemma3)
[slm] GPU falhou: Failed to load model ... Usando CPU...
POST /switch-model -> 500
```

**Causa.** Não era arquivo faltando (os ggufs estavam presentes) nem memória: o
`llama.cpp` da base `dustynv/llama_cpp:r36.2.0` é **velho demais para as
arquiteturas gemma3 e qwen3** — falha já no parse/registro da arquitetura.
Prova cruzada: os mesmos 3 modelos rodam sem falha no Raspberry Pi (CPU), que
tem um llama.cpp mais novo.

**Correção.** Reconstruir `pdc-slm:jetson-gpu` sobre uma base com llama.cpp novo:
`dustynv/llama_cpp:b5283-r36.4-cu128-24.04` (llama.cpp b5283, `llama-cpp-python
0.3.8`, venv Python 3.12). A base já traz fastapi/uvicorn/numpy 2.2.5/psutil/
pyyaml/requests/pydantic — só faltou `pynvml`. **Não** reinstalar numpy (o
llama_cpp da base foi compilado com o numpy 2.x dele). Ver `build-jetson-gpu.sh`
função `build_slm`.

**cu128 no driver 12.6.** A base é CUDA 12.8; o Jetson roda driver CUDA 12.6.
Funciona por compatibilidade forward de minor version do CUDA 12.x (o runtime
nvidia monta o driver do host) — validado empiricamente: gemma3 e qwen3-4b
carregam com offload total na GPU.

**Resultado (benchmark quick real, GPU, device=cuda):**

| modelo        | tokens/s |
|---------------|----------|
| gemma3-1b-q4  | 27,1     |
| gemma3-1b-q8  | 22,9     |
| qwen3-4b-q4   | 14,8     |

**Memória (qwen3-4b).** O Qwen3-4B usa ~2,4 GB de buffer na GPU. No Jetson de
8 GB unificados com os 3 fronts residentes, se o container de áudio estiver
segurando um whisper grande pode faltar RAM e o qwen3-4b cai em OOM → CPU. É
limite de **memória**, não de arquitetura; com o stack em estado normal
(~2,4 GB livres) ele faz offload total (37/37 layers) na GPU.

**Rollback.** A imagem antiga ficou salva como `pdc-slm:jetson-gpu-r36.2-bak`.
