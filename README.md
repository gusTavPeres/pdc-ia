# Hardware Tester AI

Ferramenta local para medir desempenho de hardware executando cargas reais de visão, linguagem e áudio em containers separados.

O projeto foi montado para comparar máquinas x86, ARM, SBCs e aceleradores de forma consistente. Os modelos não ficam no Git; eles são baixados para `data/models/` conforme a configuração em `config/models.yaml`.

## Componentes

- `vision`: detecção de objetos com modelos YOLO.
- `slm`: geração de texto com modelos GGUF via `llama-cpp-python`.
- `audio`: ASR com `faster-whisper` e TTS com Piper.
- `metrics_server.py`: dashboard local e API de controle.
- `pdc`: CLI para build, status, benchmarks, resultados e modelos.
- `shared/`: coleta de métricas, execução de benchmark, metodologia e scorecard.

## Requisitos

- Linux com Docker e Docker Compose.
- Python 3.10+ no host.
- NVIDIA Driver + `nvidia-container-toolkit` para testes com GPU NVIDIA.
- Espaço em disco para modelos em `data/models/`.

O sistema também roda sem GPU NVIDIA, mas as métricas de CUDA, VRAM, potência e temperatura ficam indisponíveis ou parciais.

## Primeiro Uso

```bash
git clone https://github.com/PDC-EAI/hardware-tester-ai.git
cd hardware-tester-ai

python3 -m pip install -r shared/requirements.txt

./pdc models download defaults
./pdc build all
./pdc up all
./pdc status

python3 metrics_server.py
```

Dashboard:

```text
http://localhost:8099
```

## Modelos

Os modelos são definidos em `config/models.yaml` e baixados para `data/models/`, que é ignorado pelo Git.

Comandos úteis:

```bash
./pdc models list
./pdc models download defaults
./pdc models download all
./pdc models download slm tinyllama-1.1b-q4
./pdc models download audio_tts --defaults
```

`defaults` baixa apenas os modelos padrão de cada frente. `all` baixa todos os modelos com URL direta. Modelos ASR do `faster-whisper` usam `model_id` e podem ser baixados pelo runtime no primeiro carregamento do container.

Também é possível cadastrar modelos pelo dashboard em `Configuração > Modelos`. Para Hugging Face, use o link do arquivo do modelo; links com `/blob/` são convertidos para `/resolve/`. Para SLM, use arquivos `.gguf`. Para TTS, use o `.onnx`; o `.onnx.json` é inferido quando possível.

Mais detalhes: [`docs/MODELS.md`](docs/MODELS.md).

## Benchmarks

Pelo CLI:

```bash
./pdc benchmark vision --profile quick
./pdc benchmark slm --profile quick
./pdc benchmark audio --profile quick
./pdc benchmark all --profile express
```

Pelo dashboard, use a aba `Benchmarks` para escolher modelo e perfil.

Perfis padrão:

| Perfil | Uso | Duração aproximada |
| --- | --- | --- |
| `express` | Sanidade rápida | 15s |
| `quick` | Teste curto | 30s |
| `standard` | Comparação principal | 2min |
| `extended` | Carga sustentada | 10min |

Os parâmetros ficam em `config/benchmark.yaml`.

## Resultados

Os resultados são salvos em `results/*.json`, fora do Git.

```bash
./pdc results
./pdc results --latest
./pdc results --scorecard
```

O dashboard mostra throughput, latência, uso de RAM/VRAM/GPU, temperatura, energia estimada, score e custo por unidade quando há telemetria suficiente.

A metodologia de comparação está em [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Hardware e Custo

Edite `config/hardware.yaml` para registrar o perfil da bancada, preço do hardware, custo de energia e referências de score.

Esses dados não mudam o benchmark bruto. Eles só alimentam campos derivados como custo por 1k operações, energia por operação e score relativo.

## Operação

```bash
./pdc detect --save
./pdc build all
./pdc up all
./pdc status
./pdc down all
```

`./pdc up` detecta GPU NVIDIA automaticamente. Para forçar modo:

```bash
./pdc up all --gpu
./pdc up all --cpu
```

Para reiniciar uma frente:

```bash
./pdc down slm
./pdc up slm
```

## Desenvolvimento

Validação rápida:

```bash
python3 -m py_compile metrics_server.py shared/*.py vision/app/*.py slm/app/*.py audio/app/*.py
python3 vision/test/test_vision.py
python3 slm/test/test_slm.py
python3 audio/test/test_audio.py
```

Para rodar com `pytest` sem alterar o Python do sistema:

```bash
python3 -m venv .venv-test
.venv-test/bin/pip install -r requirements-dev.txt
.venv-test/bin/python -m pytest vision/test/test_vision.py slm/test/test_slm.py audio/test/test_audio.py
```

Notas de desenvolvimento: [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## Estrutura

```text
.
├── audio/
├── config/
├── docs/
├── shared/
├── slm/
├── vision/
├── web/
├── metrics_server.py
└── pdc
```

## Limitações

- NPU, TPU, OpenVINO, ROCm e aceleradores externos ainda precisam de adaptadores próprios.
- ASR via `faster-whisper` pode baixar artefatos no primeiro carregamento quando o modelo é definido por `model_id`.
- Métricas de energia fora de GPU NVIDIA dependem de RAPL/powercap ou medidor externo.
- O CLI ativa o override de GPU quando detecta NVIDIA; use `./pdc up all --cpu` para forçar execução sem GPU.
