# Desenvolvimento

## Layout

- `metrics_server.py`: API local do dashboard.
- `web/dashboard.html`: interface web servida pelo `metrics_server.py`.
- `pdc`: CLI operacional.
- `shared/benchmark_suite.py`: execução dos testes e coleta de métricas.
- `shared/scorecard.py`: métricas derivadas de custo, energia e score.
- `shared/model_loader.py`: download e cache de modelos.
- `vision/`, `slm/`, `audio/`: serviços FastAPI executados em containers.

## Validação Local

Com containers ativos:

```bash
python3 -m py_compile metrics_server.py shared/*.py vision/app/*.py slm/app/*.py audio/app/*.py
python3 vision/test/test_vision.py
python3 slm/test/test_slm.py
python3 audio/test/test_audio.py
```

Com `pytest` em venv local:

```bash
python3 -m venv .venv-test
.venv-test/bin/pip install -r requirements-dev.txt
.venv-test/bin/python -m pytest vision/test/test_vision.py slm/test/test_slm.py audio/test/test_audio.py
```

## Dashboard

O dashboard é HTML/CSS/JS sem build step. O servidor lê `web/dashboard.html` a cada acesso à rota `/`, então alterações visuais não exigem restart do servidor.

```bash
python3 metrics_server.py
```

## Docker

O compose base não exige GPU. Quando o CLI detecta NVIDIA, ele adiciona `docker-compose.gpu.yml` da frente correspondente.

```bash
./pdc up all
./pdc up all --gpu
./pdc up all --cpu
```

`--cpu` define `PDC_GPU_MODE=cpu` dentro dos containers.

## API Local

Endpoints principais:

- `GET /api/metrics`
- `GET /api/results`
- `GET /api/results-html`
- `GET /api/models`
- `GET /api/models/catalog`
- `POST /api/benchmark`
- `POST /api/inference`
- `POST /api/vision-infer`
- `POST /api/tts`
- `POST /api/models/save`
- `POST /api/models/activate`

## Git

Não armazene tokens em URLs de remoto.

Use:

```bash
git remote -v
git config --get-regexp '^remote\\..*\\.url$'
```

URLs com token ou credencial embutida devem ser removidas antes de qualquer push.

## Dados Gerados

Não versionar:

- `data/models/`
- `data/datasets/`
- `results/`
- `*.gguf`
- `*.pt`
- `*.onnx`
- `*.onnx.json`

Esses padrões já estão no `.gitignore`.
