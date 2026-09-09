# Desenvolvimento

## Layout

- `pdc`: CLI operacional;
- `metrics_server.py`: API local e servidor do dashboard;
- `web/dashboard.html`: interface sem etapa de build;
- `shared/benchmark_suite.py`: fases comuns dos benchmarks;
- `shared/metrics_collector.py`: cgroup, processos, NVML, RAPL e fontes do host;
- `shared/device_telemetry.py`: sysfs, hwmon, devfreq e parser de `tegrastats`;
- `shared/capabilities.py`: inventário e proveniência do hardware;
- `shared/result_writer.py`: persistência atômica;
- `shared/result_parser.py`: normalização de resultados atuais e legados;
- `shared/scorecard.py`: performance, eficiência e custo;
- `shared/model_loader.py`: download, hash e cache;
- `vision/`, `slm/` e `audio/`: serviços FastAPI em containers;
- `telemetry/`: agente multi-arquitetura de sensores do host.

## Ambiente Local

```bash
./pdc setup
.venv/bin/python -m pip install -r requirements-dev.txt
```

O primeiro comando cria `.venv/` e instala as dependências usadas pelo CLI e
pelo dashboard. O segundo instala lint e testes, mantidos separadamente em
`requirements-dev.txt`.

## Validação

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

Os testes HTTP exigem os três containers saudáveis e são opt-in:

```bash
PDC_RUN_INTEGRATION=1 .venv/bin/python -m pytest -q \
  vision/test/test_vision.py slm/test/test_slm.py audio/test/test_audio.py
```

Valide os arquivos Compose sem iniciar serviços:

```bash
PDC_IMAGE_VARIANT=cpu PDC_GPU_MODE=cpu \
  docker compose -f vision/docker-compose.yml config --quiet
```

Repita para `slm` e `audio`. Para a variante NVIDIA, acrescente o arquivo
`docker-compose.gpu.yml` da frente e use `PDC_IMAGE_VARIANT=gpu`.

```bash
PDC_TEGRASTATS_PATH=/dev/null \
  docker compose -f telemetry/docker-compose.yml config --quiet
```

O Dockerfile do agente deve continuar exportável para `linux/amd64`,
`linux/arm64` e `linux/arm/v7`. As frentes de IA têm como alvos CPU `amd64` e
`arm64`.

Testes geram caches locais, mas nenhum cache, modelo ou resultado deve entrar no
Git. A suíte de testes é código do projeto e deve permanecer versionada.

## Dashboard

```bash
./pdc dashboard
```

O servidor lê `web/dashboard.html` a cada acesso à rota `/`; alterações de HTML,
CSS e JavaScript não exigem rebuild. Alterações em `metrics_server.py` exigem
reiniciar o dashboard.

## Docker

```bash
./pdc build all --cpu
./pdc up all --cpu
./pdc status
./pdc down all
```

Os Dockerfiles copiam o código específico da frente para a imagem. Alterações
em `vision/app`, `slm/app` ou `audio/app` exigem rebuild da imagem correspondente.
`shared/` é montado como somente leitura, mas o processo Python precisa ser
reiniciado para importar uma versão nova.

## API do Dashboard

Leitura:

| Endpoint | Uso |
| --- | --- |
| `GET /api/metrics` | telemetria do host, containers e estado de benchmarks |
| `GET /api/results` | resultados normalizados |
| `GET /api/results-html` | linhas renderizadas da tabela |
| `GET /api/scorecard` | resumo de scorecards |
| `GET /api/models` | modelos carregados nos serviços |
| `GET /api/models/catalog` | catálogo configurado |
| `GET /api/containers` | limites e estado do Docker |
| `GET /api/hardware-config` | perfil e custo da bancada |
| `GET /api/hardware-detect` | inventário físico; `?refresh=1` força captura |
| `GET /api/profiles` | perfis de benchmark calculados |
| `GET /api/config` | texto de `config/benchmark.yaml` |

Escrita:

| Endpoint | Uso |
| --- | --- |
| `POST /api/benchmark` | inicia uma frente e perfil |
| `POST /api/inference` | inferência SLM |
| `POST /api/vision-infer` | inferência a partir de imagem em base64 |
| `POST /api/tts` | síntese de fala |
| `POST /api/switch-model` | compatibilidade para troca de SLM |
| `POST /api/models/save` | cria ou atualiza definição |
| `POST /api/models/default` | altera o padrão |
| `POST /api/models/delete` | remove definição do catálogo |
| `POST /api/models/activate` | carrega modelo em uma frente |
| `POST /api/containers/update` | aplica e persiste limites |
| `POST /api/containers/restart` | reinicia uma frente |
| `POST /api/hardware-config` | salva perfil e custos |
| `POST /api/config` | valida e salva perfis de benchmark |

As APIs não possuem autenticação e devem continuar restritas a
`127.0.0.1` ou a um túnel protegido.

## Dados Gerados

Não versionar:

- `.venv/`, `.pytest_cache/`, `.ruff_cache/` e `__pycache__/`;
- `data/models/` e `data/ultralytics/`;
- `data/datasets/`;
- `data/hardware/`;
- `results/`;
- arquivos `.gguf`, `.pt`, `.onnx` e `.onnx.json`.

`data/hardware/host.json` é atualizado no host e montado nos containers. Ele
fornece o inventário físico de `system_info`; a detecção interna permanece em
`runtime_visibility`.

## Revisão Antes de Publicar

1. Executar lint, testes unitários e testes HTTP.
2. Validar Compose CPU e GPU das três frentes.
3. Confirmar que URLs de modelos respondem sem baixar os arquivos completos.
4. Abrir o dashboard em desktop e mobile e verificar o console do navegador.
5. Garantir que `results/`, caches, modelos e credenciais não estejam no diff.
6. Revisar `git diff --check` e `git status --short`.

Não armazene tokens em URLs de remoto. Verifique antes de publicar:

```bash
git remote -v
git config --get-regexp '^remote\..*\.url$'
```
