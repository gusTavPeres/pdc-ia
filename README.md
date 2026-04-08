# PDC-IA

Plataforma pra testar hardware rodando modelos de IA. A ideia é medir o desempenho real da máquina com workloads de visão computacional, linguagem e áudio.

## O que tem aqui

O projeto roda 3 containers Docker, cada um com um tipo diferente de modelo:

- **Vision** (porta 8010) — YOLOv8 pra detecção de objetos, roda na GPU
- **SLM** (porta 8011) — TinyLlama 1.1B e Qwen2.5 0.5B (modelos de linguagem pequenos em GGUF), também na GPU  
- **Audio** (porta 8012) — Whisper pra transcrição e Piper pra síntese de voz em pt-BR

Tem também um dashboard web na porta 8099 que centraliza tudo: monitoramento da GPU em tempo real, disparo de benchmarks, teste de inferência, comparação entre modelos, etc.

## Estrutura

```
pdc-ia/
├── vision/          # container de visão (YOLOv8 + PyTorch CUDA)
├── slm/             # container de linguagem (llama-cpp-python com CUDA)
├── audio/           # container de áudio (faster-whisper + piper-tts)
├── shared/          # lib compartilhada (benchmark, métricas, capabilities)
├── config/          # yamls de configuração (modelos, perfis de benchmark)
├── metrics_server.py  # dashboard web
├── pdc              # cli
└── bench_compare.py # script de comparação entre modelos
```

## Como rodar

Precisa de Docker com nvidia-container-toolkit e driver NVIDIA 530+.

```bash
cd ~/pdc-ia

# subir os containers
cd vision && docker compose up -d && cd ..
cd slm && docker compose up -d && cd ..
cd audio && docker compose up -d && cd ..

# subir o dashboard
python3 metrics_server.py &

# acessar
# http://localhost:8099
```

Ou pelo CLI:
```bash
./pdc detect        # ver hardware
./pdc status        # ver containers
./pdc benchmark vision --profile quick
```

## Benchmarks

Os perfis ficam em `config/benchmark.yaml` e dá pra editar direto pelo dashboard (aba Configuração). Cada perfil controla duração, número de amostras, se roda teste de carga sustentada, etc.

| Perfil | Tempo aprox. |
|--------|-------------|
| express | ~15s |
| quick | ~30s |
| standard | ~2min |
| extended | ~10min |

## Métricas coletadas

- Throughput (ops/s ou tokens/s)
- Latência (p50, p95, p99, desvio padrão)
- Uso de recursos (RAM, CPU%, GPU%, VRAM, temperatura)
- Carga sustentada (degradação ao longo do tempo)
- Bandwidth de memória (opcional)

## APIs

Todos os containers seguem a mesma interface REST:

```
GET  /health     — status do container
GET  /info       — info do modelo e hardware
POST /predict    — rodar inferência
POST /benchmark  — executar benchmark
```

## Hardware usado nos testes

- GPU: NVIDIA RTX A2000 (6GB VRAM)
- CPU: AMD Ryzen 5 7600
- RAM: 32GB
- SO: Ubuntu / Docker
