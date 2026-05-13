# PDC-EAI — Plataforma de Benchmark de IA em Hardware Real

Plataforma para avaliar o desempenho de hardware executando cargas de trabalho reais de inteligência artificial. O objetivo é medir, de forma objetiva e reproduzível, como uma máquina se comporta ao rodar modelos de visão computacional, linguagem e áudio — tanto em uso isolado quanto sob carga sustentada.

## Motivação

Benchmarks sintéticos não capturam o comportamento real de pipelines de IA. Esta plataforma executa modelos reais em containers isolados e coleta métricas completas: throughput, latência percentílica, uso de GPU/CPU/RAM, temperatura e custo energético estimado.

## Arquitetura

Três serviços independentes em Docker, cada um especializado em um domínio de IA:

- **Vision** — detecção de objetos em tempo real com modelos YOLO (GPU)
- **SLM** — inferência de modelos de linguagem pequenos em formato GGUF (GPU)
- **Audio** — transcrição de fala (ASR) e síntese de voz (TTS) em português

Um **dashboard web** centraliza monitoramento em tempo real, execução de benchmarks, teste de inferência interativo e comparação entre modelos.

## Métricas coletadas

- Throughput (ops/s ou tokens/s)
- Latência p50 / p95 / p99 / desvio padrão
- Utilização de GPU, VRAM, CPU, RAM e temperatura
- Degradação sob carga sustentada
- Energia estimada por operação

## Perfis de benchmark

| Perfil | Duração |
|--------|---------|
| express | ~15s |
| quick | ~30s |
| standard | ~2min |
| extended | ~10min |

## Estrutura

```
├── vision/            # container de visão
├── slm/               # container de linguagem
├── audio/             # container de áudio
├── shared/            # lib compartilhada (benchmark, métricas)
├── config/            # configuração de modelos e perfis
├── metrics_server.py  # dashboard web (porta 8099)
└── pdc                # CLI
```

## Requisitos

Docker com `nvidia-container-toolkit` e driver NVIDIA 530+.
