# tools/campanhas — campanhas de benchmark/estresse orquestradas pelo PC

Orquestração **pelo PC, via SSH**, das campanhas de teste nos devices da frota
(alternativa ao `run-batch.sh`, que roda no próprio device sem SSH). O código da
plataforma roda no device (deploy em `~/benchmarks_gustavo` lá, `--path`); estes
scripts disparam os testes, coletam telemetria e trazem tudo para
`tools/campanhas/results/` (ignorado pelo git), onde o `report.py` gera o HTML.

## Peças

| Arquivo | O quê |
| --- | --- |
| `campaign.sh` | roda N repetições de vários perfis numa frente/device, cada teste com sua telemetria, e traz tudo pro PC |
| `sampler.py` | série temporal de telemetria (lê o `host.json` do agente pdc-telemetry — dados ricos do tegrastats: potência VDD_IN, GPU, CPU normalizada, temp, throttle) |
| `report.py` | gera `report.html` autocontido: gráficos SVG por teste + tabelas de desempenho com mediana e desvio |
| `results/<device>/<campanha>/` | um dir por teste (`result.json` + `telemetria.jsonl`) e o `report.html` |

## Rodar uma campanha

```bash
bash campaign.sh jetson-nano \
  --profiles "quick:3 standard:2 soak:1" \   # reps por perfil
  --fronts "vision audio slm" \
  --path '~/benchmarks_gustavo'
```

- **desempenho**: `quick` (rápido), `standard` (2 min sustentado). Repita (`:3`) para ver dispersão.
- **estresse**: `soak` (30 min), `torture` (60 min). Carga sustentada expõe throttling e degradação.
- build+up acontece uma vez por frente; cada repetição é só o benchmark.
- o device baixa o código do GitHub quando há credencial; sem ela, `--from-local` (rsync).

## Dashboard ao vivo

No device: `PDC_DASHBOARD_HOST=<ip-mesh> PDC_DASHBOARD_PORT=8099 .venv/bin/python metrics_server.py`
Acesse no navegador em `http://<device>:8099` (bind na mesh, isolado da LAN do device).

## Telemetria — o que cada métrica significa

- **CPU**: utilização normalizada 0-100% (NÃO a soma de núcleos do cgroup, que vai a 600% num hexa-core; `docker stats` usa aquela convenção, esta não).
- **Potência**: trilho de entrada da placa `VDD_IN` no Jetson (via tegrastats). Ausente no Raspberry Pi (sem sensor no kernel).
- **GPU**: utilização/clock via tegrastats no Jetson. Fica em ~0 quando a carga roda em CPU (ex.: vision sem CUDA) — ausência real, não erro.
- **throttling**: `throttling.active` do device. Faixa vermelha no gráfico marca a janela.
- Uma fonte inexistente na placa aparece como "sem dados", nunca zerada.

## Achados 12/09/2026 (Jetson Nano, aarch64, 6 cores, 7.4 GB)

| Frente | métrica (mediana de 3 quick) | soak 30 min |
| --- | --- | --- |
| vision (YOLOv8n CPU) | 2.54 frames/s (±0.015) | 2.44 fps, tmax 55.9 °C, sem throttle, degr −2.5% |
| slm (TinyLlama Q4 CPU) | 20.7 tokens/s (±0.23) | 20.9 tok/s, tmax 60.2 °C, sem throttle |
| audio (whisper+piper) | 0.32× tempo real (±0.001) | 0.32×, tmax 59.3 °C, sem throttle |

O Nano sustentou 30 min de carga em todas as frentes **sem throttling térmico**
(picos 55-60 °C). No soak do SLM a potência quase dobrou (5.8→9.6 W) e a CPU foi
a 73%, estável. Reprodutibilidade alta (desvio baixo entre repetições).
