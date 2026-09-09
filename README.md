# PDC-IA Hardware Tester

Plataforma local para executar cargas de visão, linguagem e áudio em containers
separados e registrar desempenho, consumo de recursos e contexto do hardware.

Os modelos ficam em `data/models/` e os resultados em `results/`. Nenhum desses
arquivos é versionado.

## Escopo da V1

- visão: detecção de objetos com YOLO;
- SLM: geração de texto em GGUF com `llama-cpp-python`;
- áudio: ASR com `faster-whisper` e TTS com Piper;
- execução CPU em containers portáveis;
- variante CUDA para GPU NVIDIA em visão, SLM e ASR;
- inventário de CPU, RAM, armazenamento, GPU, aceleradores PCI e interfaces
  seriais;
- métricas por container via cgroup, com fallback explícito quando a telemetria
  não está disponível;
- agente multi-arquitetura para sensores Linux, devfreq, hwmon e Jetson
  `tegrastats`;
- score de desempenho e estimativas de custo quando a bancada está configurada.

NPU, TPU, ROCm, OpenVINO e upload para microcontroladores ainda não possuem
adaptadores de execução. A telemetria aceita provedores externos, mas a presença
do dispositivo não significa que a carga foi executada nele. Consulte
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) para a matriz de suporte.

## Requisitos

- Linux com Docker e Docker Compose;
- Python 3.10 ou mais recente no host;
- espaço em disco para imagens Docker e modelos;
- NVIDIA Driver e `nvidia-container-toolkit` somente para a variante CUDA.

O modo CPU não depende de uma GPU NVIDIA. Sem telemetria compatível, VRAM,
potência e temperatura permanecem ausentes em vez de serem estimadas.
Piper TTS executa em CPU também na variante GPU desta versão.
Em Linux ARM64, use a variante CPU; a variante CUDA atual é destinada a NVIDIA
dGPU em `x86_64`.

## Primeiro Uso

```bash
git clone https://github.com/PDC-EAI/hardware-tester-ai.git
cd hardware-tester-ai

./pdc setup
./pdc models download defaults
./pdc build all
./pdc up all
./pdc status
./pdc dashboard
```

Abra `http://127.0.0.1:8099`. O dashboard e as APIs escutam somente no host
local. Para acesso remoto, encaminhe a porta por SSH:

```bash
ssh -L 8099:127.0.0.1:8099 usuario@maquina
```

Antes de comparar custos, preencha `Hardware > Hardware e custo operacional`.
O arquivo versionado não contém preço ou tarifa de uma bancada específica.

## Modelos

```bash
./pdc models list
./pdc models download defaults
./pdc models download all
./pdc models download vision yolo11n
./pdc models download slm qwen3-4b-q4
./pdc models download audio_tts --defaults
```

`defaults` baixa os modelos padrão. `all` baixa todos os itens que possuem URL
direta e pode consumir vários gigabytes. Modelos ASR definidos por `model_id`
são obtidos pelo runtime no primeiro carregamento.

Também é possível cadastrar, ativar e remover modelos em
`Configuração > Modelos`. Links de arquivo do Hugging Face, GitHub Releases e
outros servidores HTTP são aceitos. SLM usa `.gguf`; visão usa `.pt`; Piper usa
o par `.onnx` e `.onnx.json`. Ollama não é importado diretamente.

Detalhes: [`docs/MODELS.md`](docs/MODELS.md).

## Benchmarks

```bash
./pdc benchmark vision --profile quick
./pdc benchmark slm --profile quick
./pdc benchmark audio --profile quick --mode asr
./pdc benchmark audio --profile quick --mode tts
./pdc benchmark audio --profile quick --mode pipeline
./pdc benchmark all --profile express
```

Perfis padrão:

| Perfil | Finalidade | Duração fixa mínima |
| --- | --- | ---: |
| `express` | verificação rápida | 20 s + 10 amostras de latência |
| `quick` | execução curta | 30 s + 15 amostras de latência |
| `standard` | comparação principal | 3 min 30 s + 40 amostras |
| `extended` | carga sustentada | 14 min + 200 amostras |

A duração real inclui aquecimento e o tempo de cada inferência. Os parâmetros
ficam em `config/benchmark.yaml`. Para comparar hardware, execute uma frente por
vez; cargas concorrentes compartilham CPU, memória e telemetria de dispositivo.

## Resultados

```bash
./pdc results
./pdc results --latest
./pdc results --scorecard
./pdc results --compare arquivo-a.json arquivo-b.json
```

Cada execução gera um único JSON com `run_id`. O resultado separa:

- inventário físico do host em `system_info`;
- dispositivos visíveis ao container em `system_info.runtime_visibility`;
- política de execução em `metadata.execution`;
- métricas e seus escopos em `benchmark_results`;
- indicadores derivados em `scorecard`.

O CLI atualiza `data/hardware/host.json` antes de subir containers e antes de
executar benchmarks. Isso evita registrar o filesystem `overlay` como se fosse
o SSD ou eMMC físico.
O container `pdc-telemetry` é iniciado junto das frentes e publica sensores do
host em `data/telemetry/host.json`.

Contrato dos campos: [`docs/RESULTS.md`](docs/RESULTS.md). Regras de comparação:
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Operação

```bash
./pdc detect
./pdc detect --json
./pdc detect --save
./pdc build all --cpu
./pdc up all --cpu
./pdc status
./pdc down all
```

Sem `--cpu` ou `--gpu`, o CLI usa a variante NVIDIA apenas quando
`nvidia-smi` e o runtime estão funcionais. Os limites configurados em
`Hardware > Limites de execução` são reaplicados no próximo `up`.

Valor zero não adiciona limite pelo Docker. Políticas herdadas do host ainda
podem aparecer como limites efetivos no resultado. `/dev/shm` usa 256 MB por
padrão e pode ser alterado na criação:

```bash
PDC_SHM_SIZE=1g ./pdc up vision --cpu
```

## Documentação

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): componentes, fluxo e suporte
  de hardware;
- [`docs/ARM.md`](docs/ARM.md): Raspberry Pi, Jetson, SBCs e provedores de
  acelerador;
- [`docs/RESULTS.md`](docs/RESULTS.md): esquema dos resultados e escopo das
  métricas;
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): regras de medição e comparação;
- [`docs/MODELS.md`](docs/MODELS.md): cadastro e cache de modelos;
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md): ambiente, validação e APIs.

## Limitações Conhecidas

- as imagens CPU possuem dependências ARM64, mas cada família de placa ainda
  deve ser validada nativamente;
- Jetson exige uma variante de execução alinhada ao JetPack/L4T para usar CUDA;
- energia de CPU depende de acesso a RAPL/powercap;
- utilização e potência de GPU podem representar o dispositivo inteiro quando
  o driver não oferece atribuição por processo;
- os datasets atuais priorizam repetibilidade e sanidade operacional; não são
  rankings de qualidade dos modelos;
- resultados antigos sem versão ou escopo continuam legíveis, mas aparecem
  como legados e não devem ser misturados automaticamente com a metodologia
  atual.
