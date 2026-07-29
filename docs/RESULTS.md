# Resultados

Cada benchmark válido gera um arquivo JSON em `results/`. A gravação usa arquivo
temporário e substituição atômica; uma execução concluída produz um único
resultado por frente e modo.

## Estrutura Principal

| Campo | Conteúdo |
| --- | --- |
| `schema_version` | versão do contrato do resultado |
| `run_id` | UUID da execução, usado também para eliminar duplicatas |
| `timestamp` | instante de criação com fuso horário |
| `front` | `vision`, `slm` ou `audio` |
| `model` | modelo ou combinação ASR/TTS executada |
| `mode` | `asr`, `tts` ou `pipeline` para áudio |
| `profile` | perfil de `config/benchmark.yaml` |
| `system_info` | inventário físico e visibilidade do runtime |
| `metadata` | dataset, execução, bancada e chave de comparabilidade |
| `methodology` | versão e parâmetros efetivos do perfil |
| `benchmark_results` | medições brutas de cada fase |
| `scorecard` | indicadores derivados de performance, energia e custo |

## Hardware e Proveniência

`system_info` descreve o host detectado antes da execução. Campos importantes:

- `cpu`, `ram`, `gpu`, `storage`, `accelerators` e `serial`: inventário físico;
- `runtime_visibility`: GPU, armazenamento e dispositivos acessíveis dentro do
  container;
- `provenance.inventory_source`: `host_snapshot` ou `runtime_detection`;
- `provenance.hardware_id`: hash curto dos componentes estáveis da bancada;
- `provenance.captured_at`: momento da captura no host.

`metadata.hardware_profile.id` é o nome definido pelo operador. O
`hardware_id` identifica o inventário; o perfil fornece um nome legível e os
dados de custo. Resultados antigos podem não possuir esses campos.

## Contexto de Execução

`metadata.execution` registra, quando disponível:

- arquitetura e quantidade de CPUs na afinidade;
- backend, device e modo CPU/GPU;
- dispositivos separados de ASR e TTS quando a frente é áudio;
- camadas descarregadas para GPU em SLM;
- limites efetivos de CPU, RAM e processos;
- tamanho de `/dev/shm`;
- variante da imagem Docker.

Limites efetivos podem incluir políticas herdadas do host mesmo que o Docker
não tenha recebido um limite explícito.
Piper TTS registra CPU nesta versão, mesmo quando ASR usa CUDA no mesmo
container.

## Fases do Benchmark

`benchmark_results.throughput`:

- `ops_per_sec`: inferências concluídas por segundo;
- `ops_total`: inferências concluídas na janela;
- `duration_sec`: duração medida;
- `result_metrics`: métricas específicas, como tokens ou duração de áudio;
- `error_count`, `error_rate` e `quality`: falhas observadas.

`benchmark_results.latency`:

- `p50_ms`, `p95_ms` e `p99_ms`: percentis das inferências válidas;
- `mean_ms`, `std_ms`, `min_ms` e `max_ms`: distribuição das amostras;
- `samples_collected`: amostras válidas usadas no cálculo.

`benchmark_results.resource_usage`:

- série temporal durante carga ativa;
- baseline, pico e delta de RAM/VRAM;
- CPU média e tempo acumulado do container;
- utilização, memória, temperatura e potência de GPU;
- energia de GPU e CPU quando NVML ou RAPL estão disponíveis;
- contagem de inferências e duração da mesma janela de recursos;
- fontes de telemetria e escopo de cada métrica.

`benchmark_results.sustained_load` mede variação de throughput ao longo de uma
carga prolongada. `memory_bandwidth` mede cópia de memória do processo e não
deve ser interpretado como largura de banda exclusiva do hardware.

## Escopos

O objeto `resource_usage.scopes` evita atribuir uma medição ao container quando
a fonte só enxerga o dispositivo inteiro.

| Escopo | Significado |
| --- | --- |
| `container_cgroup_working_set` | RAM do cgroup menos cache inativo |
| `container_cgroup` | CPU acumulada no cgroup |
| `process_tree_rss` | fallback de RAM pela árvore de processos |
| `process_tree` | fallback de CPU pela árvore de processos |
| `workload_processes` | telemetria atribuída aos PIDs da carga |
| `visible_devices` | valor agregado dos dispositivos visíveis |
| `device_components` | potência/energia dos componentes medidos |
| `unavailable` | fonte ausente ou sem suporte |
| `legacy_*` | resultado antigo sem contrato de escopo atual |

CPU usa 100% por núcleo ocupado. Uma carga usando seis núcleos integralmente
pode registrar aproximadamente 600%.

## Métricas Primárias

- visão: `frames/s`;
- SLM: `tokens/s`, calculado com os tokens produzidos;
- áudio: múltiplo de tempo real, calculado como segundos de áudio processados
  divididos pelo tempo da janela.

`ops/s` continua disponível para observar taxa de chamadas, mas não substitui a
métrica primária quando a quantidade produzida por chamada varia.

## Scorecard e Custo

O scorecard contém:

- `primary_metric`: métrica usada para score e comparação;
- `performance_score`: valor relativo às referências de
  `config/hardware.yaml`, limitado a 200;
- `efficiency`: energia por operação ou unidade primária, quando mensurável;
- `cost`: amortização e eletricidade por mil unidades primárias;
- `measurement_basis`: fase e escopo usados nos cálculos;
- `warnings`: dados ausentes, legados ou cobertura parcial.

Preço, tarifa de energia e amortização vazios produzem custo ausente, não zero.
O score de performance não depende desses valores financeiros.

## Comparação e Dados Legados

Antes de destacar um vencedor, compare `metadata.comparability_key`. Frente,
modelo, dataset, perfil, modo e metodologia devem representar a mesma carga.
Para comparar modelos, mantenha hardware e execução iguais; para comparar
hardware, mantenha o modelo e a carga iguais.

O parser mantém compatibilidade de leitura com resultados antigos. Campos sem
fonte conhecida são marcados como legados ou indisponíveis e não recebem um
escopo presumido. Arquivos JSON que não possuem frente válida e seções de
benchmark são ignorados pela tela de resultados.
