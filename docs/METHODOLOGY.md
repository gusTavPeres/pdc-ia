# Metodologia

Esta metodologia define como comparar execuções feitas em hardwares diferentes.

## Regra de Comparação

Compare apenas resultados com a mesma combinação de:

- frente: `vision`, `slm` ou `audio`;
- modelo e quantização;
- perfil de benchmark;
- conjunto de entradas;
- modo, quando existir: `asr`, `tts` ou `pipeline`;
- política de execução: CPU-only, GPU, NPU ou outro acelerador.

O campo `metadata.comparability_key` registra esses pontos nos resultados novos.

## Contrato do Resultado

Cada resultado novo deve conter:

- `methodology.version`;
- `metadata`;
- `benchmark_results.throughput`;
- `benchmark_results.latency`;
- `benchmark_results.resource_usage`;
- `scorecard`, quando houver dados suficientes.

Falhas não são descartadas. Erros de inferência entram nos contadores de qualidade e aparecem no resultado.

## Métricas Brutas

Throughput:

- número de inferências concluídas por segundo;
- em SLM, `tokens_per_sec` pode ser usado como métrica primária.

Latência:

- `p50_ms`: mediana;
- `p95_ms`: percentil 95;
- `p99_ms`: percentil 99;
- `std_ms`: desvio padrão das amostras.

Recursos:

- CPU e RAM do host;
- GPU, VRAM, temperatura e potência quando disponíveis;
- séries temporais quando o perfil coleta uso de recursos.

## Scorecard

O scorecard não substitui os dados brutos. Ele resume:

- métrica primária;
- score de performance;
- energia total;
- energia por operação;
- custo por 1k unidades primárias;
- avisos de dados parciais.

As referências ficam em `config/hardware.yaml`. Para uma campanha de comparação, mantenha essas referências fixas.

## Energia e Custo

Energia e custo são estimativas quando não há medição externa.

Fontes aceitas:

- GPU NVIDIA via NVML/`nvidia-smi`;
- CPU via RAPL/powercap quando disponível;
- medidor externo registrado manualmente.

Quando potência não está disponível, throughput e latência continuam válidos, mas custo e eficiência ficam parciais.

## Campanha Recomendada

1. Fixar revisão do repositório.
2. Fixar `config/models.yaml`, `config/benchmark.yaml` e `config/hardware.yaml`.
3. Rodar `./pdc detect --save`.
4. Baixar modelos com `./pdc models download defaults` ou `all`.
5. Aquecer a máquina com um perfil `express`.
6. Rodar pelo menos três execuções `standard` por frente.
7. Usar `extended` para verificar temperatura e degradação.
8. Comparar medianas ou melhor de três apenas depois de checar erro e avisos.

## Ordem de Leitura

1. Taxa de erro e avisos.
2. Chave de comparabilidade.
3. Métrica primária.
4. Latência p95/p99.
5. Degradação em carga sustentada.
6. Energia e custo, se a telemetria permitir.
