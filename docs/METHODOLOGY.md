# Metodologia

Esta metodologia define como comparar execuções feitas em hardwares diferentes.

## Regra de Comparação

Para comparar hardware, mantenha iguais:

- frente: `vision`, `slm` ou `audio`;
- modelo e quantização;
- perfil de benchmark;
- conjunto de entradas;
- modo, quando existir: `asr`, `tts` ou `pipeline`;
- versão da metodologia.

Hardware, arquitetura, acelerador e política de execução são as variáveis dessa
comparação e precisam estar registrados no resultado. Para comparar modelos,
mantenha hardware e execução iguais e altere apenas o modelo. Não use ranking
direto quando carga e ambiente mudam ao mesmo tempo.

Resultados atuais identificam a bancada por `system_info.provenance.hardware_id`
e `metadata.hardware_profile.id`. O primeiro é derivado do inventário físico; o
segundo é o nome definido pelo operador. `system_info.runtime_visibility`
registra o que o container conseguiu acessar. Essa separação evita classificar
o `overlay` do Docker como disco físico ou assumir que um acelerador detectado
no barramento estava disponível para a carga.

Os campos `metadata.comparability_key` e `metadata.execution` registram esse
contexto nos resultados novos. O dashboard bloqueia o destaque de vencedor
quando os dados necessários estão ausentes ou incompatíveis.

## Contrato do Resultado

Cada resultado novo deve conter:

- `schema_version` e `run_id`;
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

- RAM da carga pelo working set do cgroup do container;
- baseline de RAM e delta entre baseline e pico;
- RSS da árvore de processos e RAM do host como campos separados;
- CPU do cgroup calculada pela diferença do tempo acumulado de CPU na janela, com fallback para a árvore de processos;
- utilização de GPU atribuída aos PIDs da carga quando a NVML oferece essa amostragem, com fallback para o dispositivo visível;
- VRAM atribuída aos processos da carga quando a API do dispositivo fornece essa informação;
- utilização de NPU, TPU ou VPU por devfreq ou provedor do runtime;
- temperatura, potência e séries temporais quando disponíveis.

CPU usa 100% por núcleo ocupado e pode ultrapassar 100% em cargas paralelas.
Potência, temperatura e RAPL descrevem componentes físicos, não uma parcela
exclusiva do contêiner. Para uma comparação isolada, execute uma frente por vez
e mantenha o host sem outras cargas relevantes.

`benchmark_results.resource_usage.scopes` informa o alcance de cada campo. Em
resultados legados sem esse registro, o dashboard marca o escopo como não
identificado em vez de assumir que o valor pertence ao container.

Os limites registrados em `metadata.execution` e `resource_usage` são os
valores efetivos do cgroup. Eles podem incluir políticas herdadas do serviço ou
do host mesmo quando o Docker não recebeu um limite explícito.

## Scorecard

O scorecard não substitui os dados brutos. Ele resume:

- métrica primária;
- score de performance;
- energia total;
- energia por operação;
- custo por 1k unidades primárias;
- avisos de dados parciais.

As referências ficam em `config/hardware.yaml`. Para uma campanha de comparação, mantenha essas referências fixas.

Performance usa a fase de throughput. Energia por operação, eficiência por
watt e custo por unidade usam contadores e duração da fase de recursos. Dados
de fases diferentes não são divididos entre si.

## Energia e Custo

Energia cobre apenas os componentes com telemetria disponível. Custo é sempre
um valor derivado da configuração da bancada.

Fontes aceitas:

- GPU NVIDIA via NVML/`nvidia-smi`;
- CPU via RAPL/powercap quando disponível;
- entrada da placa e componentes via hwmon;
- trilhos de potência Jetson via `tegrastats`;
- aceleradores por provedores externos com amostras atuais.

Medidores externos devem usar o contrato de provedor descrito em
`docs/ARM.md`; seus valores não devem ser inseridos como se fossem telemetria de
CPU ou GPU.

Quando potência não está disponível, throughput e latência continuam válidos, mas custo e eficiência ficam parciais.

Potência NVML representa os dispositivos GPU visíveis. RAPL representa os
pacotes de CPU acessíveis. Quando as duas fontes existem, a energia total é a
soma dos componentes medidos, não uma medição completa na tomada. A energia de
GPU é integrada pela série temporal; contadores RAPL usam a diferença acumulada
na própria janela.

Quando um sensor mede a entrada da placa, essa fonte prevalece no total. Os
componentes contidos nessa leitura não são somados novamente.

`energy_coverage_ratio` informa qual parte da janela possui telemetria contínua.
Intervalos ausentes não são preenchidos. Quando a cobertura é baixa, qualquer
extrapolação aparece como estimativa no scorecard.

## Campanha Recomendada

1. Fixar revisão do repositório.
2. Fixar `config/models.yaml`, `config/benchmark.yaml` e `config/hardware.yaml`.
3. Rodar `./pdc detect --save`.
4. Baixar modelos com `./pdc models download defaults` ou `all`.
5. Aquecer a máquina com um perfil `express`.
6. Rodar pelo menos três execuções `standard` por frente.
7. Usar `extended` para verificar temperatura e degradação.
8. Comparar a mediana das execuções apenas depois de checar erros e avisos.

## Ordem de Leitura

1. Taxa de erro e avisos.
2. Chave de comparabilidade.
3. Métrica primária.
4. Latência p95/p99.
5. Degradação em carga sustentada.
6. Energia e custo, se a telemetria permitir.
