# Pipeline de Testes da Frota

Como rodar os benchmarks de desempenho e de estresse nos devices (Jetson,
Raspberry Pi, mini PCs) a partir deste PC, e onde os resultados aparecem.

O código roda **no device**; este PC só orquestra por SSH. O device baixa o
projeto **do GitHub com a internet dele** (não pela VPN) e fica travado num
commit exato, então cada resultado fica atrelado a um SHA. Detalhes de
transporte e ARM em [`ARM.md`](ARM.md).

## As duas pipelines

Não são dois programas: são **perfis** do mesmo benchmark. O que separa
desempenho de estresse é a fase `sustained_load` (carga contínua) — quanto
tempo o device fica martelado sem descanso, que é quando aparece throttling
térmico e queda de clock.

| Perfil | sustained_load | Serve para |
| --- | --- | --- |
| `express` | 0 min | fumaça: confirmar que a frente responde |
| `quick` | 0 min | desempenho rápido (padrão) |
| `standard` | 2 min | **desempenho** com carga sustentada curta |
| `extended` | 10 min | **estresse/soak**: throttling, estabilidade térmica, clock |

Perfil também aumenta amostras de latência, duração de throughput e ativa
teste de banda de memória. Um device sem refrigeração decente entrega bem no
`quick` e degrada no `extended` — é esse o ponto do teste de estresse.

## Desempenho

```bash
./pdc remote deploy jetson-nano --ref arm64      # baixa arm64 do GitHub no device
./pdc remote run    jetson-nano --profile standard
```

`run` faz build, up, status e benchmark no device, acompanha o log ao vivo e
traz os resultados. Uma queda de SSH não mata o teste (roda sob `setsid`);
reata com `./pdc remote logs jetson-nano`.

## Estresse

```bash
./pdc remote run jetson-nano --profile extended --detach
```

`--detach` volta ao terminal e deixa o soak de ~10 min por frente rodando
solto. Acompanhe com `./pdc remote logs` e recolha depois com
`./pdc remote fetch`. Numa placa ARM o build inicial (compilar `llama-cpp`) e
o soak juntos passam de uma hora — o `--detach` existe para isso.

## Onde ficam os resultados

Trazidos para **`results/remote/<device>/`** neste PC (padrão do `fetch`, que o
`run` chama sozinho no fim). Um JSON por frente e execução, com:

- throughput (ops/s), latência (p50/p95/p99), score e energia/op;
- inventário físico do device no momento (`hardware_id`);
- telemetria por amostra durante o teste (veja abaixo);
- `thermal_throttling_observed`: se a placa reduziu clock por calor no soak.

## Telemetria durante o teste

O agente `pdc-telemetry` sobe junto com as frentes e amostra o hardware
enquanto o benchmark roda, gravando em `data/telemetry/host.json` no device.
O resultado embute essas amostras. Fontes por plataforma:

- **todas**: CPU/RAM por cgroup do container, frequência, temperatura (hwmon),
  throttling térmico;
- **Jetson**: `tegrastats` — GR3D (GPU), controlador de memória, trilhos de
  potência (`VDD_IN`), RAM unificada;
- **Raspberry Pi**: temperatura e clock; potência só com sensor externo;
- **aceleradores** (NPU/TPU): via JSON de provedor em
  `data/telemetry/providers/` no device.

Se uma fonte não existe na placa, o campo fica **ausente** — nunca zerado.

## Frota

| Device | Usuário | Arq. | Acesso |
| --- | --- | --- | --- |
| `jetson-nano` | `user-jetson` | aarch64 | chave + sudo sem senha |
| `raspberry-pi5` | `helpai` | aarch64 | chave + sudo sem senha |
| `raspberry-pi5-02` | *(a confirmar)* | aarch64 | pendente de credencial |
| `minipc` | *(a confirmar)* | x86_64 | pendente de credencial |

Devices offline no mesh em 11/09/2026: `user-desktop`, `oculink-minipc`,
`werikcyano-dell`, `paulo-windows` (Windows, sem sshd).

## Fluxo de correção

Bug numa frente → corrige **neste PC** → commita e faz push na branch `arm64`
→ `./pdc remote deploy <device> --ref arm64` puxa do GitHub de novo. O device
nunca recebe patch manual; a fonte da verdade é o GitHub.

Para testar algo ainda sem commit: `./pdc remote deploy <device> --from-local`
(envia esta pasta por rsync). O `deploy` avisa quando há trabalho sem commit
que não vai para o device no modo git.
