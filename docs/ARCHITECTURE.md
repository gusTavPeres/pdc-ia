# Arquitetura e Suporte de Hardware

## Componentes

```text
pdc (CLI) ───────────────┐
                         ├─ Docker Compose
dashboard/API :8099 ─────┤
                         ├─ telemetry (host sensors)
                         ├─ vision :8010
                         ├─ slm    :8011
                         └─ audio  :8012

config/  -> configuração montada como somente leitura nos containers
data/    -> modelos, inventário e cache local
results/ -> resultados JSON
shared/  -> coleta, metodologia, persistência e scorecard
```

O CLI cuida de setup, build, ciclo de vida dos containers, download de modelos
e execução de benchmarks. O dashboard oferece as mesmas operações principais e
faz chamadas HTTP aos três serviços.

Cada frente carrega um modelo por vez e expõe `health`, `info`, inferência,
troca de modelo e benchmark. A lógica comum fica em `shared/` para que duração,
latência, coleta de recursos e formato de saída sejam iguais entre as frentes.

## Fluxo de Execução

1. O host detecta o hardware e grava `data/hardware/host.json`.
2. O container detecta os recursos que realmente consegue acessar.
3. `shared.capabilities` combina o inventário físico com a visibilidade do
   runtime sem substituir um pelo outro.
4. A frente executa aquecimento e as fases habilitadas no perfil.
5. `pdc-telemetry` publica sensores Linux do host e `tegrastats` em Jetson.
6. `MetricsCollector` amostra cgroup, processos, NVML, RAPL e a telemetria do
   host quando essas fontes existem.
7. O resultado recebe metodologia, contexto de execução, scorecard e um
   `run_id`, sendo gravado atomicamente em `results/`.
8. O dashboard normaliza resultados atuais e legados para exibição.

## Variantes de Imagem

- `cpu`: não exige runtime de GPU, possui dependências para `amd64` e `arm64` e
  é construída nativamente no host;
- `gpu`: usa os Dockerfiles e overrides de NVIDIA dGPU em `x86_64`;
- `telemetry`: imagem sem dependências binárias, comum a Linux `amd64`,
  `arm64` e `arm/v7`;
- seleção automática: só usa `gpu` quando `nvidia-smi` responde e o runtime
  NVIDIA está disponível;
- seleção explícita: `./pdc build|up ... --cpu` ou `--gpu`.

As imagens são construídas nativamente no host. O projeto não publica, nesta
versão, um manifesto multi-arquitetura pronto.

## Matriz de Suporte

| Hardware | Inventário | Execução de carga | Telemetria dedicada |
| --- | --- | --- | --- |
| CPU x86_64 | sim | sim | cgroup, processos, térmica e RAPL quando disponível |
| CPU ARM64 | sim | sim na imagem CPU; validar nativamente por placa | cgroup, processos, thermal, cpufreq e hwmon |
| CPU ARMv7 | sim pelo agente | frentes de IA não suportadas | thermal, cpufreq, devfreq e hwmon pelo agente |
| GPU NVIDIA | sim | Vision, SLM e ASR na variante CUDA; TTS em CPU | NVML/`nvidia-smi` |
| NVIDIA Jetson | sim | CPU; CUDA requer variante compatível com JetPack | sysfs e `tegrastats` |
| GPU Intel/AMD | parcial | CPU apenas nesta versão | sensores genéricos quando expostos; sem OpenVINO/ROCm |
| NPU/TPU PCI, USB, platform bus ou device node | sim, quando identificável | não | devfreq ou provedor externo |
| Acelerador USB para SBC | sim para identificações conhecidas | não | provedor externo |
| Interface serial/MCU | sim | sem upload ou benchmark de firmware | não |

“Inventário” significa presença detectada no host. O resultado só atribui uma
carga a um acelerador quando `metadata.execution` registra o backend e o device
utilizados.

## Limites da Abstração

A portabilidade atual cobre o ciclo comum de benchmark, os backends CPU e CUDA
dGPU e a coleta genérica de telemetria do host. A memória integrada de uma SBC
não é registrada como VRAM dedicada.
Adicionar um runtime de NPU requer três integrações explícitas:

1. carregador e função de inferência da frente;
2. metadados de execução com backend, device e limites;
3. coletor de telemetria que declare unidade, fonte e escopo.

Não é suficiente expor `/dev` ao container: o backend precisa comprovar que a
inferência foi executada no acelerador.

## Isolamento e Acesso

- portas de serviço e dashboard são publicadas em `127.0.0.1`;
- `config/` e `shared/` são montados como somente leitura nos containers;
- `data/`, `results/` e `/dev/shm` são graváveis;
- `/sys` é montado como somente leitura para inventário e cgroup;
- o agente monta `/sys` e `/proc` do host como somente leitura e mantém o
  filesystem raiz do container também somente leitura;
- controles de container no dashboard dependem de acesso local ao daemon
  Docker;
- não há autenticação HTTP, portanto a exposição em rede deve ser feita por
  túnel SSH ou por uma camada externa de autenticação.
