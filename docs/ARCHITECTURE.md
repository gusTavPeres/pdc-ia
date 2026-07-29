# Arquitetura e Suporte de Hardware

## Componentes

```text
pdc (CLI) ───────────────┐
                         ├─ Docker Compose
dashboard/API :8099 ─────┤
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
5. `MetricsCollector` amostra cgroup, processos, NVML, sensores térmicos e RAPL
   quando essas fontes existem.
6. O resultado recebe metodologia, contexto de execução, scorecard e um
   `run_id`, sendo gravado atomicamente em `results/`.
7. O dashboard normaliza resultados atuais e legados para exibição.

## Variantes de Imagem

- `cpu`: não exige runtime de GPU e é a base portátil;
- `gpu`: usa os Dockerfiles e overrides NVIDIA de cada frente;
- seleção automática: só usa `gpu` quando `nvidia-smi` responde e o runtime
  NVIDIA está disponível;
- seleção explícita: `./pdc build|up ... --cpu` ou `--gpu`.

As imagens são construídas nativamente no host. O projeto não publica, nesta
versão, um manifesto multi-arquitetura pronto.

## Matriz de Suporte

| Hardware | Inventário | Execução de carga | Telemetria dedicada |
| --- | --- | --- | --- |
| CPU x86_64 | sim | sim | cgroup, processos, térmica e RAPL quando disponível |
| CPU ARM64 | sim | alvo da imagem CPU; validar por placa | cgroup, processos e sensores expostos pelo kernel |
| GPU NVIDIA | sim | Vision, SLM e ASR na variante CUDA; TTS em CPU | NVML/`nvidia-smi` |
| GPU Intel/AMD | parcial | CPU apenas nesta versão | sensores genéricos quando expostos; sem OpenVINO/ROCm |
| NPU/TPU PCI ou device node | sim, quando identificável | não | não |
| Acelerador USB para SBC | parcial, conforme device node | não | não |
| Interface serial/MCU | sim | sem upload ou benchmark de firmware | não |

“Inventário” significa presença detectada no host. O resultado só atribui uma
carga a um acelerador quando `metadata.execution` registra o backend e o device
utilizados.

## Limites da Abstração

A portabilidade atual cobre o ciclo comum de benchmark e os backends CPU/CUDA.
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
- controles de container no dashboard dependem de acesso local ao daemon
  Docker;
- não há autenticação HTTP, portanto a exposição em rede deve ser feita por
  túnel SSH ou por uma camada externa de autenticação.
