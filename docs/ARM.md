# ARM, SBCs e Aceleradores

## Execução em ARM64

As variantes CPU usam imagens base multi-arquitetura e dependências disponíveis
para `linux/arm64`:

- visão: PyTorch CPU, torchvision, Ultralytics e OpenCV;
- SLM: `llama-cpp-python` compilado no dispositivo com OpenBLAS;
- áudio: CTranslate2, ONNX Runtime e Piper.

O fluxo em uma distribuição Linux ARM64 é o mesmo:

```bash
./pdc setup
./pdc models download defaults
./pdc build all --cpu
./pdc up all --cpu
./pdc status
./pdc benchmark all --profile express
```

A compilação do SLM deve ser feita nativamente na placa. Emulação por QEMU é
útil para validar o Dockerfile, mas não representa desempenho nem recursos do
hardware de destino.

## Rodando a partir de outra máquina

`./pdc remote` executa o fluxo acima em um device da frota por SSH, sem instalar
agente nem dependência nova no alvo: usa apenas `ssh` e `rsync`. O nome do device
é o que o `~/.ssh/config` resolve, ou um `usuario@host`.

```bash
./pdc remote check  jetson-nano                    # pré-requisitos do device
./pdc remote deploy jetson-nano                    # envia o código e prepara
./pdc remote run    jetson-nano --profile express  # build, up, benchmark, resultados
./pdc remote logs   jetson-nano                    # reata a uma execução em curso
./pdc remote fetch  jetson-nano                    # traz só os resultados
```

`check` recusa a execução e diz o que falta quando o device não tem Docker, o
plugin `compose`, Python 3.10+, o módulo `venv` ou `rsync`, quando o Docker exige
`sudo`, ou quando a placa é ARM de 32 bits. Em ARM64 o `run` passa `--cpu`
sozinho, já que não existe imagem CUDA para essa arquitetura neste repositório.

O `deploy` não envia `.venv`, `.git`, modelos, datasets nem resultados: o device
cria o próprio ambiente e baixa os modelos por conta (use `--no-models` para
pular). Enviar o `.venv` quebraria de qualquer jeito, por conter caminho absoluto
e binário da arquitetura errada.

O benchmark roda solto do terminal, com `setsid`. Uma queda de SSH não mata o
teste — reata com `./pdc remote logs`. O acompanhamento segue o PID do processo,
então um benchmark morto por falta de memória na placa encerra o log com aviso em
vez de travar esperando. Os resultados chegam em `results/remote/<device>/`.

## Agente de Telemetria

`pdc-telemetry` é iniciado automaticamente por `./pdc up`. A imagem não possui
dependências binárias e usa o mesmo Dockerfile em `amd64`, `arm64` e
`linux/arm/v7`. O agente monta `/sys` e `/proc` como somente leitura e publica a
amostra atual em `data/telemetry/host.json`.

Em ARM de 32 bits, somente o agente de telemetria é suportado. As frentes de
visão, SLM e áudio exigem ARM64 por causa dos runtimes de inferência. Para usar
apenas o coletor:

```bash
mkdir -p data/telemetry/providers
PDC_TEGRASTATS_PATH=/dev/null \
  docker compose -f telemetry/docker-compose.yml up -d --build
```

Fontes consultadas:

| Fonte | Métricas |
| --- | --- |
| cgroup | CPU e RAM de cada container |
| árvore de processos | fallback de CPU e RAM |
| thermal sysfs e hwmon | temperaturas e estado de limitação térmica |
| cpufreq | frequência atual das políticas de CPU |
| devfreq | utilização e frequência de GPU/NPU quando o driver expõe contadores |
| hwmon | potência e energia quando existe sensor compatível |
| `tegrastats` | GPU, EMC, temperatura e trilhos de potência em Jetson |
| provedor externo | utilização, memória, potência e temperatura de aceleradores |

Uma amostra com mais de cinco segundos é ignorada. O benchmark mantém o campo
ausente quando uma fonte para aquela métrica não existe.

## Raspberry Pi

Em Raspberry Pi com sistema operacional de 64 bits, os benchmarks CPU coletam
latência, throughput, CPU, RAM, frequência e temperatura. As placas não possuem
um medidor geral de potência acessível pelo kernel; potência e energia só
aparecem quando um sensor externo, HAT ou driver hwmon fornece a leitura.

GPU integrada usa memória compartilhada. Esse uso não é registrado como VRAM.
Se o driver publicar devfreq, a utilização aparece com escopo
`visible_devices`.

## NVIDIA Jetson

O CLI detecta `/usr/bin/tegrastats` no host e o monta no agente. Quando a
ferramenta está disponível, o resultado pode incluir:

- uso e frequência de GR3D;
- uso e frequência do controlador de memória;
- temperaturas reportadas pela plataforma;
- potência de entrada da placa, quando existe um trilho como `VDD_IN` ou
  `VIN_SYS_5V0`;
- RAM unificada do sistema.

A variante `--gpu` deste repositório usa imagens CUDA para NVIDIA dGPU em
`x86_64` e é bloqueada em ARM. JetPack exige uma imagem e versões de PyTorch,
CUDA, TensorRT e CTranslate2 compatíveis com a versão L4T instalada. A variante
CPU funciona no Jetson, mas não deve ser descrita como teste da GPU.

## NPUs, TPUs e Aceleradores Externos

O inventário reconhece dispositivos PCI, USB, platform bus e nós como
`/dev/hailo*`, `/dev/apex_*`, `/dev/rknpu*` e `/dev/accel/accel*`. Presença no
inventário não comprova execução.

Um runtime de acelerador precisa fornecer:

1. carregamento do modelo e inferência na frente correspondente;
2. `metadata.execution.backend` e `metadata.execution.device` corretos;
3. telemetria do driver, quando disponível.

Provedores externos podem atualizar um JSON em
`data/telemetry/providers/<nome>.json`. O agente aceita o contrato abaixo:

```json
{
  "schema_version": "accelerator-telemetry-2026.08-v1",
  "provider": "runtime-do-fabricante",
  "timestamp_epoch": 1785900000.0,
  "accelerators": [
    {
      "id": "npu0",
      "kind": "npu",
      "name": "Nome do dispositivo",
      "utilization_pct": 72.5,
      "memory_used_mb": 128.0,
      "temperature_c": 54.0,
      "power_w": 2.8,
      "scope": "visible_devices"
    }
  ]
}
```

O arquivo deve ser atualizado durante o teste. Campos não fornecidos pelo
driver devem ser omitidos, não preenchidos com zero. O dashboard mostra o
provedor e o escopo junto da métrica.

## Interpretação

- throughput e latência pertencem à carga executada pelo backend registrado;
- CPU e RAM pertencem ao cgroup do container;
- NVML pode atribuir GPU e VRAM aos PIDs da carga em NVIDIA dGPU;
- devfreq, `tegrastats` e a maioria dos runtimes de NPU descrevem o dispositivo
  inteiro;
- potência de entrada da placa prevalece sobre a soma de componentes para
  evitar dupla contagem;
- limitação térmica ativa durante a fase é registrada no resultado;
- memória unificada não é VRAM dedicada.

## Referências

- [Docker: builds multi-plataforma](https://docs.docker.com/build/building/multi-platform/)
- [NVIDIA Jetson: tegrastats](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
- [Kernel Linux: interface hwmon](https://docs.kernel.org/hwmon/sysfs-interface.html)
- [Raspberry Pi: temperatura e refrigeração](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
