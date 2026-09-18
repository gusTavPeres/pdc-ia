# Testes de Estresse — cenários por device

Um benchmark rápido diz a performance de pico. O **teste de estresse** diz se o
device *sustenta* essa performance sob carga contínua — ou se esquenta, entra em
**throttling** (reduz o clock) e cai. Isso muda a adequação do device: uma placa
que throttla pode servir para inferência intermitente, mas não para carga 24/7.

## O que se mede

- **degradação (%)**: quanto o throughput caiu do início ao fim da carga sustentada.
- **temperatura máxima**: pico durante o teste.
- **throttling**: se a placa reduziu o clock por calor (sinal claro de que não sustenta).

Device que sustenta: degradação ~0, sem throttling. Device que não sustenta:
degradação alta e/ou throttling — como o Raspberry Pi 5, que a 87 °C com a
ventoinha no máximo entrou em throttling no SLM sustentado (throughput caiu ~36%).
O Jetson Orin, a 56 °C, sustentou sem throttle.

## Perfis prontos

| Perfil | Carga sustentada | Uso |
| --- | --- | --- |
| **Básico** | 5 min | toque rápido de estabilidade |
| **Médio** | 20 min | uso moderado sustentado |
| **Extremo** | 60 min | pior caso térmico |

Também há `soak` (30 min) e `torture` (60 min) para as campanhas automatizadas.

## Lógicas de teste (aba Estresse do dashboard)

1. **Sustentado (uma frente)** — carga contínua numa frente (visão, SLM ou áudio).
   Revela o limite térmico daquela carga específica.
2. **Concorrente (todas as frentes)** — visão, SLM e áudio ao mesmo tempo,
   estressando CPU, GPU e memória juntos. É o cenário mais duro para o
   resfriamento e o que melhor separa um device que sustenta de um que não.

Customização: escolha a lógica, a frente e a **duração** (1–720 min). O perfil
customizado é gravado no `benchmark.yaml` e lido na hora (o container recarrega o
config a cada benchmark, sem reiniciar nada).

## Ideias de cenários futuros

- **Ciclos térmicos**: alternar carga e ocioso para medir recuperação térmica.
- **Rampa**: subir a carga em degraus até achar o ponto de throttling.
- **Bateria**: correlacionar com consumo (VDD_IN no Jetson) para ver eficiência sob calor.
