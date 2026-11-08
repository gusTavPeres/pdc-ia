#!/usr/bin/env bash
# Serie de testes de ~2h num device: varredura de modelos por frente
# (desempenho) + estresse (soak), com time-box. Troca o modelo carregado no
# container antes de cada bloco e mede com o sampler.
#
#   bash campaign6h.sh <device> [--minutes 120] [--path ~/benchmarks_gustavo]
#
# Reusa sampler.py (telemetria) e report.py (relatorio). Cada teste fica
# etiquetado com a frente, o modelo e o perfil.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

DEVICE="${1:-}"; shift || true
[ -z "$DEVICE" ] && { echo "uso: bash campaign6h.sh <device> [--minutes N]"; exit 1; }
MINUTES=360
RPATH='~/benchmarks_gustavo'
OUT="$HERE/results"
while [ $# -gt 0 ]; do
  case "$1" in
    --minutes) MINUTES="$2"; shift 2;;
    --path) RPATH="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "flag desconhecida: $1"; exit 1;;
  esac
done

STAMP="$(date +%Y%m%d-%H%M%S)"
CAMP="$OUT/$DEVICE/2h-$STAMP"
mkdir -p "$CAMP"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 $DEVICE"
DEADLINE=$(( $(date +%s) + MINUTES*60 ))

cleanup() { $SSH "pkill -f 'sampler\.py' 2>/dev/null" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# modelos por frente (diversos, viaveis em ARM; portas fixas por frente)
VISION_MODELS="yolov8n yolo11n yolo11s"
SLM_MODELS="qwen25-0.5b-q4 tinyllama-1.1b-q4 gemma3-1b-q4 gemma3-1b-q8 qwen3-4b-q4"
AUDIO_MODELS="whisper-tiny whisper-base whisper-small whisper-medium"
declare -A PORT=( [vision]=8010 [slm]=8011 [audio]=8012 )
declare -A DLNAME=( [vision]=vision [slm]=slm [audio]=audio_asr )

log() { echo "[$(date +%H:%M:%S)] $*"; }
time_left() { [ "$(date +%s)" -lt "$DEADLINE" ]; }
mins_left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }

echo "== serie 2h em $DEVICE ($STAMP) =="
echo "   ate: $(date -d @$DEADLINE +%H:%M) | saida: $CAMP"

# build+up uma vez por frente
for f in vision slm audio; do
  $SSH "cd $RPATH && ./pdc build $f --cpu >/dev/null 2>&1 && ./pdc up $f --cpu" 2>&1 | tail -1 | sed 's/^/   /' || true
done

# baixa e troca o modelo de uma frente; espera recarregar
switch_model() {
  local front="$1" model="$2" port="${PORT[$1]}" dl="${DLNAME[$1]}"
  $SSH "cd $RPATH && ./pdc models download $dl $model >/dev/null 2>&1" || true
  local body
  if [ "$front" = audio ]; then body="{\"target\":\"asr\",\"model\":\"$model\"}"; else body="{\"model\":\"$model\"}"; fi
  # troca com timeout generoso (carregar gguf grande demora); confirma via /health
  $SSH "curl -s -m 600 -X POST -H 'Content-Type: application/json' -d '$body' http://localhost:$port/switch-model >/dev/null 2>&1" || true
  # espera ficar saudavel de novo
  for _ in $(seq 1 40); do
    $SSH "curl -sf -m 5 http://localhost:$port/health >/dev/null 2>&1" && break
    sleep 3
  done
}

# roda 1 teste: sampler + benchmark + staging, numa unica sessao SSH
run_test() {
  local front="$1" model="$2" profile="$3" tag="${1}_${2}_${3}"
  log "-> $tag ($(mins_left)min restantes)"
  local out
  out=$($SSH "bash -s" <<REMOTE 2>&1
cd $RPATH || exit 90
PYTHONPATH=$RPATH python3 sampler.py --host-json data/telemetry/host.json --out /tmp/$tag.jsonl --interval 1 --label $tag >/tmp/$tag.samplog 2>&1 &
SAMP=\$!
./pdc benchmark $front --profile $profile
RC=\$?
kill \$SAMP 2>/dev/null; wait \$SAMP 2>/dev/null
mkdir -p camp/$tag
r=\$(ls -t results/${front}*_${profile}_*.json 2>/dev/null | head -1)
[ -n "\$r" ] && cp "\$r" camp/$tag/result.json
cp /tmp/$tag.jsonl camp/$tag/telemetria.jsonl 2>/dev/null
echo "__RC__=\$RC"
REMOTE
)
  echo "$out" | grep -E "Throughput|Métrica|__RC__|falhou" | sed 's/^/     /'
}

# ---- FASE 1: por frente, sweep de modelos (perf) + soak do default (estresse) ----
# Intercalado para GARANTIR estresse (no 2h o soak foi pulado por falta de tempo).
echo "== fase 1: desempenho por modelo + soak por frente =="
declare -A DEFAULT=( [vision]=yolov8n [slm]=tinyllama-1.1b-q4 [audio]=whisper-small )
for front in vision audio slm; do
  models_var="$(echo ${front}_MODELS | tr a-z A-Z)"; eval "models=\$$models_var"
  for model in $models; do
    time_left || break 2
    switch_model "$front" "$model"
    run_test "$front" "$model" quick
    time_left || break 2
    run_test "$front" "$model" standard
  done
  # soak do modelo default logo apos o sweep da frente (>=32min de folga)
  if time_left && [ "$(mins_left)" -ge 32 ]; then
    switch_model "$front" "${DEFAULT[$front]}"
    run_test "$front" "${DEFAULT[$front]}" soak
  fi
done

# ---- FASE 2: torture (60min) nas configs de referencia, enquanto houver folga ----
echo "== fase 2: torture (estresse pesado 60min) =="
for front in slm vision; do
  time_left || break
  [ "$(mins_left)" -ge 65 ] || { log "sem tempo para torture de $front"; continue; }
  switch_model "$front" "${DEFAULT[$front]}"
  run_test "$front" "${DEFAULT[$front]}" torture
done

# ---- FASE 3: repete perf (robustez) ate estourar o tempo ----
rep=2
while time_left; do
  echo "== fase 3: repeticao extra $rep =="
  for front in vision audio slm; do
    models_var="$(echo ${front}_MODELS | tr a-z A-Z)"; eval "models=\$$models_var"
    for model in $models; do
      time_left || break 2
      switch_model "$front" "$model"
      run_test "$front" "$model" quick
    done
  done
  rep=$((rep+1))
done

echo "== trazendo resultados =="
rsync -az -e "ssh -o BatchMode=yes" "$DEVICE:$RPATH/camp/" "$CAMP/" 2>&1 | tail -1 | sed 's/^/   /'
$SSH "rm -rf $RPATH/camp"
echo "== gerando relatorio =="
python3 "$HERE/report.py" "$CAMP" | sed 's/^/   /'
echo "== fim 2h em $DEVICE =="
