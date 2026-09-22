#!/usr/bin/env bash
# Runner de batches de teste que roda NO PROPRIO device (sem SSH), disparavel
# pela plataforma (endpoints /api/batch do dashboard). Escreve batch.status
# (JSON) e batch.pid para o dashboard mostrar progresso e permitir parar.
#
#   bash run-batch.sh --mode stress|perf|mixed|concorrente --minutes N
#
# Nao sobe containers (ja estao up). Resultados caem em results/ e o dashboard
# le direto (aba Estresse -> Graficos). Roda sob setsid: parar = matar o grupo.
set -uo pipefail
cd "$(dirname "$0")"

MODE=stress; MINUTES=120
while [ $# -gt 0 ]; do case "$1" in
  --mode) MODE="$2"; shift 2;;
  --minutes) MINUTES="$2"; shift 2;;
  *) echo "flag desconhecida: $1"; exit 1;;
esac; done

DEADLINE=$(( $(date +%s) + MINUTES*60 ))
START=$(date +%s)
STATUS=batch.status; PIDF=batch.pid
echo $$ > "$PIDF"

# modelos por frente (todos; no Jetson a imagem SLM nova roda gemma3/qwen3 na GPU)
VISION_MODELS="yolov8n yolo11n yolo11s"
SLM_MODELS="qwen25-0.5b-q4 tinyllama-1.1b-q4 gemma3-1b-q4 gemma3-1b-q8 qwen3-4b-q4"
AUDIO_MODELS="whisper-tiny whisper-base whisper-small whisper-medium"
declare -A DLNAME=( [vision]=vision [slm]=slm [audio]=audio_asr )
declare -A PORT=( [vision]=8010 [slm]=8011 [audio]=8012 )
declare -A DEF=( [vision]=yolov8n [slm]=tinyllama-1.1b-q4 [audio]=whisper-small )

COUNT=0; ERRS=0; CUR="-"; ROUND=1
time_left(){ [ "$(date +%s)" -lt "$DEADLINE" ]; }
mins_left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
temp_now(){ for z in /sys/class/thermal/thermal_zone*/temp; do v=$(cat "$z" 2>/dev/null); [ "${v:-0}" -gt 1000 ] 2>/dev/null && echo $((v/1000)); done | sort -rn | head -1; }
write_status(){ # $1 = running(bool)
  local t; t=$(temp_now)
  cat > "$STATUS.tmp" <<JSON
{"running": $1, "mode": "$MODE", "started": $START, "deadline": $DEADLINE, "mins_left": $(mins_left), "round": $ROUND, "count": $COUNT, "errors": $ERRS, "current": "$CUR", "temp_c": "${t:-}", "updated": $(date +%s)}
JSON
  mv -f "$STATUS.tmp" "$STATUS"
}
cleanup(){ CUR="-"; write_status false; rm -f "$PIDF"; }
trap cleanup EXIT INT TERM
write_status true

cooldown(){ local lim="${1:-70}"; for _ in $(seq 1 45); do local t; t=$(temp_now); [ -z "$t" ] && break; [ "$t" -le "$lim" ] 2>/dev/null && break; sleep 2; done; }

switch_model(){
  local front="$1" model="$2" port="${PORT[$1]}" dl="${DLNAME[$1]}" body
  ./pdc models download "$dl" "$model" >/dev/null 2>&1 || true
  if [ "$front" = audio ]; then body="{\"target\":\"asr\",\"model\":\"$model\"}"; else body="{\"model\":\"$model\"}"; fi
  curl -s -m 600 -X POST -H 'Content-Type: application/json' -d "$body" "http://localhost:$port/switch-model" >/dev/null 2>&1 || true
  for _ in $(seq 1 40); do curl -sf -m5 "http://localhost:$port/health" >/dev/null 2>&1 && break; sleep 3; done
}

run_test(){
  local front="$1" model="$2" profile="$3"
  case "$profile" in quick|standard) cooldown 70;; esac  # perf frio; estresse quente de proposito
  CUR="$front/$model/$profile"; write_status true
  ./pdc benchmark "$front" --profile "$profile" >/dev/null 2>&1
  local rc=$?
  COUNT=$((COUNT+1)); [ $rc -ne 0 ] && ERRS=$((ERRS+1))
  echo "[$(date +%H:%M:%S)] $CUR rc=$rc ($(mins_left)min restantes)"
  write_status true
}

models_of(){ case "$1" in vision) echo "$VISION_MODELS";; slm) echo "$SLM_MODELS";; audio) echo "$AUDIO_MODELS";; esac; }

sweep_stress(){ local prof="$1"; for front in vision slm audio; do for model in $(models_of $front); do time_left || return; switch_model "$front" "$model"; run_test "$front" "$model" "$prof"; done; done; }
sweep_concorrente(){ # estressa as 3 frentes AO MESMO TEMPO (pior caso termico do device)
  local prof="$1"
  for front in vision slm audio; do switch_model "$front" "${DEF[$front]}"; done
  CUR="concorrente(3 frentes)/$prof"; write_status true
  local pids=() rc=0
  for front in vision slm audio; do ( ./pdc benchmark "$front" --profile "$prof" >/dev/null 2>&1 ) & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  COUNT=$((COUNT+3)); [ $rc -ne 0 ] && ERRS=$((ERRS+1))
  echo "[$(date +%H:%M:%S)] concorrente/$prof rc=$rc ($(mins_left)min restantes)"
  write_status true
}
sweep_perf(){ for front in vision slm audio; do for model in $(models_of $front); do time_left || return; switch_model "$front" "$model"; run_test "$front" "$model" quick; time_left || return; run_test "$front" "$model" standard; done; done; }
deep_stress(){ for front in slm vision audio; do time_left || return; [ "$(mins_left)" -ge 65 ] || return; switch_model "$front" "${DEF[$front]}"; run_test "$front" "${DEF[$front]}" torture; done; }

echo "== batch $MODE por ${MINUTES}min (ate $(date -d @$DEADLINE +%H:%M)) =="
while time_left; do
  echo "== rodada $ROUND ($MODE) =="
  case "$MODE" in
    stress) sweep_stress estresse_medio; [ $((ROUND % 3)) -eq 0 ] && deep_stress;;
    perf)   sweep_perf;;
    mixed)  sweep_perf; sweep_stress estresse_medio;;
    concorrente) sweep_concorrente estresse_medio;;
    *) echo "modo invalido: $MODE"; exit 2;;
  esac
  ROUND=$((ROUND+1))
done
echo "== fim batch $MODE: $COUNT testes, $ERRS erros =="
