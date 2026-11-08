#!/usr/bin/env bash
# Campanha de benchmark na frota: N repetições de vários perfis, cada teste
# com sua série de telemetria, tudo trazido para o PC e transformado em
# relatório HTML com gráficos.
#
#   bash campaign.sh <device> [--profiles "quick standard soak"] \
#                    [--reps 3] [--fronts "vision slm audio"] [--out DIR]
#
# Constrói e sobe as frentes UMA vez; cada repetição é só o benchmark, com o
# sampler.py rodando em volta. Não reconstrói a cada rep.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

DEVICE="${1:-}"; shift || true
[ -z "$DEVICE" ] && { echo "uso: bash campaign.sh <device> [--profiles ...] [--reps N] [--fronts ...] [--out DIR]"; exit 1; }

# Perfis com reps por perfil: "quick:3 standard:2 soak:1". Sem ":" usa REPS.
PROFILES="quick:3 standard:2 soak:1"
REPS=3
FRONTS="vision slm audio"
RPATH='~/benchmarks_gustavo'
OUT="$HERE/results"
while [ $# -gt 0 ]; do
  case "$1" in
    --profiles) PROFILES="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --fronts) FRONTS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --path) RPATH="$2"; shift 2;;
    *) echo "flag desconhecida: $1"; exit 1;;
  esac
done

STAMP="$(date +%Y%m%d-%H%M%S)"
CAMP="$OUT/$DEVICE/$STAMP"
mkdir -p "$CAMP"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 $DEVICE"

# Ao sair (fim, erro ou Ctrl-C), mata qualquer sampler orfao no device.
cleanup() { $SSH "pkill -f 'sampler\.py' 2>/dev/null" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "== campanha $STAMP em $DEVICE =="
echo "   perfis: $PROFILES | reps: $REPS | frentes: $FRONTS"
echo "   saida:  $CAMP"

# code + sampler no device (sampler não faz parte do repo; empurro à parte)
scp -q -o BatchMode=yes "$HERE/sampler.py" "$DEVICE:$RPATH/sampler.py" || {
  echo "!! não consegui enviar sampler.py — device deployado?"; exit 1; }

echo "== build + up (uma vez, por frente) =="
for f in $FRONTS; do
  $SSH "cd $RPATH && ./pdc build $f --cpu >/dev/null 2>&1 && ./pdc up $f --cpu" 2>&1 | tail -1 | sed 's/^/   /' || true
done

fail=0
for spec in $PROFILES; do
  profile="${spec%%:*}"
  preps="${spec#*:}"; [ "$preps" = "$profile" ] && preps="$REPS"
  for front in $FRONTS; do
    for rep in $(seq 1 "$preps"); do
      tag="${front}_${profile}_rep${rep}"
      echo "== $tag =="
      # Sampler + benchmark + kill + staging numa UNICA sessao SSH. O sampler
      # roda em background DENTRO do script remoto (o $! capturado aqui é o pid
      # certo), o benchmark segura o canal em foreground, e ao terminar matamos
      # o sampler e montamos o staging. Assim o SSH retorna sozinho — nada de
      # canal preso por processo em background (era a causa da trava).
      out=$($SSH "bash -s" <<REMOTE 2>&1
cd $RPATH || exit 90
PYTHONPATH=$RPATH python3 sampler.py --host-json data/telemetry/host.json \
  --out /tmp/$tag.jsonl --interval 1 --label $tag >/tmp/$tag.samplog 2>&1 &
SAMP=\$!
./pdc benchmark $front --profile $profile
RC=\$?
kill \$SAMP 2>/dev/null; wait \$SAMP 2>/dev/null
mkdir -p camp/$tag
r=\$(ls -t results/${front}*_${profile}_*.json 2>/dev/null | head -1)
[ -n "\$r" ] && cp "\$r" camp/$tag/result.json
cp /tmp/$tag.jsonl camp/$tag/telemetria.jsonl 2>/dev/null
echo "__RC__=\$RC __SAMPLES__=\$(grep -c '\"t\":' /tmp/$tag.jsonl 2>/dev/null)"
REMOTE
)
      echo "$out" | tail -7 | sed 's/^/   /'
      rc=$(echo "$out" | grep -oE "__RC__=[0-9]+" | cut -d= -f2)
      [ "${rc:-1}" != 0 ] && { echo "   !! rc=${rc:-?}"; fail=$((fail+1)); }
    done
  done
done

echo "== trazendo resultados para o PC =="
rsync -az -e "ssh -o BatchMode=yes" "$DEVICE:$RPATH/camp/" "$CAMP/" 2>&1 | tail -2 | sed 's/^/   /'
$SSH "rm -rf $RPATH/camp"   # limpa staging no device

echo "== gerando relatório =="
python3 "$HERE/report.py" "$CAMP" | sed 's/^/   /'
echo "== fim (falhas: $fail) =="
echo "   relatório: $CAMP/report.html"
