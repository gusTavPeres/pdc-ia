#!/usr/bin/env python3
"""Amostrador de telemetria — série temporal durante um teste.

Roda NO device. NÃO coleta sozinho: lê o data/telemetry/host.json que o agente
pdc-telemetry já atualiza a cada intervalo. É lá que estão os dados ricos e
consistentes do Jetson (tegrastats): CPU util normalizada 0-100, GPU, trilhos
de potência (VDD_IN), temperaturas rotuladas, throttling. Coletar direto no
host, fora do container, perde tudo isso — foi o bug da 1ª versão.

    python3 sampler.py --host-json data/telemetry/host.json --out serie.jsonl --interval 1 &

Grava uma linha JSON por amostra (JSONL); report.py no PC transforma em gráfico.
Marca amostra como stale se o host.json parou de ser atualizado (agente caiu).
"""
import argparse
import json
import os
import signal
import sys
import time

_stop = False


def _handle(_signum, _frame):
    global _stop
    _stop = True


def _num(v):
    return v if isinstance(v, (int, float)) else None


def _flat_from_hostjson(path, t0):
    """Achata o host.json do agente nos campos que viram curva."""
    try:
        d = json.load(open(path))
    except Exception as exc:  # noqa: BLE001
        return {"t": round(time.time() - t0, 2), "error": f"host.json ilegível: {exc}"}

    temps = [_num(s.get("temperature_c")) for s in (d.get("temperatures") or [])]
    temps = [t for t in temps if t is not None]
    cpu = d.get("cpu") or {}
    gpu = d.get("gpu") or {}
    throt = d.get("throttling") or {}
    power = d.get("power") or {}
    # potência total: trilho de entrada da placa (VDD_IN) tem prioridade sobre a
    # soma de componentes, para não contar duas vezes. Cai p/ sensores hwmon.
    rails = power.get("rails") or []
    vdd_in = next((_num(r.get("power_w")) for r in rails
                   if str(r.get("label", "")).upper().startswith("VDD_IN")), None)
    if vdd_in is None:
        watts = [_num(r.get("power_w")) for r in rails] + \
                [_num(s.get("power_w")) for s in (power.get("sensors") or [])]
        watts = [w for w in watts if w is not None]
        vdd_in = round(sum(watts), 3) if watts else None

    # frescor: o benchmark ignora amostra com mais de 5s; sinalizamos igual.
    ts_epoch = _num(d.get("timestamp_epoch"))
    stale = (time.time() - ts_epoch) > 5 if ts_epoch else None

    return {
        "t": round(time.time() - t0, 2),
        "temp_max_c": max(temps) if temps else _num(cpu.get("temperature_c")),
        "temp_avg_c": round(sum(temps) / len(temps), 1) if temps else None,
        "cpu_util_pct": _num(cpu.get("utilization_pct")),   # normalizada 0-100
        "cpu_freq_mhz": _num(cpu.get("frequency_mhz")),
        "gpu_util_pct": _num(gpu.get("utilization_pct")),
        "gpu_freq_mhz": _num(gpu.get("frequency_mhz")),
        "power_w": vdd_in,
        "throttled": throt.get("active") if isinstance(throt.get("active"), bool) else None,
        "stale": stale,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-json", required=True, help="caminho do host.json do agente")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    t0 = time.time()
    with open(args.out, "w") as fh:
        fh.write(json.dumps({"meta": {"label": args.label, "interval": args.interval,
                                      "start_epoch": t0, "source": "agent host.json"}}) + "\n")
        fh.flush()
        while not _stop:
            row = _flat_from_hostjson(args.host_json, t0)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            slept = 0.0
            while slept < args.interval and not _stop:
                time.sleep(min(0.2, args.interval - slept))
                slept += 0.2
    return 0


if __name__ == "__main__":
    sys.exit(main())
