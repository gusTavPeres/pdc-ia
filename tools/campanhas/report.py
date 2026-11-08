#!/usr/bin/env python3
"""Gera um relatório HTML autocontido de uma campanha de benchmark.

Lê os resultados (JSON do benchmark) e as séries de telemetria (JSONL do
sampler) de results/<device>/<campanha>/ e produz um report.html com:
  - gráficos SVG inline (sem dependência externa) de temperatura, clock e
    potência ao longo de cada teste, marcando throttling;
  - tabelas de throughput/latência por repetição, com mediana e desvio;
  - metodologia e o que cada perfil significa.

Uso: python3 report.py results/<device>/<campanha>/
"""
import json
import math
import os
import statistics
import sys
from datetime import datetime

W, H, PAD = 720, 220, 44  # dimensões do gráfico SVG


def _model_label(m):
    if isinstance(m, dict):
        asr = m.get("asr"); tts = m.get("tts")
        return ("asr:" + str(asr)) + (" tts:" + str(tts) if tts else "") if asr or tts else "?"
    return str(m or "?").replace(".gguf", "").replace(".pt", "")[:22]


def _read_series(path):
    rows, meta = [], {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj:
                meta = obj["meta"]
            elif "t" in obj:
                rows.append(obj)
    return meta, rows


def _first(r, fields):
    """Primeiro campo presente e numérico dentre os candidatos."""
    for f in fields:
        v = r.get(f)
        if isinstance(v, (int, float)):
            return v
    return None


def _svg_line(rows, fields, color, label, unit):
    """Uma curva SVG ao longo de t. fields = nome ou lista de candidatos."""
    if isinstance(fields, str):
        fields = [fields]
    pts = [(r["t"], _first(r, fields)) for r in rows]
    pts = [(t, v) for t, v in pts if v is not None]
    if len(pts) < 2:
        return f'<p class="nodata">{label}: sem dados nesta placa</p>'
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymax = ymin + 1
    span = (xmax - xmin) or 1

    def px(x):
        return PAD + (x - xmin) / span * (W - 2 * PAD)

    def py(y):
        return H - PAD - (y - ymin) / (ymax - ymin) * (H - 2 * PAD)

    poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
    # faixas de throttling (fundo vermelho translúcido)
    bands = ""
    for r in rows:
        if r.get("throttled") is True or r.get("thermal_throttling") is True:
            x = px(r["t"])
            bands += f'<rect x="{x-2:.1f}" y="{PAD}" width="4" height="{H-2*PAD}" fill="#e5484d" opacity="0.25"/>'
    # eixos com min/max
    grid = (
        f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#8888" />'
        f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#8888" />'
        f'<text x="{PAD-6}" y="{py(ymax)+4:.0f}" font-size="11" fill="#888" text-anchor="end">{ymax:.0f}</text>'
        f'<text x="{PAD-6}" y="{py(ymin)+4:.0f}" font-size="11" fill="#888" text-anchor="end">{ymin:.0f}</text>'
        f'<text x="{W-PAD}" y="{H-PAD+16:.0f}" font-size="11" fill="#888" text-anchor="end">{xmax:.0f}s</text>'
    )
    return (
        f'<figure><figcaption>{label} — pico {ymax:.1f} {unit}, '
        f'mín {ymin:.1f} {unit}</figcaption>'
        f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet">'
        f'{bands}{grid}<polyline fill="none" stroke="{color}" stroke-width="2" points="{poly}"/>'
        f'</svg></figure>'
    )


def _stat_row(label, values, unit):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return f"<tr><td>{label}</td><td colspan='4' class='nodata'>—</td></tr>"
    med = statistics.median(vals)
    lo, hi = min(vals), max(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return (f"<tr><td>{label}</td><td>{med:.3f} {unit}</td><td>{lo:.3f}</td>"
            f"<td>{hi:.3f}</td><td>±{sd:.3f} (n={len(vals)})</td></tr>")


def _series_from_result(result):
    """Deriva uma série {t, ...} da timeseries embutida no resultado.

    Cobre a fase resource_usage (não o soak inteiro), mas garante gráfico
    mesmo sem o sampler. O sampler continua sendo a fonte completa no estresse.
    """
    ts = (result.get("benchmark_results", {})
          .get("resource_usage", {}).get("timeseries", []))
    rows = []
    for s in ts:
        if "timestamp" not in s:
            continue
        r = dict(s)
        r["t"] = s["timestamp"]
        rows.append(r)
    return rows


def build(campaign_dir):
    campaign_dir = os.path.abspath(campaign_dir)
    tests = []  # (nome, result_json|None, series_rows)
    for name in sorted(os.listdir(campaign_dir)):
        d = os.path.join(campaign_dir, name)
        if not os.path.isdir(d):
            continue
        result = None
        rp = os.path.join(d, "result.json")
        if os.path.isfile(rp):
            try:
                result = json.load(open(rp))
            except Exception:
                result = None
        rows = []
        sp = os.path.join(d, "telemetria.jsonl")
        if os.path.isfile(sp):
            _, rows = _read_series(sp)
        if not rows and result:
            rows = _series_from_result(result)
        tests.append((name, result, rows))

    # agrupa repetições por (frente, perfil) para estatística
    groups = {}
    for name, result, _rows in tests:
        if not result:
            continue
        key = (result.get("front", "?"), _model_label(result.get("model")), result.get("profile", "?"))
        groups.setdefault(key, []).append(result)

    parts = []
    for (front, model, profile), reps in sorted(groups.items()):
        tps = [r.get("benchmark_results", {}).get("throughput", {}).get("ops_per_sec") for r in reps]
        p50 = [r.get("benchmark_results", {}).get("latency", {}).get("p50_ms") for r in reps]
        p99 = [r.get("benchmark_results", {}).get("latency", {}).get("p99_ms") for r in reps]
        sl = [r.get("benchmark_results", {}).get("sustained_load", {}) for r in reps]
        throttled = any(x.get("throttling_detected") for x in sl)
        degr = [x.get("degradation_pct") for x in sl if x.get("degradation_pct") is not None]
        tmax = [x.get("temp_max_c") for x in sl if x.get("temp_max_c") is not None]
        parts.append(
            f"<h3>{front} · {model} · perfil {profile} · {len(reps)} repetição(ões)</h3>"
            f"<p class='{'warn' if throttled else 'ok'}'>"
            f"{'⚠ throttling térmico observado' if throttled else '✓ sem throttling térmico'}</p>"
            "<table><thead><tr><th>Métrica</th><th>mediana</th><th>mín</th>"
            "<th>máx</th><th>dispersão</th></tr></thead><tbody>"
            + _stat_row("Throughput", tps, "ops/s")
            + _stat_row("Latência p50", p50, "ms")
            + _stat_row("Latência p99", p99, "ms")
            + _stat_row("Degradação sob carga", degr, "%")
            + _stat_row("Temp. máx (soak)", tmax, "°C")
            + "</tbody></table>")

    charts = []
    for name, _result, rows in tests:
        if not rows:
            continue
        charts.append(f"<h3>{name}</h3><div class='grid'>"
                      + _svg_line(rows, ["temp_max_c", "temp_c", "cpu_temp_c", "platform_temp_c"], "#e5484d", "Temperatura", "°C")
                      + _svg_line(rows, ["cpu_freq_mhz"], "#4c6ef5", "Clock CPU", "MHz")
                      + _svg_line(rows, ["cpu_util_pct", "host_cpu_pct"], "#7048e8", "CPU (normalizada 0-100)", "%")
                      + _svg_line(rows, ["power_w", "platform_power_w", "cpu_package_power_w"], "#f08c00", "Potência", "W")
                      + _svg_line(rows, ["gpu_util_pct", "gpu_pct", "gpu_device_pct"], "#2f9e44", "GPU", "%")
                      + _svg_line(rows, ["ram_mb", "container_ram_mb"], "#0ca678", "RAM", "MB")
                      + "</div>")

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relatório de Benchmark — {os.path.basename(campaign_dir)}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:24px;color:#1a1a1a;background:#fafafa}}
 h1{{font-size:22px}} h2{{font-size:18px;margin-top:32px;border-bottom:2px solid #ddd;padding-bottom:4px}}
 h3{{font-size:15px;margin-top:24px}}
 table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}}
 th,td{{border:1px solid #ddd;padding:6px 10px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
 figure{{margin:0;border:1px solid #eee;border-radius:8px;padding:8px;background:#fff}}
 figcaption{{font-size:12px;color:#555;margin-bottom:4px}}
 .ok{{color:#2b8a3e}} .warn{{color:#e03131;font-weight:600}} .nodata{{color:#aaa;font-style:italic}}
 @media(max-width:640px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Relatório de Benchmark da Frota PDC</h1>
<p>Campanha <code>{os.path.basename(campaign_dir)}</code> · gerado em {gen}</p>
<h2>Desempenho por frente e perfil</h2>
{''.join(parts) or '<p class="nodata">Nenhum resultado de benchmark encontrado.</p>'}
<h2>Telemetria durante os testes</h2>
<p>Faixas vermelhas marcam janelas com throttling térmico. Uma fonte ausente
na placa (ex.: potência no Raspberry Pi) aparece como "sem dados".</p>
{''.join(charts) or '<p class="nodata">Nenhuma série de telemetria encontrada.</p>'}
<h2>Metodologia</h2>
<ul>
<li><b>quick/standard</b>: desempenho. <b>soak (30min)/torture (60min)</b>: estresse com carga sustentada.</li>
<li>Repetições permitem ver dispersão (desvio) e reprodutibilidade, não um número solto.</li>
<li>Telemetria amostrada 1×/s no device pelo mesmo coletor do agente pdc-telemetry.</li>
<li>Cada resultado é atrelado ao commit exato que rodou (ver metadata do JSON).</li>
<li>CPU é utilização normalizada 0-100% (não a soma de núcleos do cgroup, que iria a 600% num hexa-core). Potência é o trilho de entrada VDD_IN no Jetson.</li>
</ul>
</body></html>"""
    out = os.path.join(campaign_dir, "report.html")
    open(out, "w").write(html)
    return out, len(tests), len(groups)


def _selftest():
    """Gera dados sintéticos e confirma que o HTML sai com gráfico e tabela."""
    import tempfile
    d = tempfile.mkdtemp()
    t1 = os.path.join(d, "vision_standard_rep1")
    os.makedirs(t1)
    json.dump({"front": "vision", "profile": "standard",
               "benchmark_results": {"throughput": {"ops_per_sec": 12.5},
                                     "latency": {"p50_ms": 80, "p99_ms": 120},
                                     "sustained_load": {"throttling_detected": True, "degradation_pct": 8.3, "temp_max_c": 78.0}}},
              open(os.path.join(t1, "result.json"), "w"))
    with open(os.path.join(t1, "telemetria.jsonl"), "w") as fh:
        fh.write(json.dumps({"meta": {"label": "t"}}) + "\n")
        for i in range(30):
            fh.write(json.dumps({"t": i, "temp_max_c": 50 + i * 0.5,
                                 "cpu_freq_mhz": 1400 - i * 5,
                                 "throttled": i > 20, "power_w": 5 + i * 0.1}) + "\n")
    out, n, g = build(d)
    html = open(out).read()
    assert "<svg" in html, "faltou gráfico SVG"
    assert "throttling" in html, "faltou marcação de throttling"
    assert "Throughput" in html, "faltou tabela de desempenho"
    assert n == 1 and g == 1
    print(f"  selftest OK: {out} ({len(html)} bytes, {n} teste, {g} grupo)")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
    elif len(sys.argv) == 2:
        out, n, g = build(sys.argv[1])
        print(f"Relatório: {out}  ({n} testes, {g} grupos)")
    else:
        print(__doc__)
        sys.exit(1)
