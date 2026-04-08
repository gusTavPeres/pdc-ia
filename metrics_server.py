#!/usr/bin/env python3
"""
PDC-IA Dashboard v2 — Central de Benchmarks, Monitoramento e Inferência
Servidor completo com API REST + Dashboard HTML integrado.
"""
import json, subprocess, os, glob, time, threading, traceback, io, base64, struct, wave
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from datetime import datetime

BASE = os.path.expanduser("~/pdc-ia")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONTAINERS = {"vision": 8010, "slm": 8011, "audio": 8012}

# ─── Background state ───
_benchmark_status = {}  # {front: {running, progress, started, result}}
_lock = threading.Lock()


def _http_get(url, timeout=5):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _http_post(url, data, timeout=60):
    import urllib.request
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _http_post_raw(url, data, timeout=60):
    """POST that returns raw bytes + content-type (for binary responses like WAV)."""
    import urllib.request
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def gpu_metrics():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm,"
            "fan.speed,pstate",
            "--format=csv,noheader,nounits"
        ], text=True, timeout=3).strip()
        parts = [p.strip() for p in out.split(",")]
        safe = lambda i: float(parts[i]) if parts[i] not in ("[N/A]", "N/A", "[Not Supported]") else None
        return {
            "name": parts[0], "util_gpu_pct": safe(1), "util_mem_pct": safe(2),
            "mem_used_mb": safe(3), "mem_total_mb": safe(4), "temp_c": safe(5),
            "power_w": safe(6), "clock_mhz": safe(7), "fan_pct": safe(8),
            "pstate": parts[9] if len(parts) > 9 else None,
        }
    except Exception as e:
        return {"error": str(e)}


def system_metrics():
    try:
        import psutil
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage("/")
        temps = psutil.sensors_temperatures()
        cpu_temp = None
        for name in ("coretemp", "k10temp", "cpu_thermal"):
            if name in temps and temps[name]:
                cpu_temp = temps[name][0].current
                break
        return {
            "ram_total_gb": round(vm.total / 1e9, 1),
            "ram_used_gb": round(vm.used / 1e9, 1),
            "ram_pct": vm.percent,
            "cpu_pct": cpu,
            "cpu_count": psutil.cpu_count(),
            "disk_total_gb": round(disk.total / 1e9, 1),
            "disk_used_gb": round(disk.used / 1e9, 1),
            "disk_pct": disk.percent,
            "cpu_temp_c": cpu_temp,
        }
    except Exception as e:
        return {"error": str(e)}


def container_health(port):
    try:
        return _http_get(f"http://localhost:{port}/health", timeout=5)
    except:
        return {"status": "offline"}


def container_info(port):
    try:
        return _http_get(f"http://localhost:{port}/info", timeout=5)
    except:
        return {}


def all_results():
    files = sorted(glob.glob(f"{RESULTS_DIR}/*.json"), key=os.path.getmtime, reverse=True)
    out = []
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
            b = d.get("benchmark_results", {})
            tp = b.get("throughput", {})
            lat = b.get("latency", {})
            res = b.get("resource_usage", {})
            sust = b.get("sustained", {})
            membw = b.get("memory_bandwidth", {})
            out.append({
                "file": os.path.basename(f),
                "front": d.get("front"), "model": d.get("model", "?"),
                "profile": d.get("profile"), "timestamp": d.get("timestamp"),
                "throughput_ops": tp.get("ops_per_sec"),
                "total_iters": tp.get("iterations"),
                "p50_ms": lat.get("p50_ms"), "p95_ms": lat.get("p95_ms"), "p99_ms": lat.get("p99_ms"),
                "std_ms": lat.get("std_ms"), "min_ms": lat.get("min_ms"), "max_ms": lat.get("max_ms"),
                "peak_ram_mb": res.get("peak_ram_mb"), "avg_cpu_pct": res.get("avg_cpu_pct"),
                "peak_gpu_util": res.get("peak_gpu_util"),
                "peak_vram_mb": res.get("peak_vram_mb"),
                "peak_temp_c": res.get("peak_temp_c"),
                "sustained_avg_ops": sust.get("avg_ops_per_sec"),
                "sustained_degradation": sust.get("degradation_pct"),
                "mem_bw_gbs": membw.get("effective_bandwidth_gbs"),
                "energy_j": res.get("total_energy_j"),
            })
        except:
            pass
    return out


def trigger_benchmark(front, profile="quick"):
    """Run benchmark in background thread"""
    port = CONTAINERS.get(front)
    if not port:
        return
    with _lock:
        if _benchmark_status.get(front, {}).get("running"):
            return  # already running
        _benchmark_status[front] = {"running": True, "progress": "starting", "started": time.time()}

    def _run():
        try:
            with _lock:
                _benchmark_status[front]["progress"] = "running"
            result = _http_post(
                f"http://localhost:{port}/benchmark",
                {"profile": profile},
                timeout=300
            )
            with _lock:
                _benchmark_status[front] = {"running": False, "progress": "done", "result": result}
        except Exception as e:
            with _lock:
                _benchmark_status[front] = {"running": False, "progress": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()


def do_slm_inference(prompt, max_tokens=256):
    try:
        return _http_post("http://localhost:8011/predict", {
            "prompt": prompt, "max_tokens": max_tokens
        }, timeout=60)
    except Exception as e:
        return {"error": str(e)}


def do_slm_switch(model_key):
    try:
        return _http_post("http://localhost:8011/switch-model", {"model": model_key}, timeout=60)
    except Exception as e:
        return {"error": str(e)}


def do_tts(text):
    try:
        raw, ctype = _http_post_raw("http://localhost:8012/tts", {"text": text}, timeout=30)
        audio_b64 = base64.b64encode(raw).decode("ascii")
        # Try to compute duration from WAV header
        duration_s = None
        try:
            buf = io.BytesIO(raw)
            with wave.open(buf, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    duration_s = round(frames / rate, 2)
        except Exception:
            pass
        return {"audio_base64": audio_b64, "duration_s": duration_s}
    except Exception as e:
        return {"error": str(e)}


def read_benchmark_config():
    path = os.path.join(BASE, "config", "benchmark.yaml")
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # fallback: read raw
        with open(path) as f:
            return {"raw": f.read()}
    except Exception as e:
        return {"error": str(e)}


def write_benchmark_config(config):
    path = os.path.join(BASE, "config", "benchmark.yaml")
    try:
        import yaml
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return {"status": "ok"}
    except ImportError:
        return {"error": "pyyaml not installed on host"}
    except Exception as e:
        return {"error": str(e)}


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PDC-IA — Plataforma de Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0a0c10;--card:#12151e;--card2:#181c28;--border:#252a3a;--accent:#7c6af5;--accent2:#a78bfa;
  --green:#22c55e;--red:#ef4444;--yellow:#eab308;--blue:#3b82f6;--cyan:#06b6d4;
  --text:#e2e8f0;--sub:#94a3b8;--dim:#64748b;--radius:12px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter','Segoe UI',system-ui,sans-serif;overflow-x:hidden}
a{color:var(--accent2);text-decoration:none}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

/* Layout */
.header{padding:16px 24px;display:flex;align-items:center;gap:16px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--bg);z-index:100}
.header h1{font-size:1.2rem;font-weight:700;background:linear-gradient(135deg,var(--accent),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .tag{font-size:.7rem;padding:2px 8px;border-radius:10px;background:var(--accent);color:#fff;font-weight:600}
.header .spacer{flex:1}
.header .clock{font-size:.8rem;color:var(--sub);font-variant-numeric:tabular-nums}

.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);padding:0 24px;background:var(--card)}
.tab{padding:10px 20px;font-size:.85rem;color:var(--sub);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}

.page{display:none;padding:20px 24px}
.page.active{display:block}

.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(2,1fr)}
.g3{grid-template-columns:repeat(3,1fr)}
.g4{grid-template-columns:repeat(4,1fr)}
@media(max-width:1100px){.g3,.g4{grid-template-columns:repeat(2,1fr)}}
@media(max-width:700px){.g2,.g3,.g4{grid-template-columns:1fr}}

.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:16px;position:relative;overflow:hidden}
.card.accent-border{border-left:3px solid var(--accent)}
.card h2{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin-bottom:10px;font-weight:600}
.card h3{font-size:.85rem;color:var(--text);margin-bottom:6px}

/* Widgets */
.big-num{font-size:2.2rem;font-weight:800;line-height:1}
.big-num.gradient{background:linear-gradient(135deg,var(--accent),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.unit{font-size:.7rem;color:var(--dim);margin-left:2px}
.stat-label{font-size:.75rem;color:var(--sub);margin-top:2px}

.gauge-row{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:.82rem}
.gauge-lbl{width:90px;color:var(--sub);font-size:.78rem}
.gauge-bar{flex:1;height:7px;background:var(--border);border-radius:4px;overflow:hidden}
.gauge-fill{height:100%;border-radius:4px;transition:width .4s ease}
.gauge-val{width:50px;text-align:right;font-variant-numeric:tabular-nums;font-size:.8rem}

.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;display:inline-block}
.dot-ok{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot-off{background:var(--red);box-shadow:0 0 4px var(--red)}
.dot-warn{background:var(--yellow)}

.badge{display:inline-block;padding:2px 8px;border-radius:8px;font-size:.7rem;font-weight:600}
.b-gpu{background:#1e3a5f;color:#60a5fa}
.b-cpu{background:#2a2a1e;color:#facc15}
.b-ok{background:#14281e;color:#22c55e}
.b-run{background:#1e1e3a;color:#a78bfa;animation:pulse 1.5s infinite}
.b-err{background:#2a1414;color:#ef4444}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:8px;border:1px solid var(--border);
  background:var(--card2);color:var(--text);font-size:.82rem;cursor:pointer;transition:all .2s}
.btn:hover{background:var(--accent);border-color:var(--accent);color:#fff}
.btn:disabled{opacity:.4;pointer-events:none}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.primary:hover{background:var(--accent2)}
.btn.sm{padding:4px 10px;font-size:.75rem}
.btn.danger{border-color:var(--red);color:var(--red)}
.btn.danger:hover{background:var(--red);color:#fff}

select,input,textarea{background:var(--card2);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:8px;font-size:.82rem;font-family:inherit}
select:focus,input:focus,textarea:focus{outline:none;border-color:var(--accent)}
textarea{resize:vertical;width:100%}

table{width:100%;border-collapse:collapse;font-size:.78rem}
th{text-align:left;color:var(--dim);font-weight:600;padding:6px 10px;border-bottom:1px solid var(--border);font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}
td{padding:7px 10px;border-bottom:1px solid #151820}
tr:hover td{background:#151820}

.toast{position:fixed;bottom:20px;right:20px;background:var(--card);border:1px solid var(--accent);
  border-radius:10px;padding:12px 20px;z-index:999;font-size:.85rem;
  transform:translateY(100px);opacity:0;transition:all .3s}
.toast.show{transform:translateY(0);opacity:1}

canvas{max-height:180px}

.container-card{display:flex;flex-direction:column;gap:8px}
.container-card .status-line{display:flex;align-items:center;gap:8px;font-size:.88rem}
.container-card .detail{font-size:.78rem;color:var(--sub)}
.container-card .actions{display:flex;gap:6px;margin-top:6px}

.compare-grid{display:grid;grid-template-columns:200px 1fr 1fr;gap:0;font-size:.82rem}
.compare-grid .label{padding:8px 12px;background:var(--card2);border-bottom:1px solid var(--border);color:var(--sub);font-weight:500}
.compare-grid .val{padding:8px 12px;border-bottom:1px solid var(--border);text-align:center;font-variant-numeric:tabular-nums}
.compare-grid .val.better{color:var(--green);font-weight:600}
.compare-grid .val.worse{color:var(--red)}
.compare-grid .header{padding:8px 12px;background:var(--accent);color:#fff;font-weight:700;text-align:center;font-size:.78rem;text-transform:uppercase}

.log-box{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;font-family:'Fira Code','Cascadia Code',monospace;font-size:.75rem;color:var(--sub);max-height:200px;overflow-y:auto;white-space:pre-wrap}

.inference-output{background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:14px;
  font-size:.88rem;line-height:1.5;min-height:60px;max-height:300px;overflow-y:auto;white-space:pre-wrap}
</style>
</head>
<body>

<div class="header">
  <h1>PDC-IA</h1>
  <span class="tag">v2.0</span>
  <span style="color:var(--sub);font-size:.78rem">Plataforma de Benchmark de Hardware com IA</span>
  <div class="spacer"></div>
  <span class="clock" id="clock"></span>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('monitor')">Monitoramento</div>
  <div class="tab" onclick="showTab('benchmark')">Benchmarks</div>
  <div class="tab" onclick="showTab('inference')">Inferência</div>
  <div class="tab" onclick="showTab('results')">Resultados</div>
  <div class="tab" onclick="showTab('compare')">Comparação</div>
  <div class="tab" onclick="showTab('hardware')">Hardware</div>
  <div class="tab" onclick="showTab('config')">Configuração</div>
</div>

<!-- ═══ TAB: MONITORAMENTO ═══ -->
<div class="page active" id="page-monitor">
  <div class="grid g4" style="margin-bottom:14px">
    <div class="card accent-border">
      <h2>GPU Utilização</h2>
      <div class="big-num gradient" id="m-gpu-util">—</div>
      <div class="stat-label">%</div>
    </div>
    <div class="card accent-border">
      <h2>VRAM</h2>
      <div class="big-num gradient" id="m-vram">—</div>
      <div class="stat-label" id="m-vram-detail"></div>
    </div>
    <div class="card accent-border">
      <h2>GPU Temp</h2>
      <div class="big-num gradient" id="m-gpu-temp">—</div>
      <div class="stat-label">°C</div>
    </div>
    <div class="card accent-border">
      <h2>Potência</h2>
      <div class="big-num gradient" id="m-power">—</div>
      <div class="stat-label">W</div>
    </div>
  </div>

  <div class="grid g2" style="margin-bottom:14px">
    <div class="card">
      <h2>GPU — Tempo Real</h2>
      <canvas id="chart-gpu" height="160"></canvas>
    </div>
    <div class="card">
      <h2>Sistema — CPU & RAM</h2>
      <canvas id="chart-sys" height="160"></canvas>
    </div>
  </div>

  <div class="card" style="margin-bottom:14px">
    <h2>Recursos do Sistema</h2>
    <div class="gauge-row"><span class="gauge-lbl">GPU Compute</span><div class="gauge-bar"><div class="gauge-fill" id="bar-gpu" style="background:var(--accent);width:0%"></div></div><span class="gauge-val" id="pct-gpu">—</span></div>
    <div class="gauge-row"><span class="gauge-lbl">VRAM</span><div class="gauge-bar"><div class="gauge-fill" id="bar-vram" style="background:var(--cyan);width:0%"></div></div><span class="gauge-val" id="pct-vram">—</span></div>
    <div class="gauge-row"><span class="gauge-lbl">RAM</span><div class="gauge-bar"><div class="gauge-fill" id="bar-ram" style="background:var(--green);width:0%"></div></div><span class="gauge-val" id="pct-ram">—</span></div>
    <div class="gauge-row"><span class="gauge-lbl">CPU</span><div class="gauge-bar"><div class="gauge-fill" id="bar-cpu" style="background:var(--yellow);width:0%"></div></div><span class="gauge-val" id="pct-cpu">—</span></div>
    <div class="gauge-row"><span class="gauge-lbl">Disco</span><div class="gauge-bar"><div class="gauge-fill" id="bar-disk" style="background:var(--blue);width:0%"></div></div><span class="gauge-val" id="pct-disk">—</span></div>
  </div>

  <div class="grid g3">
    <div class="card container-card">
      <h2>Vision :8010</h2>
      <div id="c-vision"></div>
    </div>
    <div class="card container-card">
      <h2>SLM :8011</h2>
      <div id="c-slm"></div>
    </div>
    <div class="card container-card">
      <h2>Audio :8012</h2>
      <div id="c-audio"></div>
    </div>
  </div>
</div>

<!-- ═══ TAB: BENCHMARKS ═══ -->
<div class="page" id="page-benchmark">
  <div class="grid g3" style="margin-bottom:14px">
    <div class="card">
      <h2>Vision Benchmark</h2>
      <p style="font-size:.8rem;color:var(--sub);margin-bottom:10px">YOLOv8 — Detecção de objetos com imagens sintéticas 640×640</p>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">
        <select id="bench-vision-profile">
          <option value="express">Express (~15s)</option>
          <option value="quick" selected>Quick (~30s)</option>
          <option value="standard">Standard (~2min)</option>
          <option value="extended">Extended (~10min)</option>
        </select>
        <button class="btn primary" onclick="runBench('vision')" id="btn-bench-vision">Executar</button>
      </div>
      <div id="bench-vision-status"></div>
    </div>
    <div class="card">
      <h2>SLM Benchmark</h2>
      <p style="font-size:.8rem;color:var(--sub);margin-bottom:10px">Geração de texto — throughput tokens/s, latência, TTFT</p>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px">
        <select id="bench-slm-profile">
          <option value="express">Express (~15s)</option>
          <option value="quick" selected>Quick (~30s)</option>
          <option value="standard">Standard (~2min)</option>
          <option value="extended">Extended (~10min)</option>
        </select>
        <button class="btn primary" onclick="runBench('slm')" id="btn-bench-slm">Executar</button>
      </div>
      <div id="bench-slm-status"></div>
    </div>
    <div class="card">
      <h2>Audio Benchmark</h2>
      <p style="font-size:.8rem;color:var(--sub);margin-bottom:10px">Pipeline TTS→ASR — WER, latência, throughput</p>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">
        <select id="bench-audio-profile">
          <option value="express">Express (~15s)</option>
          <option value="quick" selected>Quick (~30s)</option>
          <option value="standard">Standard (~2min)</option>
          <option value="extended">Extended (~10min)</option>
        </select>
        <button class="btn primary" onclick="runBench('audio')" id="btn-bench-audio">Executar</button>
      </div>
      <div id="bench-audio-status"></div>
    </div>
  </div>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <h2 style="margin:0">Executar Todos</h2>
      <button class="btn primary" onclick="runAllBench()">Benchmark Completo (3 containers)</button>
    </div>
    <div id="bench-all-log" class="log-box" style="min-height:80px">Aguardando...</div>
  </div>
</div>

<!-- ═══ TAB: INFERÊNCIA ═══ -->
<div class="page" id="page-inference">
  <div class="grid g2">
    <div class="card">
      <h2>SLM — Chat</h2>
      <div style="margin-bottom:8px;display:flex;gap:8px;align-items:center">
        <span style="font-size:.78rem;color:var(--sub)">Modelo:</span>
        <select id="inf-model" onchange="switchModelInf()">
          <option value="tinyllama-1.1b-q4">TinyLlama 1.1B</option>
          <option value="qwen25-0.5b-q4">Qwen2.5 0.5B</option>
        </select>
        <span style="font-size:.78rem;color:var(--sub)">Max tokens:</span>
        <input type="number" id="inf-tokens" value="256" min="16" max="2048" style="width:70px">
      </div>
      <textarea id="inf-prompt" rows="3" placeholder="Digite seu prompt aqui...">Explique brevemente o que é inteligência artificial.</textarea>
      <div style="margin-top:8px;display:flex;gap:8px">
        <button class="btn primary" onclick="doInference()" id="btn-inf">Gerar</button>
        <span id="inf-time" style="font-size:.78rem;color:var(--sub);align-self:center"></span>
      </div>
      <div id="inf-output" class="inference-output" style="margin-top:10px"></div>
    </div>
    <div class="card">
      <h2>Audio — TTS</h2>
      <p style="font-size:.78rem;color:var(--sub);margin-bottom:8px">Sintetize texto em áudio (Piper pt-BR)</p>
      <textarea id="tts-text" rows="2" placeholder="Texto para sintetizar...">Olá, esta é uma demonstração do sistema de síntese de voz.</textarea>
      <div style="margin-top:8px;display:flex;gap:8px">
        <button class="btn primary" onclick="doTTS()" id="btn-tts">Sintetizar</button>
        <span id="tts-time" style="font-size:.78rem;color:var(--sub);align-self:center"></span>
      </div>
      <div id="tts-output" style="margin-top:10px"></div>
    </div>
  </div>
</div>

<!-- ═══ TAB: RESULTADOS ═══ -->
<div class="page" id="page-results">
  <div class="card" style="margin-bottom:14px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <h2 style="margin:0">Todos os Resultados de Benchmark</h2>
      <div style="display:flex;gap:8px">
        <select id="res-filter" onchange="renderResults()">
          <option value="all">Todos</option>
          <option value="vision">Vision</option>
          <option value="slm">SLM</option>
          <option value="audio">Audio</option>
        </select>
        <button class="btn sm" onclick="exportResults()">Exportar JSON</button>
      </div>
    </div>
    <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>Data/Hora</th><th>Front</th><th>Modelo</th><th>Perfil</th>
          <th>Throughput</th><th>p50</th><th>p95</th><th>p99</th>
          <th>σ ms</th><th>RAM Pico</th><th>GPU %</th><th>VRAM</th><th>Temp</th><th>Energia</th>
        </tr>
      </thead>
      <tbody id="res-body"></tbody>
    </table>
    </div>
  </div>
  <div class="grid g2">
    <div class="card">
      <h2>Throughput por Teste</h2>
      <canvas id="chart-throughput" height="200"></canvas>
    </div>
    <div class="card">
      <h2>Latência por Teste</h2>
      <canvas id="chart-latency" height="200"></canvas>
    </div>
  </div>
</div>

<!-- ═══ TAB: COMPARAÇÃO ═══ -->
<div class="page" id="page-compare">
  <div class="card" style="margin-bottom:14px">
    <h2>Comparar Modelos SLM</h2>
    <p style="font-size:.8rem;color:var(--sub);margin-bottom:12px">Selecione dois resultados de benchmark SLM para comparar lado a lado.</p>
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:14px">
      <select id="cmp-a" style="min-width:250px"></select>
      <span style="color:var(--dim)">vs</span>
      <select id="cmp-b" style="min-width:250px"></select>
      <button class="btn primary" onclick="doCompare()">Comparar</button>
    </div>
    <div id="compare-grid"></div>
  </div>
  <div class="card">
    <h2>Gráfico de Comparação</h2>
    <canvas id="chart-compare" height="200"></canvas>
  </div>
</div>

<!-- ═══ TAB: HARDWARE ═══ -->
<div class="page" id="page-hardware">
  <div class="grid g2">
    <div class="card">
      <h2>GPU</h2>
      <div id="hw-gpu"></div>
    </div>
    <div class="card">
      <h2>Sistema</h2>
      <div id="hw-sys"></div>
    </div>
  </div>
  <div class="card" style="margin-top:14px">
    <h2>Containers Docker</h2>
    <div id="hw-containers"></div>
  </div>
</div>

<!-- ═══ TAB: CONFIGURAÇÃO ═══ -->
<div class="page" id="page-config">
  <div class="grid g2">
    <div class="card">
      <h2>Perfis de Benchmark</h2>
      <p style="font-size:.8rem;color:var(--sub);margin-bottom:10px">
        Edite os parâmetros dos perfis de benchmark. As alterações são salvas no arquivo <code>config/benchmark.yaml</code>
        e ficam disponíveis imediatamente para os próximos benchmarks (sem rebuild dos containers).
      </p>
      <p style="font-size:.78rem;color:var(--dim);margin-bottom:10px">
        <b>Parâmetros por seção:</b><br>
        • <b>throughput</b>: duration_sec (duração do teste), warmup (iterações de aquecimento)<br>
        • <b>latency</b>: samples (nº de amostras), percentiles (lista: [50,95,99])<br>
        • <b>resource_usage</b>: duration_sec, interval_sec (intervalo de coleta)<br>
        • <b>sustained_load</b>: duration_min (0 = desabilitado)<br>
        • <b>memory_bandwidth</b>: enabled (true/false), patterns, block_sizes_kb
      </p>
      <textarea id="cfg-yaml" rows="25" style="font-family:monospace;font-size:.78rem"></textarea>
      <div style="margin-top:8px;display:flex;gap:8px">
        <button class="btn primary" onclick="saveConfig()">Salvar</button>
        <button class="btn" onclick="loadConfig()">Recarregar</button>
      </div>
      <div id="cfg-status" style="font-size:.78rem;color:var(--sub);margin-top:6px"></div>
    </div>
    <div class="card">
      <h2>Guia de Configuração</h2>
      <div style="font-size:.82rem;line-height:1.6;color:var(--sub)">
        <h3 style="color:var(--text);margin-bottom:6px">Como funciona</h3>
        <p>Cada perfil define parâmetros para 5 tipos de teste:</p>
        <ol style="padding-left:16px;margin:8px 0">
          <li><b>Throughput</b> — Roda inferências continuamente por <code>duration_sec</code> e mede ops/s</li>
          <li><b>Latência</b> — Executa <code>samples</code> inferências individuais e calcula percentis</li>
          <li><b>Uso de recursos</b> — Monitora CPU/RAM/GPU/VRAM/temp durante <code>duration_sec</code></li>
          <li><b>Carga sustentada</b> — Roda por <code>duration_min</code> minutos medindo degradação (0 = skip)</li>
          <li><b>Bandwidth de memória</b> — Testa largura de banda com padrões sequencial/random</li>
        </ol>
        <h3 style="color:var(--text);margin:12px 0 6px">Liberdade total</h3>
        <p>Você pode criar perfis novos, alterar qualquer parâmetro, ou remover seções inteiras.
           Por exemplo, para criar um perfil "custom":</p>
        <pre style="background:var(--bg);padding:8px;border-radius:6px;margin-top:6px;font-size:.75rem">profiles:
  custom:
    throughput: {duration_sec: 20, warmup: 2}
    latency: {samples: 30, percentiles: [50, 99]}
    resource_usage: {duration_sec: 20, interval_sec: 1}
    sustained_load: {duration_min: 0}
    memory_bandwidth: {enabled: false}</pre>
        <h3 style="color:var(--text);margin:12px 0 6px">TTS e ASR na GPU</h3>
        <p><b>Whisper (ASR)</b>: O modelo <code>faster-whisper</code> suporta GPU via CTranslate2/CUDA.
           Atualmente roda em CPU no container. Para habilitar GPU, seria necessário instalar
           <code>ctranslate2</code> com CUDA no Dockerfile do audio.</p>
        <p style="margin-top:4px"><b>Piper (TTS)</b>: Usa ONNX Runtime. Suporta GPU via <code>onnxruntime-gpu</code>,
           mas o ganho é marginal para modelos pequenos — o overhead de transferência CPU↔GPU supera o ganho.
           Para vozes pequenas como a pt-BR-faber, CPU é mais eficiente.</p>
      </div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
/* ─── State ─── */
let allResults = [];
let metrics = {};

/* ─── Tabs ─── */
function showTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const pages = ['monitor','benchmark','inference','results','compare','hardware','config'];
    t.classList.toggle('active', pages[i] === name);
  });
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === 'page-'+name));
  if(name === 'results') renderResults();
  if(name === 'compare') populateCompareSelects();
}

/* ─── Toast ─── */
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

/* ─── Clock ─── */
setInterval(() => {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString('pt-BR');
}, 1000);

/* ─── Charts ─── */
const MAX_H = 60;
const gpuHist = {util:[], vram:[], temp:[]};
const sysHist = {cpu:[], ram:[]};

const chartGpu = new Chart(document.getElementById('chart-gpu'), {
  type:'line',
  data:{labels:Array(MAX_H).fill(''),datasets:[
    {label:'GPU %',data:[],borderColor:'#7c6af5',backgroundColor:'#7c6af520',tension:.3,fill:true,pointRadius:0,borderWidth:1.5},
    {label:'VRAM %',data:[],borderColor:'#06b6d4',backgroundColor:'transparent',tension:.3,pointRadius:0,borderWidth:1.5},
    {label:'Temp °C',data:[],borderColor:'#ef4444',backgroundColor:'transparent',tension:.3,pointRadius:0,borderWidth:1,yAxisID:'y1'},
  ]},
  options:{animation:false,responsive:true,maintainAspectRatio:true,interaction:{mode:'index',intersect:false},
    scales:{
      y:{min:0,max:100,ticks:{color:'#64748b',font:{size:9}},grid:{color:'#1e2130'},title:{display:true,text:'%',color:'#64748b',font:{size:9}}},
      y1:{position:'right',min:20,max:95,ticks:{color:'#64748b',font:{size:9}},grid:{display:false},title:{display:true,text:'°C',color:'#64748b',font:{size:9}}},
      x:{display:false}
    },
    plugins:{legend:{labels:{color:'#94a3b8',font:{size:9},boxWidth:8,padding:8}}}
  }
});

const chartSys = new Chart(document.getElementById('chart-sys'), {
  type:'line',
  data:{labels:Array(MAX_H).fill(''),datasets:[
    {label:'CPU %',data:[],borderColor:'#eab308',backgroundColor:'#eab30820',tension:.3,fill:true,pointRadius:0,borderWidth:1.5},
    {label:'RAM %',data:[],borderColor:'#22c55e',backgroundColor:'transparent',tension:.3,pointRadius:0,borderWidth:1.5},
  ]},
  options:{animation:false,responsive:true,maintainAspectRatio:true,
    scales:{y:{min:0,max:100,ticks:{color:'#64748b',font:{size:9}},grid:{color:'#1e2130'}},x:{display:false}},
    plugins:{legend:{labels:{color:'#94a3b8',font:{size:9},boxWidth:8,padding:8}}}
  }
});

let chartThroughput, chartLatency, chartCompare;

function push(arr, v) { arr.push(v); if(arr.length > MAX_H) arr.shift(); }

/* ─── Container card ─── */
function containerHTML(data, front) {
  if(!data || data.status==='offline') return `<div class="status-line"><span class="dot dot-off"></span> Offline</div>`;
  const badge = data.gpu ? '<span class="badge b-gpu">CUDA</span>' : '<span class="badge b-cpu">CPU</span>';
  const model = data.model || data.asr || '';
  return `
    <div class="status-line"><span class="dot dot-ok"></span> Online ${badge}</div>
    <div class="detail">${model}</div>
    ${data.tts ? '<div class="detail">TTS: '+data.tts+'</div>' : ''}
  `;
}

/* ─── Metrics fetch ─── */
async function fetchMetrics() {
  try {
    const r = await fetch('/api/metrics');
    metrics = await r.json();
    const g = metrics.gpu || {};
    const s = metrics.system || {};

    // Big numbers
    if(!g.error) {
      document.getElementById('m-gpu-util').textContent = (g.util_gpu_pct||0).toFixed(0);
      document.getElementById('m-vram').textContent = (g.mem_used_mb||0).toFixed(0);
      document.getElementById('m-vram-detail').textContent = `${(g.mem_used_mb||0).toFixed(0)} / ${(g.mem_total_mb||0).toFixed(0)} MB`;
      document.getElementById('m-gpu-temp').textContent = (g.temp_c||0).toFixed(0);
      document.getElementById('m-power').textContent = g.power_w != null ? g.power_w.toFixed(1) : '—';

      const vramPct = g.mem_total_mb ? (g.mem_used_mb/g.mem_total_mb*100) : 0;
      setGauge('gpu', g.util_gpu_pct||0, '%');
      setGauge('vram', vramPct, '%');

      push(gpuHist.util, g.util_gpu_pct||0);
      push(gpuHist.vram, vramPct);
      push(gpuHist.temp, g.temp_c||0);
      chartGpu.data.datasets[0].data = [...gpuHist.util];
      chartGpu.data.datasets[1].data = [...gpuHist.vram];
      chartGpu.data.datasets[2].data = [...gpuHist.temp];
      chartGpu.update('none');
    }
    if(!s.error) {
      setGauge('ram', s.ram_pct||0, '%');
      setGauge('cpu', s.cpu_pct||0, '%');
      setGauge('disk', s.disk_pct||0, '%');
      push(sysHist.cpu, s.cpu_pct||0);
      push(sysHist.ram, s.ram_pct||0);
      chartSys.data.datasets[0].data = [...sysHist.cpu];
      chartSys.data.datasets[1].data = [...sysHist.ram];
      chartSys.update('none');
    }

    // Containers
    const c = metrics.containers || {};
    document.getElementById('c-vision').innerHTML = containerHTML(c.vision, 'vision');
    document.getElementById('c-slm').innerHTML = containerHTML(c.slm, 'slm');
    document.getElementById('c-audio').innerHTML = containerHTML(c.audio, 'audio');

    // Sync model select
    if(c.slm && c.slm.model) {
      const sel = document.getElementById('inf-model');
      if(c.slm.model.includes('tinyllama')) sel.value='tinyllama-1.1b-q4';
      else if(c.slm.model.toLowerCase().includes('qwen')) sel.value='qwen25-0.5b-q4';
    }

    // Store results
    allResults = metrics.results || [];

    // Benchmark status
    const bs = metrics.benchmark_status || {};
    for(const front of ['vision','slm','audio']) {
      const el = document.getElementById('bench-'+front+'-status');
      const btn = document.getElementById('btn-bench-'+front);
      const st = bs[front];
      if(st && st.running) {
        el.innerHTML = `<span class="badge b-run">Executando... ${((Date.now()/1000 - st.started)||0).toFixed(0)}s</span>`;
        btn.disabled = true;
      } else if(st && st.progress === 'done') {
        el.innerHTML = `<span class="badge b-ok">Concluído</span>`;
        btn.disabled = false;
      } else if(st && st.progress === 'error') {
        el.innerHTML = `<span class="badge b-err">Erro: ${st.error||'?'}</span>`;
        btn.disabled = false;
      } else {
        el.innerHTML = '';
        btn.disabled = false;
      }
    }

    // Hardware tab
    if(!g.error) {
      document.getElementById('hw-gpu').innerHTML = `
        <table>
          <tr><td style="color:var(--sub)">Nome</td><td>${g.name}</td></tr>
          <tr><td style="color:var(--sub)">VRAM</td><td>${g.mem_used_mb?.toFixed(0)} / ${g.mem_total_mb?.toFixed(0)} MB</td></tr>
          <tr><td style="color:var(--sub)">Utilização</td><td>${g.util_gpu_pct?.toFixed(1)}%</td></tr>
          <tr><td style="color:var(--sub)">Temperatura</td><td>${g.temp_c?.toFixed(0)} °C</td></tr>
          <tr><td style="color:var(--sub)">Potência</td><td>${g.power_w != null ? g.power_w.toFixed(1)+' W' : 'N/A'}</td></tr>
          <tr><td style="color:var(--sub)">Clock</td><td>${g.clock_mhz != null ? g.clock_mhz.toFixed(0)+' MHz' : 'N/A'}</td></tr>
          <tr><td style="color:var(--sub)">Fan</td><td>${g.fan_pct != null ? g.fan_pct.toFixed(0)+'%' : 'N/A'}</td></tr>
          <tr><td style="color:var(--sub)">P-State</td><td>${g.pstate || 'N/A'}</td></tr>
        </table>
      `;
    }
    if(!s.error) {
      document.getElementById('hw-sys').innerHTML = `
        <table>
          <tr><td style="color:var(--sub)">CPUs</td><td>${s.cpu_count} cores</td></tr>
          <tr><td style="color:var(--sub)">CPU Usage</td><td>${s.cpu_pct?.toFixed(1)}%</td></tr>
          <tr><td style="color:var(--sub)">CPU Temp</td><td>${s.cpu_temp_c != null ? s.cpu_temp_c.toFixed(0)+' °C' : 'N/A'}</td></tr>
          <tr><td style="color:var(--sub)">RAM</td><td>${s.ram_used_gb?.toFixed(1)} / ${s.ram_total_gb?.toFixed(1)} GB (${s.ram_pct?.toFixed(1)}%)</td></tr>
          <tr><td style="color:var(--sub)">Disco</td><td>${s.disk_used_gb?.toFixed(0)} / ${s.disk_total_gb?.toFixed(0)} GB (${s.disk_pct?.toFixed(1)}%)</td></tr>
        </table>
      `;
    }
    document.getElementById('hw-containers').innerHTML = `
      <table>
        <tr><th>Container</th><th>Status</th><th>Device</th><th>Model</th></tr>
        ${['vision','slm','audio'].map(f => {
          const d = c[f] || {};
          const st = d.status === 'ok' ? '<span class="badge b-ok">Online</span>' : '<span class="badge b-err">Offline</span>';
          const dev = d.gpu ? '<span class="badge b-gpu">CUDA</span>' : d.status==='ok' ? '<span class="badge b-cpu">CPU</span>' : '—';
          return `<tr><td>${f} :${({vision:8010,slm:8011,audio:8012})[f]}</td><td>${st}</td><td>${dev}</td><td style="color:var(--sub)">${d.model||d.asr||'—'}</td></tr>`;
        }).join('')}
      </table>
    `;

  } catch(e) { console.error('fetch error', e); }
}

function setGauge(name, val, suffix) {
  const bar = document.getElementById('bar-'+name);
  const pct = document.getElementById('pct-'+name);
  if(bar) bar.style.width = Math.min(val,100)+'%';
  if(pct) pct.textContent = val.toFixed(1)+suffix;
}

/* ─── Benchmark triggers ─── */
async function runBench(front) {
  const profile = document.getElementById('bench-'+front+'-profile').value;
  toast('Iniciando benchmark: '+front+' ('+profile+')');
  try {
    await fetch('/api/benchmark', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({front, profile})
    });
  } catch(e) { toast('Erro: '+e.message); }
}

async function runAllBench() {
  const log = document.getElementById('bench-all-log');
  log.textContent = '[' + new Date().toLocaleTimeString('pt-BR') + '] Iniciando benchmarks sequenciais...\n';
  for(const front of ['vision','slm','audio']) {
    const profile = document.getElementById('bench-'+front+'-profile').value;
    log.textContent += `[${new Date().toLocaleTimeString('pt-BR')}] → ${front} (${profile})...\n`;
    try {
      await fetch('/api/benchmark', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({front, profile})
      });
      // Wait for completion
      let done = false;
      while(!done) {
        await new Promise(r => setTimeout(r, 2000));
        const r = await fetch('/api/metrics');
        const d = await r.json();
        const st = (d.benchmark_status||{})[front];
        if(!st || !st.running) done = true;
      }
      log.textContent += `[${new Date().toLocaleTimeString('pt-BR')}] ✓ ${front} concluído\n`;
    } catch(e) {
      log.textContent += `[${new Date().toLocaleTimeString('pt-BR')}] ✗ ${front} erro: ${e.message}\n`;
    }
    log.scrollTop = log.scrollHeight;
  }
  log.textContent += `[${new Date().toLocaleTimeString('pt-BR')}] Todos os benchmarks concluídos!\n`;
  toast('Benchmarks completos!');
}

/* ─── Inference ─── */
async function switchModelInf() {
  const model = document.getElementById('inf-model').value;
  toast('Trocando modelo para '+model+'...');
  try {
    await fetch('/api/switch-model', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model})
    });
    toast('Modelo trocado!');
  } catch(e) { toast('Erro: '+e.message); }
}

async function doInference() {
  const prompt = document.getElementById('inf-prompt').value;
  const maxTokens = parseInt(document.getElementById('inf-tokens').value);
  const btn = document.getElementById('btn-inf');
  const out = document.getElementById('inf-output');
  btn.disabled = true;
  out.textContent = 'Gerando...';
  const t0 = performance.now();
  try {
    const r = await fetch('/api/inference', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt, max_tokens: maxTokens})
    });
    const d = await r.json();
    const elapsed = ((performance.now()-t0)/1000).toFixed(2);
    document.getElementById('inf-time').textContent = elapsed+'s';
    if(d.error) { out.textContent = 'Erro: '+d.error; }
    else {
      const text = d.response || d.text || '';
      const tps = d.tokens_per_sec ? d.tokens_per_sec.toFixed(1) : '?';
      const ttft = d.time_to_first_token_ms ? d.time_to_first_token_ms.toFixed(0) : '?';
      const tok = d.total_tokens || '?';
      out.innerHTML = `<div>${text}</div><div style="margin-top:8px;font-size:.75rem;color:var(--dim)">` +
        `${tps} tokens/s · TTFT: ${ttft}ms · ${tok} tokens · Modelo: ${d.model||'?'}</div>`;
    }
  } catch(e) { out.textContent = 'Erro: '+e.message; }
  btn.disabled = false;
}

async function doTTS() {
  const text = document.getElementById('tts-text').value;
  const btn = document.getElementById('btn-tts');
  const out = document.getElementById('tts-output');
  btn.disabled = true;
  out.innerHTML = '<span style="color:var(--sub)">Sintetizando...</span>';
  const t0 = performance.now();
  try {
    const r = await fetch('/api/tts', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const d = await r.json();
    const elapsed = ((performance.now()-t0)/1000).toFixed(2);
    document.getElementById('tts-time').textContent = elapsed+'s';
    if(d.error) { out.innerHTML = '<span style="color:var(--red)">Erro: '+d.error+'</span>'; }
    else if(d.audio_base64) {
      out.innerHTML = `<audio controls src="data:audio/wav;base64,${d.audio_base64}" style="width:100%"></audio>
        <div style="font-size:.75rem;color:var(--sub);margin-top:4px">Duração: ${d.duration_s?.toFixed(2)||'?'}s | Tempo: ${elapsed}s</div>`;
    } else {
      out.innerHTML = '<span style="color:var(--sub)">Sem áudio retornado</span>';
    }
  } catch(e) { out.innerHTML = '<span style="color:var(--red)">Erro: '+e.message+'</span>'; }
  btn.disabled = false;
}

/* ─── Results tab ─── */
function renderResults() {
  const filter = document.getElementById('res-filter').value;
  const data = filter === 'all' ? allResults : allResults.filter(r => r.front === filter);
  const tbody = document.getElementById('res-body');
  tbody.innerHTML = data.map(r => {
    const ts = r.timestamp ? new Date(r.timestamp).toLocaleString('pt-BR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}) : '—';
    const model = (r.model||'?').replace('.gguf','').slice(0,25);
    const frontBadge = r.front === 'vision' ? 'b-gpu' : r.front === 'slm' ? 'b-ok' : 'b-cpu';
    return `<tr>
      <td style="color:var(--sub)">${ts}</td>
      <td><span class="badge ${frontBadge}">${r.front}</span></td>
      <td>${model}</td>
      <td>${r.profile||'?'}</td>
      <td style="font-weight:600">${r.throughput_ops != null ? r.throughput_ops.toFixed(2) : '—'}</td>
      <td>${fmtMs(r.p50_ms)}</td>
      <td>${fmtMs(r.p95_ms)}</td>
      <td>${fmtMs(r.p99_ms)}</td>
      <td>${fmtMs(r.std_ms)}</td>
      <td>${r.peak_ram_mb != null ? r.peak_ram_mb.toFixed(0)+'MB' : '—'}</td>
      <td>${r.peak_gpu_util != null ? r.peak_gpu_util.toFixed(0)+'%' : '—'}</td>
      <td>${r.peak_vram_mb != null ? r.peak_vram_mb.toFixed(0)+'MB' : '—'}</td>
      <td>${r.peak_temp_c != null ? r.peak_temp_c.toFixed(0)+'°C' : '—'}</td>
      <td>${r.energy_j != null ? r.energy_j.toFixed(1)+'J' : '—'}</td>
    </tr>`;
  }).join('');

  // Throughput chart
  renderThroughputChart(data);
  renderLatencyChart(data);
}

function fmtMs(v) { return v != null ? v.toFixed(1) : '—'; }

function renderThroughputChart(data) {
  const canvas = document.getElementById('chart-throughput');
  if(chartThroughput) chartThroughput.destroy();
  const labels = data.map(r => (r.model||'?').replace('.gguf','').slice(0,15) + ' (' + (r.profile||'?') + ')');
  const colors = data.map(r => r.front==='vision'?'#7c6af5':r.front==='slm'?'#22c55e':'#eab308');
  chartThroughput = new Chart(canvas, {
    type:'bar',
    data:{labels, datasets:[{label:'ops/s', data:data.map(r=>r.throughput_ops||0), backgroundColor:colors, borderRadius:4}]},
    options:{responsive:true,maintainAspectRatio:true,animation:false,
      scales:{y:{ticks:{color:'#64748b'},grid:{color:'#1e2130'}},x:{ticks:{color:'#64748b',font:{size:8},maxRotation:45}}},
      plugins:{legend:{display:false}}}
  });
}

function renderLatencyChart(data) {
  const canvas = document.getElementById('chart-latency');
  if(chartLatency) chartLatency.destroy();
  const labels = data.map(r => (r.model||'?').replace('.gguf','').slice(0,15));
  chartLatency = new Chart(canvas, {
    type:'bar',
    data:{labels, datasets:[
      {label:'p50', data:data.map(r=>r.p50_ms||0), backgroundColor:'#7c6af5', borderRadius:3},
      {label:'p95', data:data.map(r=>r.p95_ms||0), backgroundColor:'#06b6d4', borderRadius:3},
      {label:'p99', data:data.map(r=>r.p99_ms||0), backgroundColor:'#ef4444', borderRadius:3},
    ]},
    options:{responsive:true,maintainAspectRatio:true,animation:false,
      scales:{y:{ticks:{color:'#64748b'},grid:{color:'#1e2130'},title:{display:true,text:'ms',color:'#64748b'}},
              x:{ticks:{color:'#64748b',font:{size:8},maxRotation:45}}},
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:9},boxWidth:8}}}}
  });
}

/* ─── Compare tab ─── */
function populateCompareSelects() {
  const slmResults = allResults.filter(r => r.front === 'slm');
  const html = slmResults.map((r,i) => {
    const m = (r.model||'?').replace('.gguf','').slice(0,25);
    const t = r.timestamp ? new Date(r.timestamp).toLocaleString('pt-BR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}) : '';
    return `<option value="${i}">${m} — ${r.profile||'?'} — ${t}</option>`;
  }).join('');
  document.getElementById('cmp-a').innerHTML = html;
  document.getElementById('cmp-b').innerHTML = html;
  if(slmResults.length > 1) document.getElementById('cmp-b').selectedIndex = 1;
}

function doCompare() {
  const slmResults = allResults.filter(r => r.front === 'slm');
  const ia = parseInt(document.getElementById('cmp-a').value);
  const ib = parseInt(document.getElementById('cmp-b').value);
  if(isNaN(ia) || isNaN(ib)) return;
  const a = slmResults[ia], b = slmResults[ib];
  if(!a || !b) return;

  const metrics = [
    ['Throughput (ops/s)', a.throughput_ops, b.throughput_ops, true],
    ['Latência p50 (ms)', a.p50_ms, b.p50_ms, false],
    ['Latência p95 (ms)', a.p95_ms, b.p95_ms, false],
    ['Latência p99 (ms)', a.p99_ms, b.p99_ms, false],
    ['Desvio padrão (ms)', a.std_ms, b.std_ms, false],
    ['RAM Pico (MB)', a.peak_ram_mb, b.peak_ram_mb, false],
    ['GPU Pico (%)', a.peak_gpu_util, b.peak_gpu_util, true],
    ['VRAM Pico (MB)', a.peak_vram_mb, b.peak_vram_mb, false],
    ['Temperatura Pico (°C)', a.peak_temp_c, b.peak_temp_c, false],
    ['Energia (J)', a.energy_j, b.energy_j, false],
  ];

  const nameA = (a.model||'?').replace('.gguf','').slice(0,20);
  const nameB = (b.model||'?').replace('.gguf','').slice(0,20);

  let html = `<div class="compare-grid">
    <div class="header">Métrica</div><div class="header">${nameA}</div><div class="header">${nameB}</div>`;
  for(const [label, va, vb, higherBetter] of metrics) {
    const fmtA = va != null ? (typeof va === 'number' ? va.toFixed(2) : va) : '—';
    const fmtB = vb != null ? (typeof vb === 'number' ? vb.toFixed(2) : vb) : '—';
    let clsA = 'val', clsB = 'val';
    if(va != null && vb != null) {
      if(higherBetter) { if(va > vb) clsA='val better'; else if(vb > va) clsB='val better'; }
      else { if(va < vb) clsA='val better'; else if(vb < va) clsB='val better'; }
    }
    html += `<div class="label">${label}</div><div class="${clsA}">${fmtA}</div><div class="${clsB}">${fmtB}</div>`;
  }
  html += '</div>';
  document.getElementById('compare-grid').innerHTML = html;

  // Radar chart
  const canvas = document.getElementById('chart-compare');
  if(chartCompare) chartCompare.destroy();
  const labels = metrics.filter(m => m[1]!=null && m[2]!=null).map(m => m[0].split('(')[0].trim());
  const normalize = (vals) => {
    const max = Math.max(...vals.filter(v=>v!=null));
    return vals.map(v => v!=null ? (v/max*100).toFixed(1) : 0);
  };
  const valsA = metrics.filter(m=>m[1]!=null&&m[2]!=null).map(m=>m[1]);
  const valsB = metrics.filter(m=>m[1]!=null&&m[2]!=null).map(m=>m[2]);
  const maxVals = valsA.map((v,i) => Math.max(v, valsB[i]));
  const normA = valsA.map((v,i) => (v/maxVals[i]*100).toFixed(1));
  const normB = valsB.map((v,i) => (v/maxVals[i]*100).toFixed(1));

  chartCompare = new Chart(canvas, {
    type:'radar',
    data:{labels, datasets:[
      {label:nameA, data:normA, borderColor:'#7c6af5', backgroundColor:'#7c6af530', pointRadius:3},
      {label:nameB, data:normB, borderColor:'#22c55e', backgroundColor:'#22c55e30', pointRadius:3},
    ]},
    options:{responsive:true,maintainAspectRatio:true,animation:false,
      scales:{r:{angleLines:{color:'#252a3a'},grid:{color:'#252a3a'},ticks:{display:false},suggestedMin:0,suggestedMax:100}},
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}}}
  });
}

/* ─── Export ─── */
function exportResults() {
  const blob = new Blob([JSON.stringify(allResults, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'pdc-ia-results-'+new Date().toISOString().slice(0,10)+'.json';
  a.click(); URL.revokeObjectURL(url);
  toast('Resultados exportados!');
}

/* ─── Config tab ─── */
async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    const d = await r.json();
    if(d.raw) { document.getElementById('cfg-yaml').value = d.raw; }
    else if(d.error) { document.getElementById('cfg-status').textContent = 'Erro: '+d.error; }
    else {
      // Convert JSON back to YAML-like display
      document.getElementById('cfg-yaml').value = JSON.stringify(d, null, 2);
    }
  } catch(e) { document.getElementById('cfg-status').textContent = 'Erro: '+e.message; }
}

async function saveConfig() {
  const text = document.getElementById('cfg-yaml').value;
  try {
    const r = await fetch('/api/config', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({yaml_text: text})
    });
    const d = await r.json();
    if(d.status === 'ok') { toast('Configuração salva!'); document.getElementById('cfg-status').textContent = 'Salvo com sucesso'; }
    else { document.getElementById('cfg-status').textContent = 'Erro: '+(d.error||'?'); }
  } catch(e) { document.getElementById('cfg-status').textContent = 'Erro: '+e.message; }
}

/* ─── Init ─── */
fetchMetrics();
setInterval(fetchMetrics, 3000);
// Load config on page load
loadConfig();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._html_response(DASHBOARD_HTML)

        elif path == "/api/metrics":
            with _lock:
                bs = {k: dict(v) for k, v in _benchmark_status.items()}
            data = {
                "gpu": gpu_metrics(),
                "system": system_metrics(),
                "containers": {k: container_health(v) for k, v in CONTAINERS.items()},
                "results": all_results(),
                "benchmark_status": bs,
                "timestamp": datetime.now().isoformat(),
            }
            self._json_response(data)

        elif path == "/api/results":
            self._json_response(all_results())

        elif path == "/api/config":
            # Read benchmark.yaml as raw text for editing
            cfg_path = os.path.join(BASE, "config", "benchmark.yaml")
            try:
                with open(cfg_path) as f:
                    self._json_response({"raw": f.read()})
            except Exception as e:
                self._json_response({"error": str(e)})

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        data = self._read_body()

        if path == "/api/benchmark":
            front = data.get("front", "")
            profile = data.get("profile", "quick")
            if front not in CONTAINERS:
                self._json_response({"error": f"unknown front: {front}"}, 400)
                return
            with _lock:
                if _benchmark_status.get(front, {}).get("running"):
                    self._json_response({"error": "already running"}, 409)
                    return
            trigger_benchmark(front, profile)
            self._json_response({"status": "started", "front": front, "profile": profile})

        elif path == "/api/switch-model":
            result = do_slm_switch(data.get("model", ""))
            self._json_response(result)

        elif path == "/api/inference":
            result = do_slm_inference(data.get("prompt", ""), data.get("max_tokens", 256))
            self._json_response(result)

        elif path == "/api/tts":
            result = do_tts(data.get("text", ""))
            self._json_response(result)

        elif path == "/api/config":
            # Write benchmark.yaml raw text
            yaml_text = data.get("yaml_text", "")
            cfg_path = os.path.join(BASE, "config", "benchmark.yaml")
            try:
                # Validate it's valid YAML
                import yaml
                yaml.safe_load(yaml_text)  # parse test
                with open(cfg_path, "w") as f:
                    f.write(yaml_text)
                self._json_response({"status": "ok"})
            except ImportError:
                # No yaml module, just write raw
                with open(cfg_path, "w") as f:
                    f.write(yaml_text)
                self._json_response({"status": "ok", "warning": "yaml validation skipped"})
            except Exception as e:
                self._json_response({"error": str(e)}, 400)

        else:
            self.send_response(404)
            self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    port = 8099
    server = ThreadedHTTPServer(("0.0.0.0", port), Handler)
    print(f"PDC-IA Dashboard v2 → http://localhost:{port}")
    print(f"API endpoints: /api/metrics, /api/benchmark, /api/inference, /api/tts, /api/switch-model")
    server.serve_forever()
