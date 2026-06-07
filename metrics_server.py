#!/usr/bin/env python3
"""Servidor local do dashboard e API de controle."""
import json, subprocess, os, glob, time, threading, traceback, io, base64, struct, wave, re, unicodedata, uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from datetime import datetime

try:
    from shared.scorecard import compute_scorecard, load_hardware_config
except Exception:
    compute_scorecard = None
    load_hardware_config = None

BASE = os.environ.get("PDC_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONTAINERS = {"vision": 8010, "slm": 8011, "audio": 8012}
MODEL_TARGETS = ("vision", "slm", "audio_asr", "audio_tts")

# Estado de execução dos benchmarks iniciados pelo dashboard.
_benchmark_status = {}  # {front: {running, progress, started, result}}
_lock = threading.Lock()


def _models_path():
    return os.path.join(BASE, "config", "models.yaml")


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path, data):
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _get_models_section(cfg, target, create=False):
    if target == "vision":
        section = cfg.setdefault("vision", {}) if create else cfg.get("vision", {})
    elif target == "slm":
        section = cfg.setdefault("slm", {}) if create else cfg.get("slm", {})
    elif target == "audio_asr":
        audio = cfg.setdefault("audio", {}) if create else cfg.get("audio", {})
        section = audio.setdefault("asr", {}) if create else audio.get("asr", {})
    elif target == "audio_tts":
        audio = cfg.setdefault("audio", {}) if create else cfg.get("audio", {})
        section = audio.setdefault("tts", {}) if create else audio.get("tts", {})
    else:
        raise ValueError(f"target inválido: {target}")

    if create:
        section.setdefault("default", "")
        section.setdefault("models", {})
    return section or {}


def _slugify_model_key(value):
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.lower()).strip("-")
    return text or f"model-{int(time.time())}"


def _basename_from_url(value):
    if not value:
        return ""
    return str(value).split("/")[-1].split("?")[0].strip()


def _strip_known_suffixes(value):
    text = str(value or "").strip()
    for suffix in (
        ".gguf",
        ".onnx.json",
        ".onnx",
        ".pt",
        ".bin",
        ".safetensors",
        ".pth",
    ):
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return text


def _normalize_artifact_url(value):
    url = str(value or "").strip()
    if not url:
        return ""
    url = url.replace("/blob/", "/resolve/")
    if "huggingface.co" in url and "/raw/" in url:
        url = url.replace("/raw/", "/resolve/")
    return url


def _guess_chat_format(*values):
    joined = " ".join(str(v or "") for v in values).lower()
    if "tinyllama" in joined:
        return "tinyllama"
    if "gemma" in joined:
        return "gemma3"
    return "chatml"


def _infer_model_key(target, data):
    for candidate in (
        data.get("key"),
        data.get("name"),
        data.get("model_id"),
        data.get("url"),
        data.get("model_url"),
        data.get("description"),
    ):
        if candidate:
            if candidate == data.get("url") or candidate == data.get("model_url"):
                return _slugify_model_key(_strip_known_suffixes(_basename_from_url(candidate)))
            return _slugify_model_key(candidate)
    return _slugify_model_key(target)


def _infer_description(target, data):
    description = str(data.get("description", "")).strip()
    if description:
        return description
    if target == "audio_asr" and data.get("model_id"):
        return f"ASR {data.get('model_id')}"
    source = data.get("model_url") or data.get("url") or data.get("model_id") or data.get("key")
    if source:
        label = _strip_known_suffixes(_basename_from_url(source) or source).replace("-", " ").replace("_", " ").strip()
        return label or ""
    return ""


def _prepare_model_payload(target, raw_data):
    data = dict(raw_data or {})

    if target == "vision":
        data["url"] = _normalize_artifact_url(data.get("url") or data.get("source"))
    elif target == "slm":
        data["url"] = _normalize_artifact_url(data.get("url") or data.get("source"))
        if not data.get("chat_format"):
            data["chat_format"] = _guess_chat_format(data.get("url"), data.get("key"), data.get("description"))
    elif target == "audio_asr":
        data["model_id"] = str(data.get("model_id") or data.get("source") or "").strip()
    elif target == "audio_tts":
        model_url = _normalize_artifact_url(data.get("model_url") or data.get("source"))
        config_url = _normalize_artifact_url(data.get("config_url") or data.get("source_extra"))
        if model_url and not config_url and model_url.lower().endswith(".onnx"):
            config_url = model_url + ".json"
        data["model_url"] = model_url
        data["config_url"] = config_url

    if not data.get("key"):
        data["key"] = _infer_model_key(target, data)
    if not data.get("description"):
        data["description"] = _infer_description(target, data)

    return data


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "sim", "yes", "on"}


def _to_number(value, integer=False):
    if value in (None, ""):
        return None
    number = float(value)
    if integer:
        return int(number)
    return int(number) if number.is_integer() else round(number, 3)


def _parse_languages(value):
    if value in (None, ""):
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def load_models_config():
    try:
        return _load_yaml(_models_path())
    except Exception:
        return {}


def get_models_api():
    """Returns structured models list for the frontend."""
    cfg = load_models_config()
    result = {"vision": [], "slm": [], "audio_asr": [], "audio_tts": []}

    def _collect(section, key="models", default_field="default"):
        default = section.get(default_field, "")
        return [
            {
                "key": k,
                "default": k == default,
                "description": (v or {}).get("description", ""),
            }
            for k, v in section.get(key, {}).items()
        ]

    vision_cfg = cfg.get("vision", {})
    result["vision"] = _collect(vision_cfg)
    slm_cfg = cfg.get("slm", {})
    result["slm"] = _collect(slm_cfg)
    asr_cfg = cfg.get("audio", {}).get("asr", {})
    result["audio_asr"] = _collect(asr_cfg)
    tts_cfg = cfg.get("audio", {}).get("tts", {})
    result["audio_tts"] = _collect(tts_cfg)
    return result


def get_models_catalog():
    cfg = load_models_config()
    catalog = {}
    for target in MODEL_TARGETS:
        section = _get_models_section(cfg, target, create=False)
        default = section.get("default", "")
        models = []
        for key, model_cfg in section.get("models", {}).items():
            models.append({
                "key": key,
                "default": key == default,
                "config": model_cfg or {},
            })
        catalog[target] = {"default": default, "models": models}
    return catalog


def _build_model_entry(target, data):
    description = str(data.get("description", "")).strip()
    size_mb = _to_number(data.get("size_mb"))
    min_ram_gb = _to_number(data.get("min_ram_gb"))
    gpu_capable = _to_bool(data.get("gpu_capable"), default=(target != "audio_tts"))

    def _common(entry):
        entry["gpu_capable"] = gpu_capable
        if description:
            entry["description"] = description
        if size_mb is not None:
            entry["size_mb"] = size_mb
        if min_ram_gb is not None:
            entry["min_ram_gb"] = min_ram_gb
        return entry

    if target == "vision":
        url = str(data.get("url", "")).strip()
        if not url:
            raise ValueError("URL do modelo de visão é obrigatória")
        entry = _common({
            "format": str(data.get("format", "ultralytics")).strip() or "ultralytics",
            "url": url,
        })
        filename = str(data.get("filename", "")).strip()
        if filename:
            entry["filename"] = filename
        return entry

    if target == "slm":
        url = str(data.get("url", "")).strip()
        if not url:
            raise ValueError("URL do modelo SLM é obrigatória")
        entry = _common({
            "format": str(data.get("format", "gguf")).strip() or "gguf",
            "url": url,
            "gpu_layers": _to_number(data.get("gpu_layers", 33), integer=True) or 0,
            "context_length": _to_number(data.get("context_length", 2048), integer=True) or 2048,
            "chat_format": str(data.get("chat_format", "tinyllama")).strip() or "tinyllama",
        })
        filename = str(data.get("filename", "")).strip()
        if filename:
            entry["filename"] = filename
        return entry

    if target == "audio_asr":
        model_id = str(data.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("model_id do ASR é obrigatório")
        entry = _common({
            "format": str(data.get("format", "faster-whisper")).strip() or "faster-whisper",
            "model_id": model_id,
        })
        languages = _parse_languages(data.get("languages"))
        if languages:
            entry["languages"] = languages
        return entry

    if target == "audio_tts":
        model_url = str(data.get("model_url", "")).strip()
        config_url = str(data.get("config_url", "")).strip()
        if not model_url:
            raise ValueError("URL do modelo TTS é obrigatória")
        if not config_url:
            raise ValueError("URL do config do TTS é obrigatória")
        return _common({
            "format": str(data.get("format", "piper")).strip() or "piper",
            "model_url": model_url,
            "config_url": config_url,
        })

    raise ValueError(f"target inválido: {target}")


def write_models_config(cfg):
    _write_yaml(_models_path(), cfg)


def activate_model_target(target, key):
    try:
        if target == "slm":
            return _http_post("http://localhost:8011/switch-model", {"model": key}, timeout=1800)
        if target == "vision":
            return _http_post("http://localhost:8010/switch-model", {"model": key}, timeout=1800)
        if target == "audio_asr":
            return _http_post("http://localhost:8012/switch-model", {"target": "asr", "model": key}, timeout=1800)
        if target == "audio_tts":
            return _http_post("http://localhost:8012/switch-model", {"target": "tts", "model": key}, timeout=1800)
        return {"error": f"target inválido: {target}"}
    except Exception as e:
        return {"error": str(e)}


def save_model_definition(data):
    target = str(data.get("target", "")).strip()
    if target not in MODEL_TARGETS:
        raise ValueError("target inválido")

    raw_source = " ".join(
        str(data.get(k, "") or "")
        for k in ("source", "source_extra", "url", "model_url", "config_url", "model_id")
    ).lower()
    if "ollama.com/" in raw_source or raw_source.startswith("ollama:") or raw_source.startswith("ollama://"):
        raise ValueError("Importação direta de Ollama ainda não é suportada. Para SLM, use o link do arquivo GGUF.")

    data = _prepare_model_payload(target, data)
    cfg = load_models_config()
    section = _get_models_section(cfg, target, create=True)
    key = _slugify_model_key(data.get("key") or target)
    section["models"][key] = _build_model_entry(target, data)

    if _to_bool(data.get("set_default")) or not section.get("default"):
        section["default"] = key

    write_models_config(cfg)

    activation = None
    status = "ok"
    if _to_bool(data.get("activate_now")):
        activation = activate_model_target(target, key)
        if activation.get("error"):
            status = "saved_with_warning"

    return {
        "status": status,
        "target": target,
        "key": key,
        "default": section.get("default") == key,
        "activation": activation,
    }


def set_default_model(target, key, activate_now=False):
    if target not in MODEL_TARGETS:
        raise ValueError("target inválido")

    cfg = load_models_config()
    section = _get_models_section(cfg, target, create=True)
    if key not in section.get("models", {}):
        raise ValueError(f"modelo '{key}' não encontrado")

    section["default"] = key
    write_models_config(cfg)

    activation = activate_model_target(target, key) if activate_now else None
    return {"status": "ok", "target": target, "key": key, "activation": activation}


def delete_model_definition(target, key):
    if target not in MODEL_TARGETS:
        raise ValueError("target inválido")

    cfg = load_models_config()
    section = _get_models_section(cfg, target, create=True)
    models = section.get("models", {})
    if key not in models:
        raise ValueError(f"modelo '{key}' não encontrado")

    del models[key]
    if section.get("default") == key:
        section["default"] = next(iter(models.keys()), "")
    write_models_config(cfg)
    return {"status": "ok", "target": target, "key": key, "new_default": section.get("default")}


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


def _http_post_multipart(url, fields=None, files=None, timeout=120):
    import urllib.request

    boundary = f"----pdc-{uuid.uuid4().hex}"
    body = io.BytesIO()

    for name, value in (fields or {}).items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(str(value).encode())
        body.write(b"\r\n")

    for file_item in (files or []):
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{file_item["name"]}"; filename="{file_item["filename"]}"\r\n'.encode()
        )
        body.write(f'Content-Type: {file_item.get("content_type", "application/octet-stream")}\r\n\r\n'.encode())
        body.write(file_item["content"])
        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


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


def render_results_html(front_filter="all"):
    """Server-side render the results list as HTML rows (flex divs, no table)."""
    results = all_results()
    if front_filter and front_filter != "all":
        results = [r for r in results if r.get("front") == front_filter]
    if not results:
        return ('<div style="color:#737373;padding:20px;text-align:center;font-size:.82rem">'
                'Nenhum resultado. Execute benchmarks primeiro.</div>')

    def fmt(v, dec=1):
        if v is None:
            return "—"
        try:
            return f"{float(v):.{dec}f}"
        except Exception:
            return str(v)

    rows = []
    for r in results:
        ts = ""
        raw_ts = r.get("timestamp", "")
        if raw_ts:
            try:
                dt = datetime.fromisoformat(raw_ts)
                ts = dt.strftime("%d/%m %H:%M")
            except Exception:
                ts = raw_ts[:16]
        front = r.get("front") or "?"
        model = (r.get("model") or "?").replace(".gguf", "").replace(".gemma4", "")[:32]
        profile = r.get("profile") or "?"
        fc = "#93c5fd" if front == "vision" else "#22c55e" if front == "slm" else "#f59e0b"
        thr = fmt(r.get("throughput_ops"), 2)
        p50 = fmt(r.get("p50_ms"))
        p95 = fmt(r.get("p95_ms"))
        p99 = fmt(r.get("p99_ms"))
        std = fmt(r.get("std_ms"))
        ram = (fmt(r.get("peak_ram_mb"), 0) + "MB") if r.get("peak_ram_mb") is not None else "—"
        gpu = (fmt(r.get("peak_gpu_util"), 0) + "%") if r.get("peak_gpu_util") is not None else "—"
        vram = (fmt(r.get("peak_vram_mb"), 0) + "MB") if r.get("peak_vram_mb") is not None else "—"
        temp = (fmt(r.get("peak_temp_c"), 0) + "°C") if r.get("peak_temp_c") is not None else "—"
        rows.append(
            '<div style="display:flex;padding:5px 0;border-bottom:1px solid #242424;'
            'font-size:.77rem;color:#e7e7e7;align-items:center">'
            f'<div style="flex:0 0 88px;color:#a3a3a3;font-size:.72rem;white-space:nowrap">{ts}</div>'
            f'<div style="flex:0 0 56px"><span style="background:#202020;color:{fc};padding:1px 7px;'
            f'border-radius:5px;font-size:.68rem;font-weight:700">{front}</span></div>'
            f'<div style="flex:1;min-width:150px;font-size:.73rem;overflow:hidden;'
            f'text-overflow:ellipsis;white-space:nowrap" title="{r.get("model","")}">{model}</div>'
            f'<div style="flex:0 0 62px;color:#a3a3a3">{profile}</div>'
            f'<div style="flex:0 0 82px;font-weight:700;color:#14b8a6" title="Inferências concluídas por segundo no teste de throughput">{thr}</div>'
            f'<div style="flex:0 0 58px" title="Latência p50 do teste de latência">{p50}</div>'
            f'<div style="flex:0 0 58px" title="Latência p95 do teste de latência">{p95}</div>'
            f'<div style="flex:0 0 58px" title="Latência p99 do teste de latência">{p99}</div>'
            f'<div style="flex:0 0 58px" title="Desvio padrão das amostras de latência">{std}</div>'
            f'<div style="flex:0 0 68px" title="Pico de RAM no teste de uso de recursos">{ram}</div>'
            f'<div style="flex:0 0 52px" title="Maior utilização percentual de GPU observada">{gpu}</div>'
            f'<div style="flex:0 0 64px" title="Maior uso de VRAM observado">{vram}</div>'
            f'<div style="flex:0 0 52px" title="Maior temperatura observada">{temp}</div>'
            '</div>'
        )
    return "\n".join(rows)


def _normalize_model(m):
    """Return a plain string for the model field regardless of JSON shape."""
    if m is None:
        return "?"
    if isinstance(m, str):
        return m
    if isinstance(m, dict):
        parts = [v for v in m.values() if isinstance(v, str)]
        return " / ".join(parts) if parts else "?"
    return str(m)


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
            sust = b.get("sustained_load", b.get("sustained", {}))
            membw = b.get("memory_bandwidth", {})
            scorecard = d.get("scorecard") or {}
            if not scorecard and compute_scorecard:
                try:
                    hw_cfg = load_hardware_config(os.path.join(BASE, "config", "hardware.yaml")) if load_hardware_config else {}
                    scorecard = compute_scorecard(d, hw_cfg)
                except Exception:
                    scorecard = {}
            ts_data = res.get("timeseries", [])
            # Derive peak GPU/VRAM from timeseries if not in top-level dict
            gpu_vals  = [s["gpu_pct"]    for s in ts_data if s.get("gpu_pct")    is not None]
            vram_vals = [s["gpu_mem_mb"] for s in ts_data if s.get("gpu_mem_mb") is not None]
            ram_vals  = [s["ram_mb"]     for s in ts_data if s.get("ram_mb")     is not None]
            peak_gpu  = res.get("peak_gpu_util")  or (max(gpu_vals)  if gpu_vals  else None)
            peak_vram = res.get("peak_vram_mb")   or (max(vram_vals) if vram_vals else None)
            peak_ram  = res.get("peak_ram_mb")    or (max(ram_vals)  if ram_vals  else None)
            out.append({
                "file": os.path.basename(f),
                "front": d.get("front"), "model": _normalize_model(d.get("model")),
                "profile": d.get("profile"), "timestamp": d.get("timestamp"),
                "throughput_ops": tp.get("ops_per_sec"),
                "total_iters": tp.get("iterations"),
                "p50_ms": lat.get("p50_ms"), "p95_ms": lat.get("p95_ms"), "p99_ms": lat.get("p99_ms"),
                "std_ms": lat.get("std_ms"), "min_ms": lat.get("min_ms"), "max_ms": lat.get("max_ms"),
                "peak_ram_mb": peak_ram, "avg_cpu_pct": res.get("avg_cpu_pct"),
                "peak_gpu_util": peak_gpu,
                "peak_vram_mb": peak_vram,
                "peak_temp_c": res.get("peak_temp_c"),
                "sustained_avg_ops": sust.get("avg_ops_per_sec"),
                "sustained_degradation": sust.get("degradation_pct"),
                "mem_bw_gbs": membw.get("effective_bandwidth_gbs"),
                "energy_j": res.get("total_energy_j"),
                "methodology_version": d.get("methodology", {}).get("version"),
                "comparability_key": d.get("metadata", {}).get("comparability_key"),
                "throughput_error_rate": tp.get("error_rate"),
                "latency_error_rate": lat.get("error_rate"),
                "performance_score": scorecard.get("performance_score"),
                "primary_metric": scorecard.get("primary_metric"),
                "energy_per_op_j": scorecard.get("efficiency", {}).get("energy_per_op_j"),
                "energy_per_primary_unit_j": scorecard.get("efficiency", {}).get("energy_per_primary_unit_j"),
                "primary_metric_per_watt": scorecard.get("efficiency", {}).get("primary_metric_per_watt"),
                "cost_per_1k": scorecard.get("cost", {}).get("cost_per_1k_primary_units"),
                "scorecard_warnings": scorecard.get("warnings", []),
            })
        except:
            pass
    return out


def scorecard_summary():
    """Aggregate high-level scorecard data for hardware comparison dashboards."""
    rows = all_results()
    by_front = {}
    for row in rows:
        front = row.get("front") or "unknown"
        by_front.setdefault(front, []).append(row)

    summary = {"generated_at": datetime.now().isoformat(), "fronts": {}, "results": rows}
    for front, items in by_front.items():
        scores = [r.get("performance_score") for r in items if r.get("performance_score") is not None]
        costs = [r.get("cost_per_1k") for r in items if r.get("cost_per_1k") is not None]
        epops = [r.get("energy_per_op_j") for r in items if r.get("energy_per_op_j") is not None]
        latest = items[0] if items else {}
        summary["fronts"][front] = {
            "runs": len(items),
            "latest_model": latest.get("model"),
            "latest_profile": latest.get("profile"),
            "avg_performance_score": round(sum(scores) / len(scores), 2) if scores else None,
            "best_performance_score": round(max(scores), 2) if scores else None,
            "best_cost_per_1k": round(min(costs), 8) if costs else None,
            "best_energy_per_op_j": round(min(epops), 6) if epops else None,
        }
    return summary


def _activate_for_benchmark(front, model=None, mode=None):
    if not model:
        return

    if front == "slm" and isinstance(model, str):
        result = activate_model_target("slm", model)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return

    if front == "vision" and isinstance(model, str):
        result = activate_model_target("vision", model)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return

    if front == "audio":
        if isinstance(model, str) and mode in ("asr", "tts"):
            result = activate_model_target(f"audio_{mode}", model)
            if result.get("error"):
                raise RuntimeError(result["error"])
            return
        if isinstance(model, dict):
            if model.get("asr"):
                result = activate_model_target("audio_asr", model["asr"])
                if result.get("error"):
                    raise RuntimeError(result["error"])
            if model.get("tts"):
                result = activate_model_target("audio_tts", model["tts"])
                if result.get("error"):
                    raise RuntimeError(result["error"])


def trigger_benchmark(front, profile="quick", model=None, mode=None):
    """Run benchmark in background thread, activating the selected model before execution."""
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
            _activate_for_benchmark(front, model=model, mode=mode)
            payload = {"profile": profile}
            if mode and front == "audio":
                payload["mode"] = mode
            result = _http_post(
                f"http://localhost:{port}/benchmark",
                payload,
                timeout=600
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
    return activate_model_target("slm", model_key)


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


def do_vision_inference(model_key, filename, content_type, image_bytes):
    try:
        if model_key:
            activation = activate_model_target("vision", model_key)
            if activation.get("error"):
                return activation
        result = _http_post_multipart(
            "http://localhost:8010/predict",
            files=[{
                "name": "file",
                "filename": filename or "image.bin",
                "content": image_bytes,
                "content_type": content_type or "application/octet-stream",
            }],
            timeout=180,
        )
        return result
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


DASHBOARD_PATH = os.path.join(BASE, "web", "dashboard.html")


def load_dashboard_html():
    with open(DASHBOARD_PATH, encoding="utf-8") as f:
        return f.read()



class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
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
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
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
            try:
                self._html_response(load_dashboard_html())
            except Exception as e:
                self._json_response({"error": f"dashboard indisponível: {e}"}, 500)

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

        elif path == "/api/scorecard":
            self._json_response(scorecard_summary())

        elif path == "/api/models":
            self._json_response(get_models_api())

        elif path == "/api/models/catalog":
            self._json_response(get_models_catalog())

        elif path == "/api/results-html":
            qs = parse_qs(urlparse(self.path).query)
            front = qs.get("front", ["all"])[0]
            total = len(all_results())
            html = render_results_html(front)
            self._json_response({"html": html, "total": total, "filter": front})

        elif path == "/api/config":
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
            model = data.get("model")
            mode = data.get("mode")
            if front not in CONTAINERS:
                self._json_response({"error": f"unknown front: {front}"}, 400)
                return
            with _lock:
                if _benchmark_status.get(front, {}).get("running"):
                    self._json_response({"error": "already running"}, 409)
                    return
            trigger_benchmark(front, profile, model=model, mode=mode)
            self._json_response({"status": "started", "front": front, "profile": profile})

        elif path == "/api/switch-model":
            result = do_slm_switch(data.get("model", ""))
            self._json_response(result)

        elif path == "/api/models/save":
            try:
                self._json_response(save_model_definition(data))
            except Exception as e:
                self._json_response({"error": str(e)}, 400)

        elif path == "/api/models/default":
            try:
                target = data.get("target", "")
                key = data.get("key", "")
                activate_now = _to_bool(data.get("activate_now"))
                self._json_response(set_default_model(target, key, activate_now=activate_now))
            except Exception as e:
                self._json_response({"error": str(e)}, 400)

        elif path == "/api/models/delete":
            try:
                self._json_response(delete_model_definition(data.get("target", ""), data.get("key", "")))
            except Exception as e:
                self._json_response({"error": str(e)}, 400)

        elif path == "/api/models/activate":
            target = data.get("target", "")
            key = data.get("key", "")
            result = activate_model_target(target, key)
            self._json_response(result, 200 if not result.get("error") else 400)

        elif path == "/api/inference":
            result = do_slm_inference(data.get("prompt", ""), data.get("max_tokens", 256))
            self._json_response(result)

        elif path == "/api/vision-infer":
            try:
                image_b64 = data.get("image_base64", "")
                image_bytes = base64.b64decode(image_b64) if image_b64 else b""
                if not image_bytes:
                    self._json_response({"error": "imagem vazia"}, 400)
                    return
                result = do_vision_inference(
                    data.get("model", ""),
                    data.get("filename", "image.bin"),
                    data.get("content_type", "application/octet-stream"),
                    image_bytes,
                )
                self._json_response(result, 200 if not result.get("error") else 400)
            except Exception as e:
                self._json_response({"error": str(e)}, 400)

        elif path == "/api/tts":
            result = do_tts(data.get("text", ""))
            self._json_response(result)

        elif path == "/api/config":
            yaml_text = data.get("yaml_text", "")
            cfg_path = os.path.join(BASE, "config", "benchmark.yaml")
            try:
                import yaml
                yaml.safe_load(yaml_text)
                with open(cfg_path, "w") as f:
                    f.write(yaml_text)
                self._json_response({"status": "ok"})
            except ImportError:
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
    print(f"PDC-IA Dashboard v1 -> http://localhost:{port}")
    print("API: /api/metrics, /api/benchmark, /api/inference, /api/vision-infer, /api/tts, /api/models/*")
    server.serve_forever()
