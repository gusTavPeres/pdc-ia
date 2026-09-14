#!/usr/bin/env python3
"""Servidor local do dashboard e API de controle."""
import base64
import csv
import glob
import hashlib
import html
import io
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from shared.capabilities import CapabilityDetector, save_hardware_snapshot
from shared.config_validation import validate_benchmark_config
from shared.device_telemetry import load_host_telemetry
from shared.result_parser import is_benchmark_result, load_result_file, normalize_result
from shared.scorecard import compute_scorecard, load_hardware_config

BASE = os.environ.get("PDC_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
HOST_TELEMETRY_PATH = os.path.join(BASE, "data", "telemetry", "host.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONTAINERS = {"vision": 8010, "slm": 8011, "audio": 8012}
CONTAINER_NAMES = {front: f"pdc-{front}" for front in CONTAINERS}
MODEL_TARGETS = ("vision", "slm", "audio_asr", "audio_tts")

# Estado de execução dos benchmarks iniciados pelo dashboard.
_benchmark_status = {}  # {front: {running, progress, started, result}}
_lock = threading.Lock()
_result_cache = {}
_result_cache_lock = threading.Lock()
_config_lock = threading.RLock()
_hardware_cache = None
_hardware_cache_lock = threading.Lock()


def _benchmark_timeout_sec():
    try:
        return max(600, min(86_400, int(os.environ.get(
            "PDC_BENCHMARK_TIMEOUT_SEC", "21600"))))
    except (TypeError, ValueError):
        return 21_600


def _models_path():
    return os.path.join(BASE, "config", "models.yaml")


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _atomic_write_text(path, text):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False) as handle:
        temporary_path = handle.name
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary_path, 0o644)
    try:
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _write_yaml(path, data):
    import yaml
    text = yaml.safe_dump(
        data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    with _config_lock:
        _atomic_write_text(path, text)


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


def _validate_artifact_url(value, label):
    url = str(value or "").strip()
    if len(url) > 4_096:
        raise ValueError(f"{label} excede 4.096 caracteres")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{label} deve usar HTTP ou HTTPS")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} não pode conter credenciais")
    return parsed.geturl()


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
    if not math.isfinite(number):
        raise ValueError("valor numérico inválido")
    if integer:
        return int(number)
    return int(number) if number.is_integer() else round(number, 3)


def _bounded_integer(value, label, minimum, maximum):
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} deve ser inteiro")
    if number < minimum or number > maximum:
        raise ValueError(f"{label} deve ficar entre {minimum} e {maximum}")
    return number


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
    """Retorna a lista de modelos usada pelo dashboard."""
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
    if len(description) > 500:
        raise ValueError("descrição deve ter até 500 caracteres")

    def optional_bounded_number(key, label, maximum):
        value = data.get(key)
        if value in (None, ""):
            return None
        return _bounded_float(value, label, 0, maximum)

    def optional_filename(value):
        filename = str(value or "").strip()
        if not filename:
            return None
        if len(filename) > 255 or os.path.basename(filename) != filename \
                or "\\" in filename or filename in (".", ".."):
            raise ValueError("nome de arquivo inválido")
        return filename

    size_mb = optional_bounded_number("size_mb", "tamanho do modelo", 10_000_000)
    min_ram_gb = optional_bounded_number("min_ram_gb", "RAM mínima", 1_000_000)
    sha256 = str(data.get("sha256") or "").strip().lower()
    if sha256 and not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("SHA-256 deve conter 64 caracteres hexadecimais")
    gpu_capable = _to_bool(data.get("gpu_capable"), default=(target != "audio_tts"))

    def _common(entry):
        entry["gpu_capable"] = gpu_capable
        if description:
            entry["description"] = description
        if size_mb is not None:
            entry["size_mb"] = size_mb
        if min_ram_gb is not None:
            entry["min_ram_gb"] = min_ram_gb
        if sha256:
            entry["sha256"] = sha256
        return entry

    if target == "vision":
        url = str(data.get("url", "")).strip()
        if not url:
            raise ValueError("URL do modelo de visão é obrigatória")
        url = _validate_artifact_url(url, "URL do modelo de visão")
        entry = _common({
            "format": str(data.get("format", "ultralytics")).strip() or "ultralytics",
            "url": url,
        })
        filename = optional_filename(data.get("filename"))
        if filename:
            entry["filename"] = filename
        return entry

    if target == "slm":
        url = str(data.get("url", "")).strip()
        if not url:
            raise ValueError("URL do modelo SLM é obrigatória")
        url = _validate_artifact_url(url, "URL do modelo SLM")
        entry = _common({
            "format": str(data.get("format", "gguf")).strip() or "gguf",
            "url": url,
            "gpu_layers": _bounded_integer(
                data.get("gpu_layers", 33), "camadas de GPU", 0, 1_000_000),
            "context_length": _bounded_integer(
                data.get("context_length", 2048), "contexto", 256, 10_000_000),
            "chat_format": str(data.get("chat_format", "tinyllama")).strip() or "tinyllama",
        })
        filename = optional_filename(data.get("filename"))
        if filename:
            entry["filename"] = filename
        return entry

    if target == "audio_asr":
        model_id = str(data.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("model_id do ASR é obrigatório")
        if len(model_id) > 512:
            raise ValueError("model_id do ASR deve ter até 512 caracteres")
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
        model_url = _validate_artifact_url(model_url, "URL do modelo TTS")
        config_url = _validate_artifact_url(config_url, "URL do config TTS")
        return _common({
            "format": str(data.get("format", "piper")).strip() or "piper",
            "model_url": model_url,
            "config_url": config_url,
        })

    raise ValueError(f"target inválido: {target}")


def write_models_config(cfg):
    _write_yaml(_models_path(), cfg)


def _expected_runtime_model(target, key):
    section = _get_models_section(load_models_config(), target, create=False)
    model = section.get("models", {}).get(key) or {}
    if target == "audio_asr":
        return str(model.get("model_id") or "")
    source = model.get("model_url") if target == "audio_tts" else model.get("url")
    return str(model.get("filename") or _basename_from_url(source) or "")


def _loaded_runtime_model(target):
    try:
        if target == "vision":
            info = _http_get("http://localhost:8010/info")
            return str((info.get("model") or {}).get("name") or "")
        if target == "slm":
            info = _http_get("http://localhost:8011/info")
            return str((info.get("model") or {}).get("name") or "")
        info = _http_get("http://localhost:8012/info")
        model = info.get("model") or {}
        field = "asr_model" if target == "audio_asr" else "tts_model"
        return str(model.get(field) or "")
    except Exception:
        return ""


def activate_model_target(target, key):
    try:
        expected = _expected_runtime_model(target, key)
        if expected and _loaded_runtime_model(target) == expected:
            return {
                "status": "ok",
                "target": target,
                "key": key,
                "already_loaded": True,
            }
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
    with _config_lock:
        cfg = load_models_config()
        section = _get_models_section(cfg, target, create=True)
        key = _slugify_model_key(data.get("key") or target)
        if len(key) > 128:
            raise ValueError("chave do modelo deve ter até 128 caracteres")
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

    with _config_lock:
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

    with _config_lock:
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
    """Executa POST e preserva respostas binárias, como áudio WAV."""
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
        field_name = re.sub(r"[^a-zA-Z0-9_-]", "", str(file_item["name"]))
        filename = os.path.basename(str(file_item["filename"]))
        filename = filename.replace('"', "").replace("\r", "").replace("\n", "")
        content_type = re.sub(
            r"[^a-zA-Z0-9.+/-]", "", str(file_item.get("content_type", "")))
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
        )
        body.write(f'Content-Type: {content_type or "application/octet-stream"}\r\n\r\n'.encode())
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


def host_telemetry_metrics():
    payload = load_host_telemetry(HOST_TELEMETRY_PATH, max_age_sec=10)
    if not payload:
        return {"error": "agente de telemetria indisponível"}
    # potência da placa (board_w) que o dashboard consome: trilho de entrada
    # VDD_IN no Jetson tem prioridade; senão soma os trilhos/sensores.
    power = payload.get("power") if isinstance(payload.get("power"), dict) else None
    if power is not None and power.get("board_w") is None:
        rails = power.get("rails") or []
        vin = next((r.get("power_w") for r in rails
                    if str(r.get("label", "")).upper().startswith("VDD_IN")
                    and isinstance(r.get("power_w"), (int, float))), None)
        if vin is None:
            watts = [r.get("power_w") for r in rails
                     if isinstance(r.get("power_w"), (int, float))]
            vin = round(sum(watts), 3) if watts else None
        power["board_w"] = vin
    return payload


def _host_gpu_metrics(error: str, detail=None):
    telemetry = host_telemetry_metrics()
    gpu = telemetry.get("gpu") if isinstance(telemetry.get("gpu"), dict) else {}
    if gpu and any(gpu.get(key) is not None for key in (
            "utilization_pct", "frequency_mhz", "temperature_c")):
        return {
            "name": gpu.get("name") or "GPU integrada",
            "util_gpu_pct": gpu.get("utilization_pct"),
            "util_mem_pct": None,
            "mem_used_mb": None,
            "mem_total_mb": None,
            "temp_c": gpu.get("temperature_c"),
            "power_w": gpu.get("power_w"),
            "clock_mhz": gpu.get("frequency_mhz"),
            "fan_pct": None,
            "pstate": None,
            "source": gpu.get("source") or "host_telemetry",
            "scope": gpu.get("scope") or "visible_devices",
            "detail": error,
        }
    result = {"error": error}
    if detail:
        result["detail"] = detail
    return result


def gpu_metrics():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm,"
            "fan.speed,pstate",
            "--format=csv,noheader,nounits"
        ], text=True, timeout=3, stderr=subprocess.PIPE).strip()
        rows = list(csv.reader(out.splitlines()))
        devices = []
        unavailable = {"[N/A]", "N/A", "[Not Supported]", ""}
        for index, row in enumerate(rows):
            parts = [part.strip() for part in row]
            if len(parts) < 9:
                continue

            def number(position):
                try:
                    return None if parts[position] in unavailable else float(parts[position])
                except (IndexError, ValueError):
                    return None

            devices.append({
                "index": index,
                "name": parts[0],
                "util_gpu_pct": number(1),
                "util_mem_pct": number(2),
                "mem_used_mb": number(3),
                "mem_total_mb": number(4),
                "temp_c": number(5),
                "power_w": number(6),
                "clock_mhz": number(7),
                "fan_pct": number(8),
                "pstate": parts[9] if len(parts) > 9 else None,
            })
        if not devices:
            return {"error": "nenhuma GPU retornada pelo nvidia-smi"}

        def aggregate(key, mode="max"):
            values = [device[key] for device in devices if device.get(key) is not None]
            if not values:
                return None
            return sum(values) if mode == "sum" else max(values)

        # No Jetson o nvidia-smi lista o device mas devolve tudo [N/A]: os dados
        # reais (util, temp, potência, clock) vêm do tegrastats. Se o nvidia-smi
        # não trouxe métrica nenhuma útil, usa a telemetria do agente.
        if all(aggregate(k) is None for k in
               ("util_gpu_pct", "temp_c", "power_w", "clock_mhz")):
            host = _host_gpu_metrics("nvidia-smi sem métricas (Jetson usa tegrastats)")
            if "error" not in host:
                return host
        return {
            "name": " / ".join(device["name"] for device in devices),
            "util_gpu_pct": aggregate("util_gpu_pct"),
            "util_mem_pct": aggregate("util_mem_pct"),
            "mem_used_mb": aggregate("mem_used_mb", "sum"),
            "mem_total_mb": aggregate("mem_total_mb", "sum"),
            "temp_c": aggregate("temp_c"),
            "power_w": aggregate("power_w", "sum"),
            "clock_mhz": aggregate("clock_mhz"),
            "fan_pct": aggregate("fan_pct"),
            "pstate": " / ".join(str(device["pstate"]) for device in devices
                                  if device.get("pstate")),
            "devices": devices,
        }
    except FileNotFoundError:
        return _host_gpu_metrics("nvidia-smi não está instalado")
    except subprocess.TimeoutExpired:
        return _host_gpu_metrics("A consulta ao driver NVIDIA excedeu 3 segundos")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        return _host_gpu_metrics(
            "Telemetria NVIDIA indisponível",
            detail[0][:240] if detail else "nvidia-smi encerrou com erro",
        )
    except Exception as exc:
        return _host_gpu_metrics("Falha ao consultar a GPU", type(exc).__name__)


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
            "ram_total_gb": round(vm.total / (1024 ** 3), 1),
            "ram_used_gb": round(vm.used / (1024 ** 3), 1),
            "ram_pct": vm.percent,
            "cpu_pct": cpu,
            "cpu_count": psutil.cpu_count(),
            "disk_total_gb": round(disk.total / (1024 ** 3), 1),
            "disk_used_gb": round(disk.used / (1024 ** 3), 1),
            "disk_pct": disk.percent,
            "cpu_temp_c": cpu_temp,
        }
    except Exception as e:
        return {"error": str(e)}


def container_health(port):
    try:
        return _http_get(f"http://localhost:{port}/health", timeout=5)
    except Exception as exc:
        try:
            body = exc.read()
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {"status": "offline"}


def _docker_inspect(front):
    if front not in CONTAINER_NAMES:
        raise ValueError(f"frente inválida: {front}")
    output = subprocess.check_output(
        ["docker", "inspect", CONTAINER_NAMES[front]],
        text=True,
        timeout=5,
        stderr=subprocess.STDOUT,
    )
    data = json.loads(output)[0]
    host = data.get("HostConfig", {})
    state = data.get("State", {})
    health = state.get("Health", {})
    memory_bytes = int(host.get("Memory") or 0)
    nano_cpus = int(host.get("NanoCpus") or 0)
    shm_bytes = int(host.get("ShmSize") or 0)
    return {
        "front": front,
        "name": CONTAINER_NAMES[front],
        "status": health.get("Status") or state.get("Status") or "unknown",
        "running": bool(state.get("Running")),
        "cpu_limit": nano_cpus / 1_000_000_000 if nano_cpus else 0,
        "memory_limit_mb": round(memory_bytes / (1024 ** 2)) if memory_bytes else 0,
        "pids_limit": host.get("PidsLimit")
        if isinstance(host.get("PidsLimit"), int) and host.get("PidsLimit") > 0 else 0,
        "shm_size_mb": round(shm_bytes / (1024 ** 2)),
        "restart_policy": (host.get("RestartPolicy") or {}).get("Name") or "no",
    }


def _container_limits_config():
    path = os.path.join(BASE, "config", "container_limits.yaml")
    try:
        config = _load_yaml(path)
    except FileNotFoundError:
        config = {}
    for front in CONTAINERS:
        config.setdefault(front, {
            "cpu_limit": 0,
            "memory_limit_mb": 0,
            "pids_limit": 0,
        })
    return config


def container_runtime_info():
    desired = _container_limits_config()
    result = {}
    for front in CONTAINERS:
        try:
            result[front] = _docker_inspect(front)
            result[front]["configured"] = desired.get(front, {})
        except Exception as exc:
            result[front] = {
                "front": front,
                "name": CONTAINER_NAMES[front],
                "status": "unavailable",
                "error": str(exc).strip(),
                "configured": desired.get(front, {}),
            }
    return result


def _bounded_float(value, label, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} deve ser numérico")
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{label} deve ficar entre {minimum} e {maximum}")
    return number


def update_container_limits(data):
    front = str(data.get("front", ""))
    if front not in CONTAINER_NAMES:
        raise ValueError("frente inválida")
    with _lock:
        if _benchmark_status.get(front, {}).get("running"):
            raise RuntimeError("não altere limites durante um benchmark")

    cpu_limit = _bounded_float(data.get("cpu_limit", 0), "limite de CPU", 0, 1024)
    if cpu_limit == 0:
        cpu_limit = 0
    memory_limit = _bounded_integer(
        data.get("memory_limit_mb", 0), "limite de RAM", 0, 1_048_576)
    pids_limit = _bounded_integer(
        data.get("pids_limit", 0), "limite de processos", 0, 4_194_304)
    if 0 < memory_limit < 64:
        raise ValueError("limite de RAM deve ser zero ou pelo menos 64 MB")
    if 0 < cpu_limit < 0.1:
        raise ValueError("limite de CPU deve ser zero ou pelo menos 0,1")
    if 0 < pids_limit < 16:
        raise ValueError("limite de processos deve ser zero ou pelo menos 16")

    command = [
        "docker", "update",
        "--cpus", str(cpu_limit),
        "--memory", "0" if memory_limit == 0 else f"{memory_limit}m",
        "--pids-limit", "-1" if pids_limit == 0 else str(pids_limit),
        CONTAINER_NAMES[front],
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "docker update falhou")
    limits_path = os.path.join(BASE, "config", "container_limits.yaml")
    with _config_lock:
        limits = _container_limits_config()
        limits[front] = {
            "cpu_limit": cpu_limit,
            "memory_limit_mb": memory_limit,
            "pids_limit": pids_limit,
        }
        _write_yaml(limits_path, limits)
    return {"status": "ok", "container": _docker_inspect(front)}


def restart_container(front):
    if front not in CONTAINER_NAMES:
        raise ValueError("frente inválida")
    with _lock:
        if _benchmark_status.get(front, {}).get("running"):
            raise RuntimeError("não reinicie o container durante um benchmark")
    result = subprocess.run(
        ["docker", "restart", CONTAINER_NAMES[front]],
        capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "docker restart falhou")
    return {"status": "ok", "front": front}


def hardware_config_api():
    path = os.path.join(BASE, "config", "hardware.yaml")
    return _load_yaml(path)


def benchmark_profiles_api():
    config = _load_yaml(os.path.join(BASE, "config", "benchmark.yaml"))
    validate_benchmark_config(config)
    profiles = []
    for name, profile in config["profiles"].items():
        fixed_duration = 0.0
        for section_name in ("throughput", "resource_usage"):
            section = profile.get(section_name) or {}
            if section.get("enabled", True):
                fixed_duration += float(section.get("duration_sec", 0))
        sustained = profile.get("sustained_load") or {}
        if sustained.get("enabled", True):
            fixed_duration += float(sustained.get("duration_min", 0)) * 60.0
        latency = profile.get("latency") or {}
        latency_samples = (
            int(latency.get("samples", 0)) if latency.get("enabled", True) else 0
        )
        profiles.append({
            "name": name,
            "fixed_duration_sec": round(fixed_duration, 3),
            "latency_samples": latency_samples,
        })
    return {"default": config.get("default_profile"), "profiles": profiles}


def detected_hardware_api(refresh=False):
    global _hardware_cache
    with _hardware_cache_lock:
        if refresh or _hardware_cache is None:
            hardware = CapabilityDetector().detect(use_host_snapshot=False)
            snapshot_path = os.path.join(BASE, "data", "hardware", "host.json")
            snapshot = save_hardware_snapshot(hardware, snapshot_path)
            _hardware_cache = {
                "detected_at": snapshot["captured_at"],
                "hardware_id": snapshot["hardware_id"],
                "snapshot_path": os.path.relpath(snapshot_path, BASE),
                "hardware": hardware,
            }
        return _hardware_cache


def save_hardware_config(data):
    path = os.path.join(BASE, "config", "hardware.yaml")
    current = hardware_config_api()
    profile_id = str(data.get("profile_id", "")).strip()
    description = str(data.get("description", "")).strip()
    currency = str(data.get("currency", "BRL")).strip().upper()
    if not profile_id or len(profile_id) > 128:
        raise ValueError("identificador da bancada é obrigatório e deve ter até 128 caracteres")
    if len(description) > 500:
        raise ValueError("descrição deve ter até 500 caracteres")
    if not re.fullmatch(r"[A-Z]{3,8}", currency):
        raise ValueError("moeda deve usar de 3 a 8 letras")

    def optional_number(key, label, strictly_positive=False):
        value = data.get(key)
        if value in (None, ""):
            return None
        number = _bounded_float(value, label, 0, 1_000_000_000_000)
        if strictly_positive and number <= 0:
            raise ValueError(f"{label} deve ser maior que zero")
        return number

    current["profile"] = {"id": profile_id, "description": description}
    current["cost_model"] = {
        "currency": currency,
        "hardware_price": optional_number("hardware_price", "preço do hardware"),
        "electricity_per_kwh": optional_number(
            "electricity_per_kwh", "custo de energia"),
        "amortization_hours": optional_number(
            "amortization_hours", "horas de amortização", strictly_positive=True),
    }
    _write_yaml(path, current)
    return {"status": "ok", "hardware": current}


def _format_result_timestamp(value):
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return parsed.strftime("%d/%m %H:%M")
    except (TypeError, ValueError):
        return str(value)[:16]


def render_results_html(front_filter="all"):
    """Renderiza a tabela de resultados (cabeçalho + linhas).

    Orientada a colunas: cada métrica se descreve (rótulo + tooltip) e só é
    exibida se ao menos um resultado tiver valor. Assim cada device mostra
    apenas o que faz sentido nele — o Raspberry Pi some com GPU/VRAM/potência,
    o Jetson mantém potência e temperatura do tegrastats.
    """
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

    def cell_cpu(r):
        # Fração de CPU do device usada pela carga: núcleos do container / total.
        # Consistente com os núcleos exibidos e sempre 0-100 (nunca 600% do cgroup).
        cores = r.get("cpu_cores_used")
        total = r.get("cpu_cores_total")
        if cores is not None and total:
            pct = min(100.0, cores / total * 100.0)
            return f"{fmt(pct, 0)}% ({fmt(cores, 1)}c)"
        host_cpu = r.get("avg_host_cpu_pct")
        if host_cpu is not None:
            return fmt(host_cpu, 0) + "%" + (f" ({fmt(cores, 1)}c)" if cores is not None else "")
        if cores is not None:
            return fmt(cores, 1) + " cores"
        return "—"

    def cell_temp(r):
        if r.get("peak_temp_c") is None:
            return "—"
        return fmt(r.get("peak_temp_c"), 0) + "°C" + (" (lim.)" if r.get("thermal_throttling_observed") else "")

    def cell_cost(r):
        if r.get("cost_per_1k") is None:
            return "—"
        return f"{r.get('cost_currency') or ''} {fmt(r.get('cost_per_1k'), 6)}".strip()

    def unit(field, suffix, dec=0):
        def f(rr):
            v = rr.get(field)
            return (fmt(v, dec) + suffix) if v is not None else "—"
        return f

    # (chave, rotulo, tooltip, largura_px, sempre_exibir, funcao-da-celula)
    COLS = [
        ("ts", "Quando", "Horario em que o benchmark rodou.", 88, True,
         lambda r: _format_result_timestamp(r.get("timestamp", ""))),
        ("front", "Frente", "Carga: vision (YOLO), slm (LLM) ou audio (ASR+TTS).", 56, True,
         lambda r: str(r.get("front") or "?")),
        ("model", "Modelo", "Modelo executado nesta medicao.", 150, True,
         lambda r: str(r.get("model") or "?").replace(".gguf", "")[:32]),
        ("profile", "Perfil", "quick/standard = desempenho; soak/torture = estresse sustentado.", 62, True,
         lambda r: str(r.get("profile") or "?")),
        ("primary", "Primaria", "Metrica principal: frames/s (visao), tokens/s (SLM), x tempo real (audio). Total produzido dividido pela duracao da fase.", 92, True,
         lambda r: (f"{fmt((r.get('primary_metric') or {}).get('value'), 2)} "
                    f"{(r.get('primary_metric') or {}).get('unit') or ''}".strip()
                    if isinstance(r.get("primary_metric"), dict)
                    and (r.get('primary_metric') or {}).get('value') is not None else "—")),
        ("thr", "ops/s", "Throughput: inferencias concluidas por segundo, fora o warmup.", 74, True,
         lambda r: fmt(r.get("throughput_ops"), 2)),
        ("p50", "p50ms", "Latencia mediana das amostras individuais.", 56, True,
         lambda r: fmt(r.get("p50_ms"))),
        ("p95", "p95ms", "Latencia no percentil 95.", 56, False,
         lambda r: fmt(r.get("p95_ms"))),
        ("p99", "p99ms", "Latencia no percentil 99.", 56, True,
         lambda r: fmt(r.get("p99_ms"))),
        ("std", "σ ms", "Desvio padrao da latencia entre as amostras.", 52, False,
         lambda r: fmt(r.get("std_ms"))),
        ("ram", "RAM", "Pico de RAM da carga (working set do cgroup, ou RSS da arvore de processos).", 66, True,
         unit("peak_ram_mb", "MB")),
        ("ramd", "Δ RAM", "Pico de RAM menos o baseline coletado antes da fase.", 60, False,
         unit("peak_ram_delta_mb", "MB")),
        ("cpu", "CPU", "Fracao de CPU do device usada pela carga (nucleos do container / total de nucleos, 0-100%); entre parenteses, os nucleos ocupados (ex.: 5.9c de 6).", 80, True,
         cell_cpu),
        ("gpu", "GPU%", "Pico de uso de GPU. Zero ou ausente quando a carga roda em CPU.", 52, False,
         unit("peak_gpu_util", "%")),
        ("accel", "Acel.%", "Pico de uso de NPU/TPU/VPU quando ha acelerador.", 52, False,
         unit("peak_accelerator_util", "%")),
        ("vram", "VRAM", "Pico de VRAM (so em GPU dedicada).", 62, False,
         unit("peak_vram_mb", "MB")),
        ("temp", "Temp", "Maior temperatura na fase; (lim.) indica limitacao termica ativa.", 58, False,
         cell_temp),
        ("power", "Potencia", "Potencia media. Entrada da placa (VDD_IN) no Jetson; ausente sem sensor (ex.: Raspberry Pi).", 66, False,
         unit("avg_power_w", "W", 1)),
        ("energy", "Energia", "Energia dos componentes medidos na fase de recursos.", 68, False,
         unit("energy_j", "J", 1)),
        ("cov", "Cob.", "Fracao da fase coberta pela telemetria de energia; abaixo de 90% vira estimativa.", 52, False,
         lambda r: (fmt(float(r["energy_coverage_ratio"]) * 100.0, 0) + "%")
         if r.get("energy_coverage_ratio") is not None else "—"),
        ("jperu", "J/unid.", "Energia por unidade principal (frame, token ou segundo de audio).", 68, False,
         lambda r: fmt(r.get("energy_per_primary_unit_j"), 5)),
        ("cost", "Custo/1k", "Custo estimado por mil unidades (amortizacao do hardware + energia da aba Hardware).", 78, False,
         cell_cost),
    ]

    computed = [[(c, c[5](r)) for c in COLS] for r in results]
    has_data = {c[0]: False for c in COLS}
    for row in computed:
        for col, val in row:
            if val not in (None, "—", ""):
                has_data[col[0]] = True
    active = [c for c in COLS if c[4] or has_data[c[0]]]

    def esc(v):
        return html.escape(str(v), quote=True)

    header_cells = "".join(
        f'<div class="res-h metric-help" style="flex:0 0 {c[3]}px" data-tip="{esc(c[2])}">{esc(c[1])}</div>'
        for c in active)
    header = f'<div class="res-head">{header_cells}</div>'

    body_rows = []
    for r, row in zip(results, computed):
        vals = {c[0]: v for c, v in row}
        front = str(r.get("front") or "?")
        fc = "#93c5fd" if front == "vision" else "#22c55e" if front == "slm" else "#f59e0b"
        cells = []
        for c in active:
            key = c[0]
            val = vals[key]
            style = f"flex:0 0 {c[3]}px"
            if key == "front":
                cells.append(f'<div style="{style}"><span style="background:#202020;color:{fc};'
                             f'padding:1px 7px;border-radius:5px;font-size:.68rem;font-weight:700">{esc(val)}</span></div>')
            elif key == "model":
                cells.append(f'<div style="flex:1;min-width:140px;overflow:hidden;text-overflow:ellipsis;'
                             f'white-space:nowrap" title="{esc(r.get("model") or "")}">{esc(val)}</div>')
            elif key in ("ts", "profile"):
                cells.append(f'<div style="{style};color:#a3a3a3;font-size:.72rem">{esc(val)}</div>')
            elif key in ("primary", "thr"):
                cells.append(f'<div style="{style};color:#14b8a6;font-weight:700">{esc(val)}</div>')
            else:
                cells.append(f'<div style="{style}">{esc(val)}</div>')
        body_rows.append(f'<div class="res-row">{"".join(cells)}</div>')

    return header + "\n" + "\n".join(body_rows)

def all_results():
    files = sorted(glob.glob(f"{RESULTS_DIR}/*.json"), key=os.path.getmtime, reverse=True)
    hardware_path = os.path.join(BASE, "config", "hardware.yaml")
    try:
        hardware_stamp = os.stat(hardware_path).st_mtime_ns
    except OSError:
        hardware_stamp = 0
    hardware_config = load_hardware_config(hardware_path)
    out = []
    seen_results = set()
    active_files = set(files)
    for path in files:
        cache_key = None
        try:
            stat = os.stat(path)
            cache_key = (stat.st_mtime_ns, stat.st_size, hardware_stamp)
            with _result_cache_lock:
                cached = _result_cache.get(path)
            if cached and cached[0] == cache_key:
                row = cached[1]
                if row is None:
                    continue
            else:
                data, row = load_result_file(path)
                if not isinstance(data, dict):
                    raise ValueError("o JSON raiz deve ser um objeto")
                if not is_benchmark_result(data):
                    with _result_cache_lock:
                        _result_cache[path] = (cache_key, None)
                    continue
                data["scorecard"] = compute_scorecard(data, hardware_config)
                row = normalize_result(data, filename=path)
                with open(path, "rb") as result_file:
                    row["source_sha256"] = hashlib.sha256(result_file.read()).hexdigest()
                with _result_cache_lock:
                    _result_cache[path] = (cache_key, row)
            identity = (
                "run_id", row["run_id"]
            ) if row.get("run_id") else ("source_sha256", row.get("source_sha256"))
            if identity[1] and identity in seen_results:
                continue
            seen_results.add(identity)
            out.append(row)
        except Exception as exc:
            with _result_cache_lock:
                _result_cache[path] = (cache_key, None)
            print(f"[results] {os.path.basename(path)} ignorado: {exc}")

    with _result_cache_lock:
        for path in list(_result_cache):
            if path not in active_files:
                del _result_cache[path]
    return out


def scorecard_summary():
    """Resume os scorecards disponíveis para comparação."""
    rows = all_results()
    by_front = {}
    for row in rows:
        front = row.get("front") or "unknown"
        by_front.setdefault(front, []).append(row)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "fronts": {},
        "results": rows,
    }
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
    """Ativa o modelo e inicia o benchmark em segundo plano."""
    port = CONTAINERS.get(front)
    if not port:
        return False
    with _lock:
        if _benchmark_status.get(front, {}).get("running"):
            return False
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
                timeout=_benchmark_timeout_sec(),
            )
            with _lock:
                _benchmark_status[front] = {"running": False, "progress": "done", "result": result}
        except Exception as e:
            with _lock:
                _benchmark_status[front] = {"running": False, "progress": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return True


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
        # A duração do WAV permite calcular o fator de tempo real.
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


DASHBOARD_PATH = os.path.join(BASE, "web", "dashboard.html")


def load_dashboard_html():
    with open(DASHBOARD_PATH, encoding="utf-8") as f:
        return f.read()



class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")

    def _json_response(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._security_headers()
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _html_response(self, html):
        body = html.encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                "media-src 'self' data: blob:; connect-src 'self'; object-src 'none'; "
                "frame-ancestors 'none'",
            )
            self._security_headers()
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 32 * 1024 * 1024:
            raise ValueError("corpo da requisição excede 32 MB")
        data = json.loads(self.rfile.read(length)) if length else {}
        if not isinstance(data, dict):
            raise ValueError("o corpo JSON deve ser um objeto")
        return data

    def do_OPTIONS(self):
        self.send_response(405)
        self._security_headers()
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
                "telemetry": host_telemetry_metrics(),
                "system": system_metrics(),
                "containers": {k: container_health(v) for k, v in CONTAINERS.items()},
                "results": all_results(),
                "benchmark_status": bs,
                "timestamp": datetime.now().astimezone().isoformat(),
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

        elif path == "/api/containers":
            self._json_response(container_runtime_info())

        elif path == "/api/hardware-config":
            try:
                self._json_response(hardware_config_api())
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

        elif path == "/api/profiles":
            try:
                self._json_response(benchmark_profiles_api())
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

        elif path == "/api/hardware-detect":
            try:
                query = parse_qs(urlparse(self.path).query)
                refresh = query.get("refresh", ["0"])[0] == "1"
                self._json_response(detected_hardware_api(refresh=refresh))
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

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
        try:
            data = self._read_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._json_response({"error": str(exc)}, 400)
            return

        if path == "/api/benchmark":
            front = data.get("front", "")
            profile = data.get("profile", "quick")
            model = data.get("model")
            mode = data.get("mode")
            if front not in CONTAINERS:
                self._json_response({"error": f"unknown front: {front}"}, 400)
                return
            try:
                benchmark_config = _load_yaml(os.path.join(BASE, "config", "benchmark.yaml"))
                if profile not in benchmark_config.get("profiles", {}):
                    raise ValueError(f"perfil '{profile}' não encontrado")
                if front == "audio" and mode not in ("asr", "tts", "pipeline"):
                    raise ValueError("modo de áudio deve ser asr, tts ou pipeline")
            except Exception as exc:
                self._json_response({"error": str(exc)}, 400)
                return
            if not trigger_benchmark(front, profile, model=model, mode=mode):
                self._json_response({"error": "already running"}, 409)
                return
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

        elif path == "/api/containers/update":
            try:
                self._json_response(update_container_limits(data))
            except ValueError as exc:
                self._json_response({"error": str(exc)}, 400)
            except RuntimeError as exc:
                self._json_response({"error": str(exc)}, 409)
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

        elif path == "/api/containers/restart":
            try:
                self._json_response(restart_container(str(data.get("front", ""))))
            except ValueError as exc:
                self._json_response({"error": str(exc)}, 400)
            except RuntimeError as exc:
                self._json_response({"error": str(exc)}, 409)

        elif path == "/api/hardware-config":
            try:
                self._json_response(save_hardware_config(data))
            except ValueError as exc:
                self._json_response({"error": str(exc)}, 400)
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

        elif path == "/api/inference":
            try:
                prompt = str(data.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("prompt vazio")
                if len(prompt) > 100_000:
                    raise ValueError("prompt excede 100.000 caracteres")
                max_tokens = _bounded_integer(
                    data.get("max_tokens", 256), "max_tokens", 1, 4096)
                result = do_slm_inference(prompt, max_tokens)
                self._json_response(result, 200 if not result.get("error") else 502)
            except ValueError as exc:
                self._json_response({"error": str(exc)}, 400)

        elif path == "/api/vision-infer":
            try:
                image_b64 = data.get("image_base64", "")
                image_bytes = base64.b64decode(image_b64, validate=True) if image_b64 else b""
                if not image_bytes:
                    self._json_response({"error": "imagem vazia"}, 400)
                    return
                if len(image_bytes) > 20 * 1024 * 1024:
                    raise ValueError("imagem excede 20 MB")
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
            text = str(data.get("text", "")).strip()
            if not text:
                self._json_response({"error": "texto vazio"}, 400)
                return
            if len(text) > 5_000:
                self._json_response({"error": "texto excede 5.000 caracteres"}, 400)
                return
            result = do_tts(text)
            self._json_response(result, 200 if not result.get("error") else 502)

        elif path == "/api/config":
            yaml_text = data.get("yaml_text", "")
            cfg_path = os.path.join(BASE, "config", "benchmark.yaml")
            try:
                import yaml
                parsed = yaml.safe_load(yaml_text)
                validate_benchmark_config(parsed)
                with _config_lock:
                    _atomic_write_text(cfg_path, yaml_text)
                self._json_response({"status": "ok"})
            except Exception as e:
                self._json_response({"error": str(e)}, 400)

        else:
            self.send_response(404)
            self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    host = os.environ.get("PDC_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("PDC_DASHBOARD_PORT", "8099"))
    server = ThreadedHTTPServer((host, port), Handler)
    print(f"PDC-IA Dashboard v1 -> http://{host}:{port}")
    print("API: /api/metrics, /api/benchmark, /api/inference, /api/vision-infer, /api/tts, /api/models/*")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
