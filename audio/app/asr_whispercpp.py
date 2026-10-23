"""ASR via whisper.cpp (whisper-server) com backend Vulkan (GPU AMD/iGPU).

Mesma API pública do inference_asr (faster-whisper), mas roda o whisper.cpp
compilado com GGML_VULKAN. O modelo fica residente num whisper-server local
(sobe 1x por modelo); transcribe() faz POST do WAV.
"""
import io
import os
import subprocess
import sys
import threading
import time
import urllib.request
import wave

import requests

sys.path.insert(0, "/app")
from shared.cpu_config import cpu_threads as _cpu_threads

_PORT = int(os.environ.get("PDC_WHISPERCPP_PORT", "8181"))
_BIN = os.environ.get("PDC_WHISPERCPP_BIN", "/opt/whisper.cpp/build/bin/whisper-server")
_MODELS_DIR = "/app/data/models/whispercpp"
_GGML_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin"

_proc = None
_model_id = None
_lock = threading.RLock()


def _ensure_model(size: str) -> str:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    path = os.path.join(_MODELS_DIR, f"ggml-{size}.bin")
    if not os.path.exists(path):
        url = _GGML_URL.format(size)
        print(f"[asr-vk] baixando {url}")
        urllib.request.urlretrieve(url, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def _wait_ready(timeout: int = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            requests.get(f"http://127.0.0.1:{_PORT}/", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def load_asr(config: dict) -> None:
    """Sobe (ou reinicia) o whisper-server Vulkan com o modelo pedido."""
    global _proc, _model_id
    with _lock:
        size = config.get("model_id", "small")
        model_path = _ensure_model(size)
        if _proc is not None:
            _proc.terminate()
            try:
                _proc.wait(timeout=10)
            except Exception:
                _proc.kill()
            _proc = None
        print(f"[asr-vk] subindo whisper-server {size} (Vulkan)")
        _proc = subprocess.Popen(
            [_BIN, "-m", model_path, "--host", "127.0.0.1", "--port", str(_PORT),
             "-t", str(_cpu_threads())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not _wait_ready():
            raise RuntimeError("whisper-server não respondeu")
        _model_id = size
        print(f"[asr-vk] pronto em VULKAN ({size})")


def _wav_duration(audio_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio_bytes)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def transcribe(audio_bytes: bytes) -> dict:
    with _lock:
        if _proc is None:
            raise RuntimeError("Modelo ASR não carregado")
        t0 = time.perf_counter()
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"response_format": "verbose_json", "temperature": "0"}
        r = requests.post(f"http://127.0.0.1:{_PORT}/inference",
                          files=files, data=data, timeout=600)
        r.raise_for_status()
        js = r.json()
        elapsed = time.perf_counter() - t0

    text = (js.get("text") or "").strip()
    segments = []
    for s in (js.get("segments") or []):
        st_ = s.get("start", s.get("t0", 0))
        en_ = s.get("end", s.get("t1", 0))
        try:
            st_, en_ = round(float(st_), 2), round(float(en_), 2)
        except Exception:
            st_, en_ = 0, 0
        segments.append({"start": st_, "end": en_, "text": (s.get("text") or "").strip()})
    dur = _wav_duration(audio_bytes)
    rtf = round(elapsed / dur, 4) if dur > 0 else None
    return {
        "text": text,
        "language": js.get("language"),
        "rtf": rtf,
        "duration_sec": round(dur, 2) if dur else None,
        "processing_sec": round(elapsed, 3),
        "segments": segments,
    }


def get_asr_device() -> str:
    return "vulkan" if _proc is not None else "cpu"


def is_asr_loaded() -> bool:
    return _proc is not None


def get_asr_model_name() -> str:
    return _model_id or "não carregado"
