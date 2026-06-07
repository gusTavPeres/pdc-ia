"""Transcrição de áudio com faster-whisper."""

import io
import os
import sys
import time
from typing import Optional

sys.path.insert(0, "/app")


_asr_model = None
_asr_device = "cpu"
_asr_compute_type = "int8"
_asr_model_id = None


def load_asr(config: dict) -> None:
    """Carrega o modelo ASR configurado."""
    global _asr_model, _asr_device, _asr_compute_type, _asr_model_id

    from faster_whisper import WhisperModel

    model_id = config.get("model_id", "small")
    _asr_model_id = model_id

    gpu_mode = os.environ.get("PDC_GPU_MODE", "auto")
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types and gpu_mode != "cpu":
            _asr_device = "cuda"
            _asr_compute_type = "float16"
        else:
            _asr_device = "cpu"
            _asr_compute_type = "int8"
    except Exception:
        _asr_device = "cpu"
        _asr_compute_type = "int8"

    try:
        print(f"[asr] Carregando Whisper {model_id} em {_asr_device.upper()}...")
        _asr_model = WhisperModel(
            model_id,
            device=_asr_device,
            compute_type=_asr_compute_type,
            download_root="/app/data/models",
        )
        print(f"[asr] Whisper carregado em {_asr_device.upper()}")
    except Exception as e:
        if _asr_device == "cuda":
            print(f"[asr] GPU falhou: {e}. Tentando CPU...")
            _asr_device = "cpu"
            _asr_compute_type = "int8"
            _asr_model = WhisperModel(
                model_id,
                device="cpu",
                compute_type="int8",
                download_root="/app/data/models",
            )
            print("[asr] Whisper carregado em CPU (fallback)")
        else:
            raise


def transcribe(audio_bytes: bytes) -> dict:
    """Transcreve áudio WAV e retorna texto, segmentos e tempo de execução."""
    global _asr_model

    if _asr_model is None:
        raise RuntimeError("Modelo ASR não carregado")

    audio_io = io.BytesIO(audio_bytes)
    t0 = time.perf_counter()

    try:
        segments_iter, info = _asr_model.transcribe(
            audio_io,
            language=None,
            task="transcribe",
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
    except ValueError as e:
        if "empty sequence" not in str(e):
            raise
        elapsed = time.perf_counter() - t0
        return {
            "text": "",
            "language": None,
            "rtf": None,
            "duration_sec": None,
            "processing_sec": round(elapsed, 3),
            "segments": [],
        }

    segments = []
    full_text = ""
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        full_text += seg.text

    elapsed = time.perf_counter() - t0
    audio_duration = info.duration if hasattr(info, "duration") else 0

    rtf = round(elapsed / audio_duration, 4) if audio_duration > 0 else None

    return {
        "text": full_text.strip(),
        "language": info.language if hasattr(info, "language") else None,
        "rtf": rtf,
        "duration_sec": round(audio_duration, 2),
        "processing_sec": round(elapsed, 3),
        "segments": segments,
    }


def get_asr_device() -> str:
    """Retorna dispositivo do modelo ASR."""
    return _asr_device


def is_asr_loaded() -> bool:
    """Verifica se o modelo ASR está carregado."""
    return _asr_model is not None


def get_asr_model_name() -> str:
    """Retorna o identificador do modelo ASR carregado."""
    return _asr_model_id or "não carregado"
