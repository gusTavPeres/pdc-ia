"""
inference_asr.py — Transcrição de áudio com Whisper (faster-whisper).

Carrega modelo Whisper e transcreve áudio WAV para texto.
Detecta GPU automaticamente, fallback para CPU.
"""

import io
import os
import sys
import time
from typing import Optional

sys.path.insert(0, "/app")


# Variáveis globais do modelo ASR
_asr_model = None
_asr_device = "cpu"
_asr_compute_type = "int8"


def load_asr(config: dict) -> None:
    """
    Carrega modelo Whisper via faster-whisper.

    Detecta CUDA automaticamente:
    - GPU disponível: device="cuda", compute_type="float16"
    - CPU: device="cpu", compute_type="int8"

    Args:
        config: Dict com "model_id" (ex: "small") e opções.
    """
    global _asr_model, _asr_device, _asr_compute_type

    from faster_whisper import WhisperModel

    model_id = config.get("model_id", "small")

    # Detecta dispositivo
    gpu_mode = os.environ.get("PDC_GPU_MODE", "auto")
    try:
        import torch
        if torch.cuda.is_available() and gpu_mode != "cpu":
            _asr_device = "cuda"
            _asr_compute_type = "float16"
        else:
            _asr_device = "cpu"
            _asr_compute_type = "int8"
    except ImportError:
        _asr_device = "cpu"
        _asr_compute_type = "int8"

    # Tenta GPU, fallback para CPU
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
    """
    Transcreve áudio para texto.

    Args:
        audio_bytes: Áudio em formato WAV (bytes).

    Retorna:
        dict com:
            text: transcrição completa
            language: idioma detectado
            rtf: Real Time Factor (tempo_processamento / duração_áudio)
            duration_sec: duração do áudio em segundos
            segments: lista de segmentos com timestamps
    """
    global _asr_model

    if _asr_model is None:
        raise RuntimeError("Modelo ASR não carregado")

    # Carrega áudio de bytes
    audio_io = io.BytesIO(audio_bytes)

    t0 = time.perf_counter()

    # Transcrição com faster-whisper
    segments_iter, info = _asr_model.transcribe(
        audio_io,
        language=None,          # Detecta idioma automaticamente
        task="transcribe",
        beam_size=5,
        best_of=5,
        vad_filter=True,        # Remove silêncio
        vad_parameters={"min_silence_duration_ms": 500},
    )

    # Coleta segmentos (iterador lazy)
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

    # RTF: tempo de processamento / duração do áudio
    # RTF < 1.0 = mais rápido que tempo real
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
