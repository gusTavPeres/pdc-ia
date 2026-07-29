"""Síntese de voz com Piper TTS."""

import io
import os
import sys
import threading
import time
import wave
from functools import wraps

sys.path.insert(0, "/app")

from shared.model_loader import get_or_download

# Variáveis globais do TTS
_tts_voice = None
_tts_model_path = None
_tts_config_path = None
_tts_lock = threading.RLock()


def _serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _tts_lock:
            return function(*args, **kwargs)
    return wrapped


@_serialized
def load_tts(config: dict) -> None:
    """Baixa os artefatos ONNX e inicializa o Piper."""
    global _tts_voice, _tts_model_path, _tts_config_path

    model_dir = "/app/data/models"
    os.makedirs(model_dir, exist_ok=True)

    # Baixa arquivo de configuração JSON
    config_url = config.get("config_url", "")
    if config_url:
        config_filename = config_url.split("/")[-1].split("?")[0]
        config_path = get_or_download(
            {"url": config_url, "filename": config_filename},
            cache_dir=model_dir
        )
    else:
        config_path = None

    # Baixa modelo ONNX
    model_url = config.get("model_url", "")
    if model_url:
        model_filename = model_url.split("/")[-1].split("?")[0]
        model_path = get_or_download(
            {
                "url": model_url,
                "filename": model_filename,
                "sha256": config.get("sha256"),
            },
            cache_dir=model_dir
        )
        print(f"[tts] Modelo TTS: {model_path}")
    else:
        model_path = None

    if not model_path:
        raise ValueError("URL do modelo TTS não configurada")

    try:
        from piper import PiperVoice
        voice = PiperVoice.load(
            model_path,
            config_path=config_path,
            use_cuda=False,
        )
        _tts_voice = voice
        _tts_model_path = model_path
        _tts_config_path = config_path
        print("[tts] PiperVoice carregado com sucesso")
    except Exception as e:
        raise RuntimeError(f"Falha ao carregar PiperVoice: {e}") from e


@_serialized
def synthesize(text: str) -> bytes:
    """Sintetiza o texto e devolve áudio WAV em memória."""
    global _tts_voice

    if _tts_voice is None:
        raise RuntimeError("Modelo TTS não carregado")

    try:
        chunks = list(_tts_voice.synthesize(text))

        if not chunks:
            raise RuntimeError("O modelo TTS não produziu áudio")

        # Usa parâmetros do primeiro chunk
        sample_rate = chunks[0].sample_rate
        sample_width = chunks[0].sample_width
        sample_channels = chunks[0].sample_channels

        # Concatena todos os bytes de áudio
        audio_bytes = b"".join(
            chunk.audio_int16_bytes for chunk in chunks
        )

        # Cria arquivo WAV em memória
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav:
            wav.setnchannels(sample_channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_bytes)

        return wav_io.getvalue()

    except Exception as e:
        raise RuntimeError(f"Falha na síntese TTS: {e}") from e


def synthesize_with_metrics(text: str) -> dict:
    started = time.perf_counter()
    audio = synthesize(text)
    processing_sec = time.perf_counter() - started
    with wave.open(io.BytesIO(audio), "rb") as wav:
        sample_rate = wav.getframerate()
        duration_sec = wav.getnframes() / sample_rate if sample_rate else 0.0
    if duration_sec <= 0:
        raise RuntimeError("O modelo TTS produziu áudio sem duração")
    return {
        "audio_bytes": len(audio),
        "duration_sec": round(duration_sec, 4),
        "processing_sec": round(processing_sec, 4),
        "rtf": round(processing_sec / duration_sec, 4),
    }


def is_tts_loaded() -> bool:
    """Verifica se TTS está carregado."""
    return _tts_voice is not None


def get_tts_model_name() -> str:
    """Retorna o nome do arquivo do modelo TTS carregado."""
    if _tts_model_path:
        return os.path.basename(_tts_model_path)
    return "não carregado"
