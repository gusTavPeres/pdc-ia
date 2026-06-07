"""
inference_tts.py — Síntese de voz com Piper TTS (pt-BR).

Usa a API correta do piper-tts 1.4+: voice.synthesize() retorna Iterable[AudioChunk].
"""

import io
import os
import sys
import wave
import struct
from typing import Optional

sys.path.insert(0, "/app")

from shared.model_loader import get_or_download


# Variáveis globais do TTS
_tts_voice = None
_tts_model_path = None
_tts_config_path = None


def load_tts(config: dict) -> None:
    """
    Carrega modelo Piper TTS.

    Baixa os arquivos .onnx e .onnx.json e inicializa PiperVoice.

    Args:
        config: Dict com "model_url" e "config_url".
    """
    global _tts_voice, _tts_model_path, _tts_config_path

    model_dir = "/app/data/models"
    os.makedirs(model_dir, exist_ok=True)

    # Baixa arquivo de configuração JSON
    config_url = config.get("config_url", "")
    if config_url:
        config_filename = config_url.split("/")[-1].split("?")[0]
        _tts_config_path = get_or_download(
            {"url": config_url, "filename": config_filename},
            cache_dir=model_dir
        )

    # Baixa modelo ONNX
    model_url = config.get("model_url", "")
    if model_url:
        model_filename = model_url.split("/")[-1].split("?")[0]
        _tts_model_path = get_or_download(
            {"url": model_url, "filename": model_filename},
            cache_dir=model_dir
        )
        print(f"[tts] Modelo TTS: {_tts_model_path}")

    # Carrega PiperVoice
    try:
        from piper import PiperVoice
        _tts_voice = PiperVoice.load(
            _tts_model_path,
            config_path=_tts_config_path,
            use_cuda=False,
        )
        print("[tts] PiperVoice carregado com sucesso")
    except Exception as e:
        print(f"[tts] ERRO ao carregar PiperVoice: {e}")
        _tts_voice = None


def synthesize(text: str) -> bytes:
    """
    Sintetiza texto em áudio WAV.

    Usa a API do piper-tts 1.4+: voice.synthesize() → Iterable[AudioChunk].
    Cada AudioChunk tem audio_int16_bytes, sample_rate, sample_channels, sample_width.

    Args:
        text: Texto para converter em voz.

    Retorna:
        bytes: Áudio em formato WAV (16-bit mono PCM).
    """
    global _tts_voice

    if _tts_voice is None:
        print("[tts] Modelo não carregado, usando fallback de silêncio")
        return _generate_silence_wav()

    try:
        # Coleta todos os chunks de áudio
        chunks = list(_tts_voice.synthesize(text))

        if not chunks:
            return _generate_silence_wav()

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
        print(f"[tts] Erro na síntese: {e}")
        return _generate_silence_wav()


def _generate_silence_wav(sample_rate: int = 22050, duration_sec: float = 1.5) -> bytes:
    """Gera WAV de silêncio como fallback."""
    n_samples = int(sample_rate * duration_sec)
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return wav_io.getvalue()


def is_tts_loaded() -> bool:
    """Verifica se TTS está carregado."""
    return _tts_voice is not None


def get_tts_model_name() -> str:
    """Retorna o nome do arquivo do modelo TTS carregado."""
    if _tts_model_path:
        return os.path.basename(_tts_model_path)
    return "não carregado"
