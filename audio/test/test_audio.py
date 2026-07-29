"""Teste de integração do container de áudio."""

import os

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.environ.get("PDC_RUN_INTEGRATION") != "1",
    reason="defina PDC_RUN_INTEGRATION=1 para testar containers ativos",
)

BASE_URL = "http://localhost:8012"

def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    print(f"  Health: {data}")


def request_tts_audio():
    payload = {"text": "Teste de síntese de voz em português."}
    r = requests.post(f"{BASE_URL}/tts", json=payload, timeout=30)
    assert r.status_code == 200, f"TTS retornou {r.status_code}: {r.text}"
    assert len(r.content) > 44, "WAV muito pequeno (provavelmente vazio)"
    print(f"  TTS OK: {len(r.content)} bytes de áudio")
    return r.content


def test_tts():
    request_tts_audio()


def test_asr():
    """Testa transcrição de áudio."""
    wav_bytes = request_tts_audio()

    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    r = requests.post(f"{BASE_URL}/asr", files=files, timeout=60)
    assert r.status_code == 200, f"ASR retornou {r.status_code}: {r.text}"
    data = r.json()
    assert "text" in data, "Sem campo 'text'"
    print(f"  ASR OK: texto='{data['text'][:80]}', rtf={data.get('rtf')}")
