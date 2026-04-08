"""
test_audio.py — Teste de integração do container de áudio.
"""

import sys
import time
import requests

BASE_URL = "http://localhost:8012"


def wait_for_health(max_wait=400):
    print(f"Aguardando Audio ficar saudável (máx {max_wait}s)...")
    for i in range(max_wait // 10):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print(f"Container saudável após ~{i*10}s")
                return True
        except Exception:
            pass
        time.sleep(10)
        if i % 6 == 0:
            print(f"  {i*10}s decorridos...")
    return False


def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    print(f"  Health: {data}")


def test_tts():
    """Testa síntese de voz."""
    payload = {"text": "Teste de síntese de voz em português."}
    r = requests.post(f"{BASE_URL}/tts", json=payload, timeout=30)
    assert r.status_code == 200, f"TTS retornou {r.status_code}: {r.text}"
    assert len(r.content) > 44, "WAV muito pequeno (provavelmente vazio)"
    print(f"  TTS OK: {len(r.content)} bytes de áudio")
    return r.content


def test_asr(wav_bytes=None):
    """Testa transcrição de áudio."""
    if wav_bytes is None:
        # Gera WAV mínimo de silêncio para teste
        import io, wave, struct
        audio_io = io.BytesIO()
        with wave.open(audio_io, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            samples = [0] * 32000  # 2 segundos de silêncio
            w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        wav_bytes = audio_io.getvalue()

    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    r = requests.post(f"{BASE_URL}/asr", files=files, timeout=60)
    assert r.status_code == 200, f"ASR retornou {r.status_code}: {r.text}"
    data = r.json()
    assert "text" in data, "Sem campo 'text'"
    print(f"  ASR OK: texto='{data['text'][:80]}', rtf={data.get('rtf')}")


if __name__ == "__main__":
    print("=== Teste Audio Container ===")

    if not wait_for_health():
        print("FALHA: Container não ficou saudável")
        sys.exit(1)

    passed = failed = 0
    wav_bytes = None

    for fn in [test_health]:
        try:
            print(f"\nTestando {fn.__name__}...")
            fn()
            print("  PASSOU")
            passed += 1
        except Exception as e:
            print(f"  FALHOU: {e}")
            failed += 1

    # TTS primeiro para ter áudio para o ASR
    try:
        print("\nTestando test_tts...")
        wav_bytes = test_tts()
        print("  PASSOU")
        passed += 1
    except Exception as e:
        print(f"  FALHOU: {e}")
        failed += 1

    try:
        print("\nTestando test_asr...")
        test_asr(wav_bytes)
        print("  PASSOU")
        passed += 1
    except Exception as e:
        print(f"  FALHOU: {e}")
        failed += 1

    print(f"\nResultado: {passed} passou, {failed} falhou")
    sys.exit(0 if failed == 0 else 1)
