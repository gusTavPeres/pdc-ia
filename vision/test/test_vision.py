"""
test_vision.py — Teste de integração do container de visão.

Gera imagem aleatória, envia para /predict e verifica resposta 200.
"""

import io
import sys
import time
import numpy as np
import requests

BASE_URL = "http://localhost:8010"


def wait_for_health(max_wait=300):
    """Aguarda o container ficar saudável."""
    print(f"Aguardando container ficar saudável (máx {max_wait}s)...")
    for i in range(max_wait // 5):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"Container saudável após {i*5}s")
                return True
        except Exception:
            pass
        time.sleep(5)
        if i % 12 == 0:
            print(f"  {i*5}s decorridos...")
    return False


def test_health():
    """Testa endpoint /health."""
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, f"Health retornou {r.status_code}"
    data = r.json()
    assert "status" in data
    print(f"  Health: {data}")
    return True


def test_predict():
    """Testa detecção de objetos com imagem sintética."""
    # Gera imagem JPEG 640x640 em memória
    try:
        import cv2
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        image_bytes = encoded.tobytes()
    except ImportError:
        # Sem cv2: cria imagem JPEG mínima válida
        # JPEG header mínimo (imagem 1x1 branca)
        image_bytes = (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e\xbb\x00\x0b'
        )

    files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
    r = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    assert r.status_code == 200, f"Predict retornou {r.status_code}: {r.text}"
    data = r.json()
    assert "detections" in data, "Resposta não contém 'detections'"
    assert "count" in data
    print(f"  Predict OK: {data['count']} detecções, device={data.get('device')}")
    return True


def test_info():
    """Testa endpoint /info."""
    r = requests.get(f"{BASE_URL}/info", timeout=10)
    assert r.status_code == 200, f"Info retornou {r.status_code}"
    print(f"  Info OK: model={r.json().get('model', {}).get('name')}")
    return True


if __name__ == "__main__":
    print("=== Teste Vision Container ===")

    if not wait_for_health():
        print("FALHA: Container não ficou saudável a tempo")
        sys.exit(1)

    tests = [test_health, test_predict, test_info]
    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            print(f"\nTestando {test_fn.__name__}...")
            test_fn()
            print(f"  PASSOU")
            passed += 1
        except Exception as e:
            print(f"  FALHOU: {e}")
            failed += 1

    print(f"\nResultado: {passed} passou, {failed} falhou")
    sys.exit(0 if failed == 0 else 1)
