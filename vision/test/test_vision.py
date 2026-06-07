"""Teste de integração do container de visão."""

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
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, f"Health retornou {r.status_code}"
    data = r.json()
    assert "status" in data
    print(f"  Health: {data}")


def test_predict():
    try:
        import cv2
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        image_bytes = encoded.tobytes()
    except ImportError:
        image_bytes = b"P6\n1 1\n255\n\xff\xff\xff"

    files = {"file": ("test.ppm", image_bytes, "image/x-portable-pixmap")}
    r = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    assert r.status_code == 200, f"Predict retornou {r.status_code}: {r.text}"
    data = r.json()
    assert "detections" in data, "Resposta não contém 'detections'"
    assert "count" in data
    print(f"  Predict OK: {data['count']} detecções, device={data.get('device')}")


def test_info():
    r = requests.get(f"{BASE_URL}/info", timeout=10)
    assert r.status_code == 200, f"Info retornou {r.status_code}"
    print(f"  Info OK: model={r.json().get('model', {}).get('name')}")


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
