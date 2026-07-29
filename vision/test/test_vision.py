"""Teste de integração do container de visão."""

import os

import numpy as np
import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.environ.get("PDC_RUN_INTEGRATION") != "1",
    reason="defina PDC_RUN_INTEGRATION=1 para testar containers ativos",
)

BASE_URL = "http://localhost:8010"

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
