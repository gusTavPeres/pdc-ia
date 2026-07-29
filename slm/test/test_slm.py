"""
test_slm.py — Teste de integração do container SLM.
"""

import os

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.environ.get("PDC_RUN_INTEGRATION") != "1",
    reason="defina PDC_RUN_INTEGRATION=1 para testar containers ativos",
)

BASE_URL = "http://localhost:8011"

def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok", f"Status não ok: {data}"
    print(f"  Health: {data}")


def test_predict():
    payload = {"prompt": "O que é inteligência artificial?", "max_tokens": 50}
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=120)
    assert r.status_code == 200, f"Predict retornou {r.status_code}: {r.text}"
    data = r.json()
    assert "response" in data, "Sem campo 'response'"
    assert len(data["response"]) > 0, "Resposta vazia"
    print(f"  Predict OK: {data['total_tokens']} tokens, {data['tokens_per_sec']} t/s")
    print(f"  Resposta: {data['response'][:100]}...")


def test_info():
    r = requests.get(f"{BASE_URL}/info", timeout=10)
    assert r.status_code == 200
    print(f"  Info OK: model={r.json().get('model', {}).get('name')}")
