"""
test_slm.py — Teste de integração do container SLM.
"""

import sys
import time
import requests

BASE_URL = "http://localhost:8011"


def wait_for_health(max_wait=400):
    """Aguarda container ficar saudável (modelo ~637MB demora para baixar)."""
    print(f"Aguardando SLM ficar saudável (máx {max_wait}s)...")
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


if __name__ == "__main__":
    print("=== Teste SLM Container ===")

    if not wait_for_health():
        print("FALHA: Container não ficou saudável")
        sys.exit(1)

    tests = [test_health, test_predict, test_info]
    passed = failed = 0

    for fn in tests:
        try:
            print(f"\nTestando {fn.__name__}...")
            fn()
            print("  PASSOU")
            passed += 1
        except Exception as e:
            print(f"  FALHOU: {e}")
            failed += 1

    print(f"\nResultado: {passed} passou, {failed} falhou")
    sys.exit(0 if failed == 0 else 1)
