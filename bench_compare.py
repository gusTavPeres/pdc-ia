#!/usr/bin/env python3
"""Comparação entre modelos SLM — TinyLlama vs Qwen2.5."""
import json, urllib.request, time

BASE = "http://localhost:8011"

def post(path, payload, timeout=600):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())

def show_result(r):
    b = r.get("benchmark_results", {})
    tp = b.get("throughput", {})
    lat = b.get("latency", {})
    res = b.get("resource_usage", {})
    print(f"  Throughput:    {tp.get('ops_per_sec'):.3f} ops/s")
    print(f"  Latência p50:  {lat.get('p50_ms')} ms")
    print(f"  Latência p99:  {lat.get('p99_ms')} ms")
    print(f"  Desvio padrão: {lat.get('std_ms')} ms")
    print(f"  RAM pico:      {res.get('peak_ram_mb')} MB")
    print(f"  CPU médio:     {res.get('avg_cpu_pct')}%")
    print(f"  Temp pico:     {res.get('peak_temp_c')}°C")

modelos = [
    ("tinyllama-1.1b-q4", "TinyLlama 1.1B Chat Q4"),
    ("qwen25-0.5b-q4",    "Qwen2.5 0.5B Instruct Q4"),
]
results = {}

for model_key, model_label in modelos:
    print(f"\n{'='*55}")
    print(f"MODELO: {model_label}")
    print(f"{'='*55}")

    # Troca modelo (pode demorar se precisar baixar)
    print(f"Carregando...")
    info = post("/switch-model", {"model": model_key}, timeout=600)
    loaded = info["loaded"]
    print(f"  {loaded['name']} | {loaded['device'].upper()} | {loaded['gpu_layers']} layers GPU")

    # Exemplo de geração
    pred = post("/predict", {"prompt": "O que é uma rede neural convolucional?", "max_tokens": 60})
    print(f"\nResposta exemplo:")
    print(f"  \"{pred['response'][:150]}\"")
    print(f"  ({pred['total_tokens']} tokens a {pred['tokens_per_sec']} t/s)")

    # Benchmark
    print(f"\nBenchmark quick...")
    r = post("/benchmark", {"profile": "quick"}, timeout=300)
    results[model_label] = r
    show_result(r)

# Comparação final
print(f"\n{'='*55}")
print("RESUMO COMPARATIVO")
print(f"{'='*55}")
header = f"{'Métrica':<22} {'TinyLlama 1.1B':>18} {'Qwen2.5 0.5B':>18}"
print(header)
print("-" * len(header))

labels = list(results.keys())
if len(labels) == 2:
    r0 = results[labels[0]].get("benchmark_results", {})
    r1 = results[labels[1]].get("benchmark_results", {})
    rows = [
        ("Throughput (ops/s)",  r0.get("throughput",{}).get("ops_per_sec"), r1.get("throughput",{}).get("ops_per_sec")),
        ("Latência p50 (ms)",   r0.get("latency",{}).get("p50_ms"),         r1.get("latency",{}).get("p50_ms")),
        ("Latência p99 (ms)",   r0.get("latency",{}).get("p99_ms"),         r1.get("latency",{}).get("p99_ms")),
        ("Desvio padrão (ms)",  r0.get("latency",{}).get("std_ms"),         r1.get("latency",{}).get("std_ms")),
        ("RAM pico (MB)",       r0.get("resource_usage",{}).get("peak_ram_mb"), r1.get("resource_usage",{}).get("peak_ram_mb")),
        ("CPU médio (%)",       r0.get("resource_usage",{}).get("avg_cpu_pct"), r1.get("resource_usage",{}).get("avg_cpu_pct")),
        ("Temp pico (°C)",      r0.get("resource_usage",{}).get("peak_temp_c"), r1.get("resource_usage",{}).get("peak_temp_c")),
    ]
    for label, v0, v1 in rows:
        v0s = f"{v0}" if v0 is not None else "N/A"
        v1s = f"{v1}" if v1 is not None else "N/A"
        print(f"{label:<22} {v0s:>18} {v1s:>18}")
