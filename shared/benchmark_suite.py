"""
benchmark_suite.py — Suite de benchmarks para avaliar desempenho de inferência.

Fornece funções para medir throughput, latência, uso de recursos,
carga sustentada e largura de banda de memória.

Não usa classes abstratas: tudo são funções diretas que aceitam
um callable de inferência (inference_fn).
"""

import time
import threading
import random
import statistics
from typing import Callable, List, Optional

from shared.methodology import (
    base_methodology,
    benchmark_metadata,
    result_quality,
    summarize_errors,
    summarize_numeric_result_fields,
)
from shared.scorecard import compute_scorecard, load_hardware_config

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import pynvml
    _PYNVML = True
except ImportError:
    _PYNVML = False

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


def _collect_system_metrics() -> dict:
    """
    Coleta uma snapshot de métricas do sistema neste momento.
    Retorna dicionário com ram_mb, cpu_pct, gpu_pct, gpu_mem_mb, temp_c.
    """
    metrics = {
        "ram_mb": None,
        "cpu_pct": None,
        "gpu_pct": None,
        "gpu_mem_mb": None,
        "temp_c": None,
        "power_w": None,
    }

    if _PSUTIL:
        try:
            mem = psutil.virtual_memory()
            metrics["ram_mb"] = round((mem.total - mem.available) / (1024**2), 1)
            metrics["cpu_pct"] = psutil.cpu_percent(interval=None)
        except Exception:
            pass

    # Coleta GPU via pynvml
    if _PYNVML:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_pct"] = util.gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["gpu_mem_mb"] = round(mem_info.used / (1024**2), 1)
            # Temperatura GPU
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics["temp_c"] = temp
            try:
                metrics["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 3)
            except Exception:
                pass
            pynvml.nvmlShutdown()
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # Temperatura CPU via sysfs (fallback)
    if metrics["temp_c"] is None:
        try:
            import os
            thermal_base = "/sys/class/thermal"
            readings = []
            if os.path.exists(thermal_base):
                for zone in os.listdir(thermal_base):
                    if not zone.startswith("thermal_zone"):
                        continue
                    temp_path = f"{thermal_base}/{zone}/temp"
                    try:
                        with open(temp_path) as f:
                            readings.append(int(f.read().strip()) / 1000.0)
                    except Exception:
                        continue
            if readings:
                metrics["temp_c"] = round(max(readings), 1)
        except Exception:
            pass

    return metrics


def run_throughput(inference_fn: Callable, duration_sec: int = 60, warmup: int = 5) -> dict:
    """
    Mede throughput de inferência: quantas operações por segundo o modelo executa.

    Chama inference_fn em loop por duration_sec segundos.
    Descarta as primeiras 'warmup' chamadas para aquecimento do modelo/cache.

    Args:
        inference_fn: Função sem argumentos que executa uma inferência.
        duration_sec: Duração do benchmark em segundos (sem warmup).
        warmup: Número de chamadas de aquecimento descartadas antes de medir.

    Retorna:
        dict com:
            ops_per_sec: operações por segundo (float)
            ops_total: total de operações realizadas (int)
            duration_sec: duração real medida (float)
    """
    # Fase de aquecimento: roda inference_fn sem medir
    warmup_errors = []
    for _ in range(warmup):
        try:
            inference_fn()
        except Exception as e:
            warmup_errors.append(f"{type(e).__name__}: {e}")

    # Fase de medição
    ops_total = 0
    error_count = 0
    errors = []
    result_samples = []
    start_time = time.perf_counter()
    end_time = start_time + duration_sec

    while time.perf_counter() < end_time:
        try:
            result = inference_fn()
            ops_total += 1
            if isinstance(result, dict):
                result_samples.append(result)
        except Exception as e:
            error_count += 1
            if len(errors) < 20:
                errors.append(f"{type(e).__name__}: {e}")

    actual_duration = time.perf_counter() - start_time
    ops_per_sec = ops_total / actual_duration if actual_duration > 0 else 0

    quality = result_quality(ops_total, error_count)
    return {
        "ops_per_sec": round(ops_per_sec, 3),
        "ops_total": ops_total,
        "duration_sec": round(actual_duration, 2),
        "warmup": {
            "attempts": warmup,
            "error_count": len(warmup_errors),
            "errors": summarize_errors(warmup_errors),
        },
        "error_count": error_count,
        "error_rate": quality["error_rate"],
        "quality": quality,
        "errors": summarize_errors(errors),
        "result_metrics": summarize_numeric_result_fields(result_samples),
    }


def run_latency(inference_fn: Callable, samples: int = 50,
                percentiles: List[int] = None) -> dict:
    """
    Mede a latência de cada inferência individual.

    Executa inference_fn 'samples' vezes, mede o tempo de cada uma.
    Calcula percentis, média e desvio padrão.

    Args:
        inference_fn: Função sem argumentos que executa uma inferência.
        samples: Número de amostras a coletar.
        percentiles: Lista de percentis a calcular (ex: [50, 95, 99]).

    Retorna:
        dict com p50_ms, p95_ms, p99_ms (e outros percentis), mean_ms, std_ms
    """
    if percentiles is None:
        percentiles = [50, 95, 99]

    latencies_ms = []
    error_count = 0
    errors = []
    result_samples = []

    for i in range(samples):
        try:
            t0 = time.perf_counter()
            result = inference_fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)
            if isinstance(result, dict):
                result_samples.append(result)
        except Exception as e:
            error_count += 1
            if len(errors) < 20:
                errors.append(f"{type(e).__name__}: {e}")

    if not latencies_ms:
        return {"error": "Nenhuma amostra coletada"}

    # Ordena para cálculo de percentis
    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    result = {}

    # Calcula cada percentil solicitado
    for p in percentiles:
        # Índice com interpolação linear
        idx = (p / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        value = sorted_latencies[lower] * (1 - frac) + sorted_latencies[upper] * frac
        result[f"p{p}_ms"] = round(value, 2)

    result["mean_ms"] = round(statistics.mean(latencies_ms), 2)
    result["std_ms"] = round(statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0, 2)
    result["samples_collected"] = len(latencies_ms)
    result["samples_requested"] = samples
    result["error_count"] = error_count
    result["error_rate"] = round(error_count / samples, 4) if samples else 0
    result["quality"] = result_quality(len(latencies_ms), error_count)
    result["errors"] = summarize_errors(errors)
    result["result_metrics"] = summarize_numeric_result_fields(result_samples)

    return result


def run_resource_usage(inference_fn: Callable, duration_sec: int = 120,
                       interval_sec: float = 1.0) -> dict:
    """
    Monitora uso de recursos enquanto o modelo roda em paralelo.

    Thread 1: executa inference_fn em loop pelo tempo definido.
    Thread 2: coleta métricas do sistema a cada interval_sec.

    Args:
        inference_fn: Função sem argumentos que executa uma inferência.
        duration_sec: Duração total do monitoramento em segundos.
        interval_sec: Intervalo entre coletas de métricas.

    Retorna:
        dict com:
            timeseries: lista de snapshots com timestamp, ram_mb, cpu_pct, gpu_pct, temp_c
            peak_ram_mb: pico de RAM usada
            avg_cpu_pct: média de uso de CPU
            peak_temp_c: temperatura máxima registrada
    """
    # Dados compartilhados entre threads
    timeseries = []
    stop_event = threading.Event()
    worker_errors = []
    worker_success = 0

    def inference_worker():
        """Thread que roda inferência continuamente."""
        nonlocal worker_success
        while not stop_event.is_set():
            try:
                inference_fn()
                worker_success += 1
            except Exception as e:
                if len(worker_errors) < 20:
                    worker_errors.append(f"{type(e).__name__}: {e}")

    def metrics_worker():
        """Thread que coleta métricas periodicamente."""
        start = time.time()
        while not stop_event.is_set():
            snapshot = _collect_system_metrics()
            snapshot["timestamp"] = round(time.time() - start, 2)
            timeseries.append(snapshot)
            time.sleep(interval_sec)

    # Inicia as duas threads
    t_infer = threading.Thread(target=inference_worker, daemon=True)
    t_metrics = threading.Thread(target=metrics_worker, daemon=True)

    t_infer.start()
    t_metrics.start()

    # Aguarda duração total
    time.sleep(duration_sec)
    stop_event.set()

    t_infer.join(timeout=5)
    t_metrics.join(timeout=5)

    # Calcula agregações
    ram_values = [s["ram_mb"] for s in timeseries if s.get("ram_mb") is not None]
    cpu_values = [s["cpu_pct"] for s in timeseries if s.get("cpu_pct") is not None]
    temp_values = [s["temp_c"] for s in timeseries if s.get("temp_c") is not None]
    gpu_values = [s["gpu_pct"] for s in timeseries if s.get("gpu_pct") is not None]
    vram_values = [s["gpu_mem_mb"] for s in timeseries if s.get("gpu_mem_mb") is not None]
    power_values = [s["power_w"] for s in timeseries if s.get("power_w") is not None]
    total_energy_j = None
    if power_values:
        total_energy_j = statistics.mean(power_values) * duration_sec

    return {
        "timeseries": timeseries,
        "peak_ram_mb": round(max(ram_values), 1) if ram_values else None,
        "avg_cpu_pct": round(statistics.mean(cpu_values), 1) if cpu_values else None,
        "peak_gpu_util": round(max(gpu_values), 1) if gpu_values else None,
        "peak_vram_mb": round(max(vram_values), 1) if vram_values else None,
        "peak_temp_c": round(max(temp_values), 1) if temp_values else None,
        "avg_power_w": round(statistics.mean(power_values), 3) if power_values else None,
        "peak_power_w": round(max(power_values), 3) if power_values else None,
        "total_energy_j": round(total_energy_j, 3) if total_energy_j is not None else None,
        "samples_collected": len(timeseries),
        "inference_success_count": worker_success,
        "inference_error_count": len(worker_errors),
        "errors": summarize_errors(worker_errors),
    }


def run_sustained_load(inference_fn: Callable, duration_min: int = 10) -> dict:
    """
    Avalia degradação de desempenho sob carga sustentada.

    Roda inference_fn continuamente por duration_min minutos,
    registrando throughput a cada minuto. Detecta throttling térmico
    comparando throughput do primeiro e último minuto.

    Args:
        inference_fn: Função sem argumentos que executa uma inferência.
        duration_min: Duração total em minutos.

    Retorna:
        dict com:
            throughput_per_minute: lista com ops/sec de cada minuto
            degradation_pct: queda percentual do primeiro para o último minuto
            throttling_detected: True se degradação > 15%
            temp_max_c: temperatura máxima registrada
    """
    throughput_per_minute = []
    temp_max_c = None
    stop_event = threading.Event()
    temps = []
    total_ops = 0
    total_errors = 0

    def temp_monitor():
        """Coleta temperatura em background."""
        while not stop_event.is_set():
            m = _collect_system_metrics()
            if m.get("temp_c"):
                temps.append(m["temp_c"])
            time.sleep(2)

    t_temp = threading.Thread(target=temp_monitor, daemon=True)
    t_temp.start()

    # Roda por duration_min minutos, medindo throughput por minuto
    for minute in range(duration_min):
        ops = 0
        minute_start = time.perf_counter()
        minute_end = minute_start + 60.0

        while time.perf_counter() < minute_end:
            try:
                inference_fn()
                ops += 1
            except Exception:
                total_errors += 1

        actual_time = time.perf_counter() - minute_start
        tps = ops / actual_time if actual_time > 0 else 0
        throughput_per_minute.append(round(tps, 3))
        total_ops += ops

    stop_event.set()
    t_temp.join(timeout=5)

    if temps:
        temp_max_c = round(max(temps), 1)

    # Calcula degradação: compara primeiro vs último minuto
    degradation_pct = 0.0
    throttling_detected = False
    if len(throughput_per_minute) >= 2 and throughput_per_minute[0] > 0:
        first = throughput_per_minute[0]
        last = throughput_per_minute[-1]
        degradation_pct = round((first - last) / first * 100, 2)
        throttling_detected = degradation_pct > 15.0

    return {
        "throughput_per_minute": throughput_per_minute,
        "degradation_pct": degradation_pct,
        "throttling_detected": throttling_detected,
        "temp_max_c": temp_max_c,
        "quality": result_quality(total_ops, total_errors),
    }


def run_memory_bandwidth(patterns: List[str] = None,
                         block_sizes_kb: List[int] = None) -> dict:
    """
    Mede largura de banda da memória RAM com padrões sequencial e aleatório.

    Aloca arrays numpy e realiza operações de leitura/escrita em diferentes
    padrões de acesso para identificar gargalos de memória.

    Args:
        patterns: Lista de padrões a testar: "sequential", "random"
        block_sizes_kb: Lista de tamanhos de bloco em KB

    Retorna:
        dict com resultados por padrão:
            sequential: {"bandwidth_gbs": float, "latency_ns": float}
            random: {"bandwidth_gbs": float, "latency_ns": float}
    """
    if patterns is None:
        patterns = ["sequential", "random"]
    if block_sizes_kb is None:
        block_sizes_kb = [4, 64, 1024, 16384]

    if not _NUMPY:
        return {"error": "numpy não disponível"}

    results = {}

    for pattern in patterns:
        bandwidths = []
        latencies = []

        for block_kb in block_sizes_kb:
            # Tamanho em elementos float64 (8 bytes cada)
            n_elements = (block_kb * 1024) // 8
            if n_elements < 1:
                continue

            try:
                # Aloca array
                arr = np.random.random(n_elements).astype(np.float64)
                iterations = max(1, min(100, 1_000_000 // n_elements))

                t0 = time.perf_counter()

                if pattern == "sequential":
                    # Acesso sequencial: soma todos os elementos
                    for _ in range(iterations):
                        _ = arr.sum()
                        arr[:] = arr + 1.0

                elif pattern == "random":
                    # Acesso aleatório: índices embaralhados
                    indices = np.random.permutation(n_elements)
                    chunk = min(n_elements, 10000)
                    for _ in range(iterations):
                        _ = arr[indices[:chunk]].sum()
                        arr[indices[:chunk]] += 1.0

                elapsed = time.perf_counter() - t0

                # Bytes transferidos: leitura + escrita = 2x tamanho do array * iterações
                bytes_transferred = 2 * n_elements * 8 * iterations
                bandwidth_gbs = bytes_transferred / elapsed / (1024**3)
                latency_ns = (elapsed / (iterations * n_elements)) * 1e9

                bandwidths.append(bandwidth_gbs)
                latencies.append(latency_ns)

                del arr

            except Exception:
                pass

        if bandwidths:
            results[pattern] = {
                "bandwidth_gbs": round(max(bandwidths), 3),   # Pico de bandwidth
                "latency_ns": round(min(latencies), 3),        # Menor latência
                "by_block_size": {
                    f"{bs}kb": {"bandwidth_gbs": round(bw, 3), "latency_ns": round(lat, 3)}
                    for bs, bw, lat in zip(block_sizes_kb, bandwidths, latencies)
                }
            }

    return results


def run_full_benchmark(inference_fn: Callable, profile_config: dict,
                       capabilities: dict, profile_name: Optional[str] = None,
                       front: Optional[str] = None, model: Optional[str] = None,
                       mode: Optional[str] = None, dataset: Optional[str] = None,
                       hardware_config: Optional[dict] = None) -> dict:
    """
    Orquestra todos os benchmarks habilitados no perfil de configuração.

    Lê os parâmetros de cada teste do profile_config e executa os testes
    habilitados em sequência. Salva resultados com timestamp e info do sistema.

    Args:
        inference_fn: Função sem argumentos que executa uma inferência.
        profile_config: Dict com configuração do perfil (ex: conteúdo de benchmark.yaml).
        capabilities: Dict retornado por CapabilityDetector.detect().

    Retorna:
        JSON completo com system_info, model_info e resultados de cada teste.
    """
    import datetime

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "system_info": capabilities,
        "methodology": base_methodology(profile_name, profile_config),
        "metadata": benchmark_metadata(front, model, mode, dataset),
        "benchmark_results": {},
    }

    # --- Throughput ---
    tp_cfg = profile_config.get("throughput", {})
    try:
        results["benchmark_results"]["throughput"] = run_throughput(
            inference_fn,
            duration_sec=tp_cfg.get("duration_sec", 60),
            warmup=tp_cfg.get("warmup", 5),
        )
    except Exception as e:
        results["benchmark_results"]["throughput"] = {"error": str(e)}

    # --- Latência ---
    lat_cfg = profile_config.get("latency", {})
    try:
        results["benchmark_results"]["latency"] = run_latency(
            inference_fn,
            samples=lat_cfg.get("samples", 50),
            percentiles=lat_cfg.get("percentiles", [50, 95, 99]),
        )
    except Exception as e:
        results["benchmark_results"]["latency"] = {"error": str(e)}

    # --- Uso de recursos ---
    res_cfg = profile_config.get("resource_usage", {})
    try:
        results["benchmark_results"]["resource_usage"] = run_resource_usage(
            inference_fn,
            duration_sec=res_cfg.get("duration_sec", 120),
            interval_sec=res_cfg.get("interval_sec", 1),
        )
    except Exception as e:
        results["benchmark_results"]["resource_usage"] = {"error": str(e)}

    # --- Carga sustentada ---
    sl_cfg = profile_config.get("sustained_load", {})
    try:
        results["benchmark_results"]["sustained_load"] = run_sustained_load(
            inference_fn,
            duration_min=sl_cfg.get("duration_min", 10),
        )
    except Exception as e:
        results["benchmark_results"]["sustained_load"] = {"error": str(e)}

    # --- Largura de banda de memória (opcional) ---
    mb_cfg = profile_config.get("memory_bandwidth", {})
    if mb_cfg.get("enabled", False):
        try:
            results["benchmark_results"]["memory_bandwidth"] = run_memory_bandwidth(
                patterns=mb_cfg.get("patterns", ["sequential", "random"]),
                block_sizes_kb=mb_cfg.get("block_sizes_kb", [4, 64, 1024, 16384]),
            )
        except Exception as e:
            results["benchmark_results"]["memory_bandwidth"] = {"error": str(e)}

    if front and not results.get("front"):
        results["front"] = front
    if model is not None and not results.get("model"):
        results["model"] = model
    if mode is not None and not results.get("mode"):
        results["mode"] = mode
    if profile_name is not None and not results.get("profile"):
        results["profile"] = profile_name

    try:
        cfg = hardware_config if hardware_config is not None else load_hardware_config()
        results["scorecard"] = compute_scorecard(results, cfg)
    except Exception as e:
        results["scorecard"] = {"error": str(e)}

    return results
