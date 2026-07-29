"""Execução dos perfis de benchmark de inferência."""

import statistics
import threading
import time
import uuid
from typing import Callable, List, Optional

from shared.capabilities import merge_host_capabilities
from shared.config_validation import validate_profile_config
from shared.methodology import (
    NumericResultAccumulator,
    base_methodology,
    benchmark_metadata,
    result_quality,
    summarize_errors,
)
from shared.metrics_collector import MetricsCollector
from shared.scorecard import compute_scorecard, load_hardware_config

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


MAX_CONSECUTIVE_FAILURES = 10


def run_throughput(inference_fn: Callable, duration_sec: int = 60, warmup: int = 5) -> dict:
    """Mede operações por segundo após as chamadas de aquecimento."""
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
    consecutive_errors = 0
    aborted_due_to_errors = False
    errors = []
    result_metrics = NumericResultAccumulator()
    start_time = time.perf_counter()
    end_time = start_time + duration_sec

    while time.perf_counter() < end_time:
        try:
            result = inference_fn()
            ops_total += 1
            consecutive_errors = 0
            result_metrics.update(result)
        except Exception as e:
            error_count += 1
            consecutive_errors += 1
            if len(errors) < 20:
                errors.append(f"{type(e).__name__}: {e}")
            if consecutive_errors >= MAX_CONSECUTIVE_FAILURES:
                aborted_due_to_errors = True
                break

    actual_duration = time.perf_counter() - start_time
    ops_per_sec = ops_total / actual_duration if actual_duration > 0 else 0

    quality = result_quality(ops_total, error_count)
    if aborted_due_to_errors:
        quality["valid"] = False
        quality["warnings"].append("phase stopped after consecutive inference failures")
    return {
        "ops_per_sec": round(ops_per_sec, 3),
        "ops_total": ops_total,
        "duration_sec": round(actual_duration, 2),
        "warmup": {
            "attempts": warmup,
            "error_count": len(warmup_errors),
            "errors": summarize_errors(warmup_errors, total_count=len(warmup_errors)),
        },
        "error_count": error_count,
        "error_rate": quality["error_rate"],
        "quality": quality,
        "aborted_due_to_errors": aborted_due_to_errors,
        "errors": summarize_errors(errors, total_count=error_count),
        "result_metrics": result_metrics.summary(),
    }


def run_latency(inference_fn: Callable, samples: int = 50,
                percentiles: List[int] = None) -> dict:
    """Mede a distribuição de latência de inferências individuais."""
    if percentiles is None:
        percentiles = [50, 95, 99]

    latencies_ms = []
    error_count = 0
    consecutive_errors = 0
    aborted_due_to_errors = False
    errors = []
    result_metrics = NumericResultAccumulator()

    for i in range(samples):
        try:
            t0 = time.perf_counter()
            result = inference_fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)
            consecutive_errors = 0
            result_metrics.update(result)
        except Exception as e:
            error_count += 1
            consecutive_errors += 1
            if len(errors) < 20:
                errors.append(f"{type(e).__name__}: {e}")
            if consecutive_errors >= MAX_CONSECUTIVE_FAILURES:
                aborted_due_to_errors = True
                break

    if not latencies_ms:
        quality = result_quality(0, error_count)
        quality["valid"] = False
        if aborted_due_to_errors:
            quality["warnings"].append(
                "phase stopped after consecutive inference failures")
        return {
            "error": "Nenhuma amostra coletada",
            "samples_collected": 0,
            "samples_requested": samples,
            "error_count": error_count,
            "error_rate": quality["error_rate"],
            "quality": quality,
            "aborted_due_to_errors": aborted_due_to_errors,
            "errors": summarize_errors(errors, total_count=error_count),
            "result_metrics": result_metrics.summary(),
        }

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
    result["min_ms"] = round(min(latencies_ms), 2)
    result["max_ms"] = round(max(latencies_ms), 2)
    result["samples_collected"] = len(latencies_ms)
    result["samples_requested"] = samples
    result["error_count"] = error_count
    result["quality"] = result_quality(len(latencies_ms), error_count)
    result["error_rate"] = result["quality"]["error_rate"]
    result["aborted_due_to_errors"] = aborted_due_to_errors
    if aborted_due_to_errors:
        result["quality"]["valid"] = False
        result["quality"]["warnings"].append(
            "phase stopped after consecutive inference failures")
    result["errors"] = summarize_errors(errors, total_count=error_count)
    result["result_metrics"] = result_metrics.summary()

    return result


def run_resource_usage(inference_fn: Callable, duration_sec: int = 120,
                       interval_sec: float = 1.0) -> dict:
    """Monitora recursos enquanto executa inferências na mesma janela."""
    stop_event = threading.Event()
    worker_errors = []
    worker_success = 0
    worker_error_count = 0
    worker_done = threading.Event()
    worker_lock = threading.Lock()
    aborted_due_to_errors = False
    result_metrics = NumericResultAccumulator()

    def inference_worker():
        nonlocal worker_success, worker_error_count, aborted_due_to_errors
        consecutive_errors = 0
        while not stop_event.is_set():
            try:
                result = inference_fn()
                with worker_lock:
                    worker_success += 1
                    consecutive_errors = 0
                    result_metrics.update(result)
            except Exception as e:
                with worker_lock:
                    worker_error_count += 1
                    consecutive_errors += 1
                    if len(worker_errors) < 20:
                        worker_errors.append(f"{type(e).__name__}: {e}")
                    if consecutive_errors >= MAX_CONSECUTIVE_FAILURES:
                        aborted_due_to_errors = True
                        break
        worker_done.set()

    collector = MetricsCollector(interval_sec=interval_sec)
    t_infer = threading.Thread(target=inference_worker, daemon=True)
    started = time.perf_counter()
    collector.start()
    t_infer.start()
    worker_done.wait(max(0.0, duration_sec))
    stop_event.set()
    t_infer.join(timeout=max(30.0, min(float(duration_sec), 120.0)))
    worker_stopped = not t_infer.is_alive()
    collector.stop()
    actual_duration = time.perf_counter() - started

    with worker_lock:
        success_count = worker_success
        error_count = worker_error_count
        metric_summary = result_metrics.summary()
        error_summary = summarize_errors(worker_errors, total_count=error_count)

    summary = collector.get_summary()
    measured_duration = summary.get("measurement_duration_sec") or actual_duration
    summary.update({
        "requested_duration_sec": duration_sec,
        "measurement_duration_sec": round(measured_duration, 3),
        "inference_success_count": success_count,
        "inference_error_count": error_count,
        "inference_ops_per_sec": round(success_count / measured_duration, 4)
        if measured_duration > 0 else None,
        "energy_per_inference_j": round(summary["total_energy_j"] / success_count, 6)
        if summary.get("total_energy_j") is not None and success_count > 0 else None,
        "result_metrics": metric_summary,
        "worker_stopped": worker_stopped,
        "aborted_due_to_errors": aborted_due_to_errors,
        "quality": result_quality(success_count, error_count),
        "errors": error_summary,
    })
    if aborted_due_to_errors:
        summary["quality"]["valid"] = False
        summary["quality"]["warnings"].append(
            "phase stopped after consecutive inference failures")
    if not worker_stopped:
        summary["quality"]["valid"] = False
        summary["quality"]["warnings"].append("inference worker did not stop within timeout")
    return summary


def run_sustained_load(inference_fn: Callable, duration_min: int = 10) -> dict:
    """Mede a variação de throughput durante carga sustentada."""
    throughput_per_minute = []
    temp_max_c = None
    total_ops = 0
    total_errors = 0
    consecutive_errors = 0
    errors = []
    aborted_due_to_errors = False
    collector = MetricsCollector(interval_sec=2.0)
    collector.start()

    # Roda por duration_min minutos, medindo throughput por minuto
    for minute in range(duration_min):
        ops = 0
        minute_start = time.perf_counter()
        minute_end = minute_start + 60.0

        while time.perf_counter() < minute_end:
            try:
                inference_fn()
                ops += 1
                consecutive_errors = 0
            except Exception as e:
                total_errors += 1
                consecutive_errors += 1
                if len(errors) < 20:
                    errors.append(f"{type(e).__name__}: {e}")
                if consecutive_errors >= MAX_CONSECUTIVE_FAILURES:
                    aborted_due_to_errors = True
                    break

        actual_time = time.perf_counter() - minute_start
        tps = ops / actual_time if actual_time > 0 else 0
        throughput_per_minute.append(round(tps, 3))
        total_ops += ops
        if aborted_due_to_errors:
            break

    collector.stop()
    resource_summary = collector.get_summary()
    temp_max_c = resource_summary.get("peak_temp_c")

    # Calcula degradação: compara primeiro vs último minuto
    degradation_pct = 0.0
    throttling_detected = False
    if len(throughput_per_minute) >= 2 and throughput_per_minute[0] > 0:
        first = throughput_per_minute[0]
        last = throughput_per_minute[-1]
        degradation_pct = round((first - last) / first * 100, 2)
        throttling_detected = degradation_pct > 15.0

    quality = result_quality(total_ops, total_errors)
    if aborted_due_to_errors:
        quality["valid"] = False
        quality["warnings"].append("phase stopped after consecutive inference failures")

    return {
        "throughput_per_minute": throughput_per_minute,
        "degradation_pct": degradation_pct,
        "throttling_detected": throttling_detected,
        "temp_max_c": temp_max_c,
        "aborted_due_to_errors": aborted_due_to_errors,
        "quality": quality,
        "errors": summarize_errors(errors, total_count=total_errors),
    }


def run_memory_bandwidth(patterns: List[str] = None,
                         block_sizes_kb: List[int] = None) -> dict:
    """Mede acesso sequencial e aleatório à memória por tamanho de bloco."""
    if patterns is None:
        patterns = ["sequential", "random"]
    if block_sizes_kb is None:
        block_sizes_kb = [4, 64, 1024, 16384]

    if not _NUMPY:
        return {"error": "numpy não disponível"}

    results = {}

    for pattern in patterns:
        measurements = []

        for block_kb in block_sizes_kb:
            # Tamanho em elementos float64 (8 bytes cada)
            n_elements = (block_kb * 1024) // 8
            if n_elements < 1:
                continue

            try:
                arr = np.random.random(n_elements).astype(np.float64)
                iterations = max(1, min(100, 1_000_000 // n_elements))

                t0 = time.perf_counter()

                if pattern == "sequential":
                    for _ in range(iterations):
                        np.add(arr, 1.0, out=arr)
                    elements_touched = n_elements

                elif pattern == "random":
                    indices = np.random.permutation(n_elements)
                    chunk = min(n_elements, 10000)
                    for _ in range(iterations):
                        arr[indices[:chunk]] += 1.0
                    elements_touched = chunk
                else:
                    continue

                elapsed = time.perf_counter() - t0

                bytes_transferred = 2 * elements_touched * 8 * iterations
                bandwidth_gbs = bytes_transferred / elapsed / (1024**3)
                latency_ns = (elapsed / (iterations * elements_touched)) * 1e9
                measurements.append((block_kb, bandwidth_gbs, latency_ns))

                del arr

            except Exception:
                pass

        if measurements:
            bandwidths = [item[1] for item in measurements]
            latencies = [item[2] for item in measurements]
            results[pattern] = {
                "bandwidth_gbs": round(max(bandwidths), 3),
                "latency_ns": round(min(latencies), 3),
                "by_block_size": {
                    f"{bs}kb": {"bandwidth_gbs": round(bw, 3), "latency_ns": round(lat, 3)}
                    for bs, bw, lat in measurements
                }
            }

    return results


def run_full_benchmark(inference_fn: Callable, profile_config: dict,
                       capabilities: dict, profile_name: Optional[str] = None,
                       front: Optional[str] = None, model: Optional[str] = None,
                       mode: Optional[str] = None, dataset: Optional[str] = None,
                       hardware_config: Optional[dict] = None,
                       execution: Optional[dict] = None) -> dict:
    """Executa, em sequência, as fases habilitadas no perfil."""
    import datetime

    validate_profile_config(profile_config, profile_name or "profile")
    run_id = str(uuid.uuid4())
    capabilities = merge_host_capabilities(capabilities)
    cfg = hardware_config if hardware_config is not None else load_hardware_config()
    cfg = cfg if isinstance(cfg, dict) else {}
    metadata = benchmark_metadata(
        front,
        model,
        mode,
        dataset,
        execution,
        profile=profile_name,
        run_id=run_id,
    )
    hardware_profile = cfg.get("profile")
    if isinstance(hardware_profile, dict):
        metadata["hardware_profile"] = {
            key: hardware_profile.get(key)
            for key in ("id", "description") if hardware_profile.get(key) is not None
        }

    results = {
        "schema_version": "benchmark-result-2026.07-v2",
        "run_id": run_id,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "system_info": capabilities,
        "methodology": base_methodology(profile_name, profile_config),
        "metadata": metadata,
        "benchmark_results": {},
    }

    # --- Throughput ---
    tp_cfg = profile_config.get("throughput")
    if tp_cfg and tp_cfg.get("enabled", True):
        try:
            results["benchmark_results"]["throughput"] = run_throughput(
                inference_fn,
                duration_sec=tp_cfg.get("duration_sec", 60),
                warmup=tp_cfg.get("warmup", 5),
            )
        except Exception as e:
            results["benchmark_results"]["throughput"] = {"error": str(e)}

    # --- Latência ---
    lat_cfg = profile_config.get("latency")
    if lat_cfg and lat_cfg.get("enabled", True):
        try:
            results["benchmark_results"]["latency"] = run_latency(
                inference_fn,
                samples=lat_cfg.get("samples", 50),
                percentiles=lat_cfg.get("percentiles", [50, 95, 99]),
            )
        except Exception as e:
            results["benchmark_results"]["latency"] = {"error": str(e)}

    # --- Uso de recursos ---
    res_cfg = profile_config.get("resource_usage")
    if res_cfg and res_cfg.get("enabled", True):
        try:
            results["benchmark_results"]["resource_usage"] = run_resource_usage(
                inference_fn,
                duration_sec=res_cfg.get("duration_sec", 120),
                interval_sec=res_cfg.get("interval_sec", 1),
            )
        except Exception as e:
            results["benchmark_results"]["resource_usage"] = {"error": str(e)}

    # --- Carga sustentada ---
    sl_cfg = profile_config.get("sustained_load")
    if sl_cfg and sl_cfg.get("enabled", True) and sl_cfg.get("duration_min", 10) > 0:
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
        results["scorecard"] = compute_scorecard(results, cfg)
    except Exception as e:
        results["scorecard"] = {"error": str(e)}

    return results
