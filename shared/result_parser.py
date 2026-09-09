"""Normalização dos arquivos de resultado produzidos pela plataforma."""

import json
import math
import os
from typing import Any, Iterable, Optional

from shared.methodology import get_nested

BENCHMARK_FRONTS = {"vision", "slm", "audio"}


def is_benchmark_result(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    front = data.get("front") or get_nested(data, "metadata.front")
    benchmark = data.get("benchmark_results")
    if front not in BENCHMARK_FRONTS or not isinstance(benchmark, dict):
        return False
    return any(isinstance(benchmark.get(section), dict) for section in (
        "throughput", "latency", "resource_usage", "sustained_load",
        "sustained", "memory_bandwidth",
    ))


def finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def first_number(*values: Any) -> Optional[float]:
    for value in values:
        number = finite_number(value)
        if number is not None:
            return number
    return None


def normalize_model(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [f"{key}: {item}" for key, item in value.items()
                 if isinstance(item, (str, int, float)) and not isinstance(item, bool)]
        return " / ".join(parts) if parts else "?"
    return str(value)


def text_value(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _series_values(samples: Iterable[dict], *keys: str) -> list:
    values = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        number = first_number(*(sample.get(key) for key in keys))
        if number is not None:
            values.append(number)
    return values


def _series_average(samples: Iterable[dict], *keys: str) -> Optional[float]:
    samples = list(samples)
    fallback = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        value = first_number(*(sample.get(key) for key in keys))
        if value is None:
            continue
        fallback.append(value)
    area = 0.0
    duration = 0.0
    for previous, current in zip(samples, samples[1:]):
        t0 = finite_number(previous.get("timestamp"))
        t1 = finite_number(current.get("timestamp"))
        v0 = first_number(*(previous.get(key) for key in keys))
        v1 = first_number(*(current.get(key) for key in keys))
        if None in (t0, t1, v0, v1):
            continue
        delta = t1 - t0
        if delta <= 0:
            continue
        area += (v0 + v1) * 0.5 * delta
        duration += delta
    if duration > 0:
        return area / duration
    return sum(fallback) / len(fallback) if fallback else None


def _series_interval_average(samples: Iterable[dict], *keys: str) -> Optional[float]:
    samples = list(samples)
    fallback = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        value = first_number(*(sample.get(key) for key in keys))
        if value is not None:
            fallback.append(value)
    total = 0.0
    duration = 0.0
    for previous, current in zip(samples, samples[1:]):
        t0 = finite_number(previous.get("timestamp"))
        t1 = finite_number(current.get("timestamp"))
        value = first_number(*(current.get(key) for key in keys))
        if None in (t0, t1, value):
            continue
        delta = t1 - t0
        if delta <= 0:
            continue
        total += value * delta
        duration += delta
    if duration > 0:
        return total / duration
    return sum(fallback) / len(fallback) if fallback else None


def _maximum(samples: Iterable[dict], *keys: str) -> Optional[float]:
    values = _series_values(samples, *keys)
    return max(values) if values else None


def normalize_resource_usage(resource: Any) -> dict:
    resource = resource if isinstance(resource, dict) else {}
    samples = [sample for sample in resource.get("timeseries", [])
               if isinstance(sample, dict)]
    schema_version = resource.get("schema_version")
    scopes = resource.get("scopes") if isinstance(resource.get("scopes"), dict) else {}
    current_schema = str(schema_version or "").startswith("resources-2026.07")

    if current_schema:
        peak_ram = first_number(
            resource.get("peak_ram_mb"),
            resource.get("peak_container_ram_mb"),
            resource.get("peak_process_ram_mb"),
            _maximum(samples, "ram_mb", "container_ram_mb", "process_ram_mb"),
        )
        ram_scope = (scopes.get("ram_mb") or "workload") \
            if peak_ram is not None else "unavailable"
    else:
        process_ram = first_number(
            resource.get("peak_process_ram_mb"),
            _maximum(samples, "process_ram_mb"),
        )
        container_ram = first_number(
            resource.get("peak_container_ram_mb"),
            _maximum(samples, "container_ram_mb"),
        )
        legacy_ram = first_number(
            resource.get("peak_ram_mb"),
            _maximum(samples, "ram_mb"),
        )
        peak_ram = first_number(container_ram, process_ram, legacy_ram)
        ram_scope = (
            "container_cgroup_working_set" if container_ram is not None
            else "process_tree_rss" if process_ram is not None
            else "legacy_unspecified" if legacy_ram is not None
            else "unavailable"
        )

    if current_schema:
        peak_vram = first_number(
            resource.get("peak_vram_mb"),
            resource.get("peak_gpu_mem_mb"),
            _maximum(samples, "gpu_mem_mb"),
        )
        vram_scope = (scopes.get("gpu_mem_mb") or "visible_devices") \
            if peak_vram is not None else "unavailable"
    else:
        process_vram = first_number(
            resource.get("peak_process_vram_mb"),
            _maximum(samples, "gpu_process_mem_mb"),
        )
        legacy_vram = first_number(
            resource.get("peak_gpu_mem_mb"),
            resource.get("peak_vram_mb"),
            _maximum(samples, "gpu_mem_mb"),
        )
        peak_vram = first_number(process_vram, legacy_vram)
        vram_scope = (
            "workload_processes" if process_vram is not None
            else "legacy_device" if legacy_vram is not None
            else "unavailable"
        )

    avg_cpu = first_number(
        resource.get("avg_cpu_pct") if current_schema else None,
        resource.get("avg_process_cpu_pct"),
        _series_interval_average(samples, "process_cpu_pct"),
        resource.get("avg_cpu_pct"),
        _series_interval_average(samples, "cpu_pct"),
    )
    if avg_cpu is None:
        cpu_scope = "unavailable"
    else:
        cpu_scope = scopes.get("cpu_pct") or (
            "process_tree" if first_number(resource.get("avg_process_cpu_pct"),
                                           _series_interval_average(
                                               samples, "process_cpu_pct")) is not None
            else "legacy_unspecified"
        )
    peak_gpu = first_number(
        resource.get("peak_gpu_util"),
        _maximum(samples, "gpu_pct"),
    )
    avg_gpu = first_number(
        resource.get("avg_gpu_pct"),
        _series_average(samples, "gpu_pct"),
    )
    peak_accelerator = first_number(
        resource.get("peak_accelerator_util"),
        _maximum(samples, "accelerator_pct"),
    )
    avg_accelerator = first_number(
        resource.get("avg_accelerator_pct"),
        _series_average(samples, "accelerator_pct"),
    )
    avg_power = first_number(
        resource.get("avg_power_w"),
        resource.get("avg_gpu_power_w"),
        _series_average(samples, "power_w", "gpu_power_w"),
    )
    total_energy = first_number(resource.get("total_energy_j"))
    power_sources = resource.get("power_sources")
    power_sources = power_sources if isinstance(power_sources, list) else []
    energy_source = resource.get("energy_source") or (
        "+".join(str(source) for source in power_sources) if power_sources else "measured"
    )
    if total_energy is None:
        energy_wh = first_number(resource.get("gpu_energy_wh"))
        if energy_wh is not None:
            total_energy = energy_wh * 3600.0
            energy_source = "legacy_gpu_energy_wh"
        else:
            energy_source = "unavailable"

    duration = first_number(
        resource.get("measurement_duration_sec"),
        resource.get("duration_sec"),
    )
    if duration is None and len(samples) >= 2:
        first_timestamp = finite_number(samples[0].get("timestamp"))
        last_timestamp = finite_number(samples[-1].get("timestamp"))
        if first_timestamp is not None and last_timestamp is not None:
            duration = max(0.0, last_timestamp - first_timestamp)

    energy_coverage = first_number(resource.get("energy_coverage_ratio"))
    if energy_coverage is not None:
        energy_coverage = min(1.0, max(0.0, energy_coverage))

    return {
        "schema_version": schema_version,
        "timeseries": samples,
        "peak_ram_mb": peak_ram,
        "baseline_ram_mb": first_number(resource.get("baseline_ram_mb")),
        "peak_ram_delta_mb": first_number(resource.get("peak_ram_delta_mb")),
        "ram_scope": ram_scope,
        "avg_cpu_pct": avg_cpu,
        "cpu_scope": cpu_scope,
        "peak_gpu_util": peak_gpu,
        "avg_gpu_pct": avg_gpu,
        "gpu_scope": (scopes.get("gpu_pct") or "visible_devices")
        if peak_gpu is not None or avg_gpu is not None else "unavailable",
        "peak_process_gpu_util": first_number(
            resource.get("peak_process_gpu_util"),
            _maximum(samples, "gpu_process_pct"),
        ),
        "peak_device_gpu_util": first_number(
            resource.get("peak_device_gpu_util"),
            _maximum(samples, "gpu_device_pct", "gpu_pct"),
        ),
        "peak_accelerator_util": peak_accelerator,
        "avg_accelerator_pct": avg_accelerator,
        "accelerator_scope": (scopes.get("accelerator_pct") or "visible_devices")
        if peak_accelerator is not None or avg_accelerator is not None else "unavailable",
        "peak_accelerator_memory_mb": first_number(
            resource.get("peak_accelerator_memory_mb"),
            _maximum(samples, "accelerator_memory_mb"),
        ),
        "accelerator_memory_scope": (
            scopes.get("accelerator_memory_mb") or "visible_devices"
        ) if first_number(
            resource.get("peak_accelerator_memory_mb"),
            _maximum(samples, "accelerator_memory_mb"),
        ) is not None else "unavailable",
        "peak_vram_mb": peak_vram,
        "baseline_vram_mb": first_number(resource.get("baseline_vram_mb")),
        "peak_vram_delta_mb": first_number(resource.get("peak_vram_delta_mb")),
        "vram_scope": vram_scope,
        "peak_temp_c": first_number(
            resource.get("peak_temp_c"),
            _maximum(samples, "temp_c", "gpu_temp_c", "cpu_temp_c"),
        ),
        "thermal_throttling_observed": resource.get("thermal_throttling_observed") is True
        or any(sample.get("thermal_throttling") is True for sample in samples),
        "avg_power_w": avg_power,
        "power_scope": (scopes.get("power_w") or "legacy_device")
        if avg_power is not None or total_energy is not None else "unavailable",
        "total_energy_j": total_energy,
        "energy_source": energy_source,
        "energy_coverage_ratio": energy_coverage,
        "energy_observed_duration_sec": first_number(
            resource.get("energy_observed_duration_sec")),
        "measurement_duration_sec": duration,
        "container_cpu_time_sec": first_number(resource.get("container_cpu_time_sec")),
        "container_cpu_limit_cores": first_number(
            resource.get("container_cpu_limit_cores")),
        "container_memory_limit_mb": first_number(
            resource.get("container_memory_limit_mb")),
        "container_pids_limit": first_number(resource.get("container_pids_limit")),
        "inference_success_count": first_number(resource.get("inference_success_count")),
        "inference_error_count": first_number(resource.get("inference_error_count")),
        "inference_ops_per_sec": first_number(resource.get("inference_ops_per_sec")),
        "result_metrics": resource.get("result_metrics")
        if isinstance(resource.get("result_metrics"), dict) else {},
        "quality": resource.get("quality") if isinstance(resource.get("quality"), dict) else {},
    }


def _memory_bandwidth_value(section: dict) -> Optional[float]:
    direct = first_number(section.get("effective_bandwidth_gbs"))
    if direct is not None:
        return direct
    values = []
    for item in section.values():
        if isinstance(item, dict):
            value = finite_number(item.get("bandwidth_gbs"))
            if value is not None:
                values.append(value)
    return max(values) if values else None


def normalize_result(data: Any, filename: Optional[str] = None) -> dict:
    data = data if isinstance(data, dict) else {}
    benchmark = data.get("benchmark_results")
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    throughput = benchmark.get("throughput")
    throughput = throughput if isinstance(throughput, dict) else {}
    latency = benchmark.get("latency")
    latency = latency if isinstance(latency, dict) else {}
    resources = normalize_resource_usage(benchmark.get("resource_usage"))
    sustained = benchmark.get("sustained_load", benchmark.get("sustained", {}))
    sustained = sustained if isinstance(sustained, dict) else {}
    memory_bandwidth = benchmark.get("memory_bandwidth")
    memory_bandwidth = memory_bandwidth if isinstance(memory_bandwidth, dict) else {}
    scorecard = data.get("scorecard") if isinstance(data.get("scorecard"), dict) else {}

    per_minute = [finite_number(value) for value in sustained.get("throughput_per_minute", [])]
    per_minute = [value for value in per_minute if value is not None]
    sustained_average = first_number(sustained.get("avg_ops_per_sec"))
    if sustained_average is None and per_minute:
        sustained_average = sum(per_minute) / len(per_minute)

    front = data.get("front") or get_nested(data, "metadata.front")
    model = data.get("model", get_nested(data, "metadata.model"))
    profile = data.get("profile") or get_nested(data, "methodology.profile")
    mode = data.get("mode") or get_nested(data, "metadata.mode")
    dataset = data.get("dataset") or get_nested(data, "metadata.dataset")
    execution = get_nested(data, "metadata.execution")
    if not isinstance(execution, dict):
        execution = get_nested(data, "system_info.runtime", {})
    execution = execution if isinstance(execution, dict) else {}
    cpu_model = text_value(get_nested(data, "system_info.cpu.model"))
    gpu_model = text_value(get_nested(data, "system_info.gpu.model"))
    architecture = text_value(
        get_nested(data, "system_info.cpu.arch") or execution.get("architecture"))
    hardware_profile = get_nested(data, "metadata.hardware_profile")
    hardware_profile = hardware_profile if isinstance(hardware_profile, dict) else {}
    primary_metric = scorecard.get("primary_metric")
    primary_metric = primary_metric if isinstance(primary_metric, dict) else None

    return {
        "schema_version": data.get("schema_version"),
        "run_id": data.get("run_id") or get_nested(data, "metadata.run_id"),
        "file": os.path.basename(filename) if filename else None,
        "front": front,
        "model": normalize_model(model),
        "profile": profile,
        "mode": mode,
        "dataset": dataset,
        "execution": execution,
        "cpu_model": cpu_model,
        "gpu_model": gpu_model,
        "architecture": architecture,
        "hardware_id": text_value(get_nested(data, "system_info.provenance.hardware_id")),
        "inventory_source": text_value(
            get_nested(data, "system_info.provenance.inventory_source")),
        "hardware_profile_id": text_value(hardware_profile.get("id")),
        "hardware_profile_description": text_value(hardware_profile.get("description")),
        "storage_type": text_value(get_nested(data, "system_info.storage.type")),
        "storage_device": text_value(get_nested(data, "system_info.storage.device")),
        "timestamp": data.get("timestamp"),
        "throughput_ops": first_number(throughput.get("ops_per_sec")),
        "total_iters": first_number(throughput.get("ops_total"), throughput.get("iterations")),
        "p50_ms": first_number(latency.get("p50_ms")),
        "p95_ms": first_number(latency.get("p95_ms")),
        "p99_ms": first_number(latency.get("p99_ms")),
        "std_ms": first_number(latency.get("std_ms")),
        "min_ms": first_number(latency.get("min_ms")),
        "max_ms": first_number(latency.get("max_ms")),
        "peak_ram_mb": resources["peak_ram_mb"],
        "baseline_ram_mb": resources["baseline_ram_mb"],
        "peak_ram_delta_mb": resources["peak_ram_delta_mb"],
        "ram_scope": resources["ram_scope"],
        "avg_cpu_pct": resources["avg_cpu_pct"],
        "cpu_scope": resources["cpu_scope"],
        "peak_gpu_util": resources["peak_gpu_util"],
        "avg_gpu_pct": resources["avg_gpu_pct"],
        "gpu_scope": resources["gpu_scope"],
        "peak_process_gpu_util": resources["peak_process_gpu_util"],
        "peak_device_gpu_util": resources["peak_device_gpu_util"],
        "peak_accelerator_util": resources["peak_accelerator_util"],
        "avg_accelerator_pct": resources["avg_accelerator_pct"],
        "accelerator_scope": resources["accelerator_scope"],
        "peak_accelerator_memory_mb": resources["peak_accelerator_memory_mb"],
        "accelerator_memory_scope": resources["accelerator_memory_scope"],
        "peak_vram_mb": resources["peak_vram_mb"],
        "baseline_vram_mb": resources["baseline_vram_mb"],
        "peak_vram_delta_mb": resources["peak_vram_delta_mb"],
        "vram_scope": resources["vram_scope"],
        "peak_temp_c": resources["peak_temp_c"],
        "thermal_throttling_observed": resources["thermal_throttling_observed"],
        "avg_power_w": resources["avg_power_w"],
        "power_scope": resources["power_scope"],
        "energy_j": resources["total_energy_j"],
        "energy_source": resources["energy_source"],
        "energy_coverage_ratio": resources["energy_coverage_ratio"],
        "resource_duration_sec": resources["measurement_duration_sec"],
        "resource_inference_count": resources["inference_success_count"],
        "container_cpu_time_sec": resources["container_cpu_time_sec"],
        "container_cpu_limit_cores": resources["container_cpu_limit_cores"],
        "container_memory_limit_mb": resources["container_memory_limit_mb"],
        "container_pids_limit": resources["container_pids_limit"],
        "sustained_avg_ops": sustained_average,
        "sustained_degradation": first_number(sustained.get("degradation_pct")),
        "mem_bw_gbs": _memory_bandwidth_value(memory_bandwidth),
        "methodology_version": get_nested(data, "methodology.version"),
        "comparability_key": get_nested(data, "metadata.comparability_key"),
        "throughput_error_rate": first_number(throughput.get("error_rate")),
        "latency_error_rate": first_number(latency.get("error_rate")),
        "performance_score": first_number(scorecard.get("performance_score")),
        "primary_metric": primary_metric,
        "energy_per_op_j": first_number(get_nested(scorecard, "efficiency.energy_per_op_j")),
        "energy_per_primary_unit_j": first_number(
            get_nested(scorecard, "efficiency.energy_per_primary_unit_j")),
        "primary_metric_per_watt": first_number(
            get_nested(scorecard, "efficiency.primary_metric_per_watt")),
        "cost_per_1k": first_number(get_nested(scorecard, "cost.cost_per_1k_primary_units")),
        "cost_currency": get_nested(scorecard, "cost.currency"),
        "scorecard_warnings": scorecard.get("warnings", [])
        if isinstance(scorecard.get("warnings"), list) else [],
    }


def load_result_file(path: str) -> tuple:
    with open(path) as handle:
        data = json.load(handle)
    return data, normalize_result(data, filename=path)
