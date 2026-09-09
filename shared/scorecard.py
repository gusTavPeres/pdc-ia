"""Indicadores comparáveis de desempenho, eficiência e custo."""

from typing import Any, Optional

from shared.methodology import get_nested
from shared.result_parser import finite_number, first_number, normalize_resource_usage

DEFAULT_REFERENCES = {
    "vision": {"metric": "ops_per_sec", "reference": 30.0, "unit": "frames/s"},
    "slm": {"metric": "tokens_per_sec", "reference": 20.0, "unit": "tokens/s"},
    "audio": {"metric": "ops_per_sec", "reference": 2.0, "unit": "ops/s"},
}

DEFAULT_METRIC_REFERENCES = {
    "vision": {"ops_per_sec": 30.0},
    "slm": {"tokens_per_sec": 20.0, "ops_per_sec": 2.0},
    "audio": {"ops_per_sec": 2.0, "audio_realtime_factor_inverse": 1.0},
}


def _num(value: Any) -> Optional[float]:
    return finite_number(value)


def load_hardware_config(path: str = "/app/config/hardware.yaml") -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _front(result: dict) -> str:
    return result.get("front") or get_nested(result, "metadata.front") or "unknown"


def _mode(result: dict) -> Optional[str]:
    return result.get("mode") or get_nested(result, "metadata.mode")


def _primary_metric(result: dict) -> dict:
    front = _front(result)
    bench = result.get("benchmark_results", {})
    tp = bench.get("throughput", {})
    result_metrics = tp.get("result_metrics", {})
    throughput_duration = _num(tp.get("duration_sec"))

    if front == "slm":
        total_tokens = _num(get_nested(result_metrics, "total_tokens.sum"))
        if total_tokens is not None and throughput_duration and throughput_duration > 0:
            return {
                "name": "tokens_per_sec",
                "value": round(total_tokens / throughput_duration, 4),
                "unit": "tokens/s",
                "higher_is_better": True,
            }
        tps = _num(get_nested(result_metrics, "tokens_per_sec.avg"))
        if tps is not None:
            return {"name": "tokens_per_sec", "value": tps, "unit": "tokens/s", "higher_is_better": True}

    if front == "audio":
        audio_duration = _num(get_nested(result_metrics, "duration_sec.sum"))
        if audio_duration is not None and throughput_duration and throughput_duration > 0:
            return {
                "name": "audio_realtime_factor_inverse",
                "value": round(audio_duration / throughput_duration, 4),
                "unit": "x real-time",
                "higher_is_better": True,
            }
        rtf = first_number(
            get_nested(result, "pipeline_results.avg_rtf"),
            get_nested(result_metrics, "pipeline_rtf.avg"),
            get_nested(result_metrics, "rtf.avg"),
        )
        if rtf is not None and rtf > 0:
            return {"name": "audio_realtime_factor_inverse", "value": round(1.0 / rtf, 4),
                    "unit": "x real-time", "higher_is_better": True}

    ops = _num(tp.get("ops_per_sec"))
    unit = "frames/s" if front == "vision" else "ops/s"
    return {"name": "ops_per_sec", "value": ops, "unit": unit,
            "higher_is_better": True}


def _peak_or_avg_power(result: dict) -> Optional[float]:
    res = result.get("benchmark_results", {}).get("resource_usage", {})
    return normalize_resource_usage(res).get("avg_power_w")


def _resource_primary_rate(result: dict, metric: dict, resource: dict) -> Optional[float]:
    duration = _num(resource.get("measurement_duration_sec"))
    operations = _num(resource.get("inference_success_count"))
    if duration is None or duration <= 0 or operations is None:
        return None

    name = metric.get("name")
    if name == "tokens_per_sec":
        total_tokens = _num(get_nested(resource, "result_metrics.total_tokens.sum"))
        if total_tokens is not None:
            return total_tokens / duration
        avg_tokens = _num(get_nested(resource, "result_metrics.total_tokens.avg"))
        return operations * avg_tokens / duration if avg_tokens is not None else None
    if name == "audio_realtime_factor_inverse":
        total_audio_duration = _num(get_nested(resource, "result_metrics.duration_sec.sum"))
        if total_audio_duration is not None:
            return total_audio_duration / duration
        avg_audio_duration = _num(get_nested(resource, "result_metrics.duration_sec.avg"))
        return operations * avg_audio_duration / duration if avg_audio_duration is not None else None
    return operations / duration


def _cost_model(config: dict) -> dict:
    cm = config.get("cost_model", {}) if isinstance(config, dict) else {}
    hardware_price = _num(cm.get("hardware_price"))
    amortization_hours = _num(cm.get("amortization_hours"))
    electricity_per_kwh = _num(cm.get("electricity_per_kwh"))
    return {
        "currency": cm.get("currency", "BRL"),
        "hardware_price": hardware_price,
        "amortization_hours": amortization_hours,
        "electricity_per_kwh": electricity_per_kwh,
    }


def _performance_score(front: str, metric: dict, config: dict) -> Optional[float]:
    value = _num(metric.get("value"))
    if value is None:
        return None
    refs = config.get("score_references", {}) if isinstance(config, dict) else {}
    ref_cfg = refs.get(front, {}) if isinstance(refs, dict) else {}
    reference = _num(ref_cfg.get(metric["name"]))
    if reference is None:
        reference = DEFAULT_METRIC_REFERENCES.get(front, {}).get(metric["name"])
    if not reference:
        return None
    return round(max(0.0, min(200.0, value / reference * 100.0)), 2)


def compute_scorecard(result: dict, hardware_config: Optional[dict] = None) -> dict:
    hardware_config = hardware_config or {}
    front = _front(result)
    metric = _primary_metric(result)
    metric_value = _num(metric.get("value"))
    cm = _cost_model(hardware_config)
    power_w = _peak_or_avg_power(result)
    resource_raw = get_nested(result, "benchmark_results.resource_usage", {})
    resource = normalize_resource_usage(resource_raw)
    energy_j = _num(resource.get("total_energy_j"))
    energy_coverage = _num(resource.get("energy_coverage_ratio"))
    resource_ops = _num(resource.get("inference_success_count"))
    resource_duration = _num(resource.get("measurement_duration_sec"))
    resource_rate = _resource_primary_rate(result, metric, resource)
    energy_derived = False

    if ((energy_j is None or (energy_coverage is not None and energy_coverage < 0.9))
            and power_w is not None and resource_duration is not None):
        energy_j = power_w * resource_duration
        energy_derived = True

    energy_per_op = None
    if energy_j is not None and resource_ops is not None and resource_ops > 0:
        energy_per_op = energy_j / resource_ops

    primary_unit_count = resource_ops
    primary_count_unit = "operation"
    if metric.get("name") == "tokens_per_sec" and resource_ops is not None:
        primary_unit_count = _num(get_nested(resource, "result_metrics.total_tokens.sum"))
        if primary_unit_count is None:
            avg_tokens = _num(get_nested(resource, "result_metrics.total_tokens.avg"))
            primary_unit_count = resource_ops * avg_tokens if avg_tokens is not None else None
        primary_count_unit = "token"
    elif metric.get("name") == "audio_realtime_factor_inverse" and resource_ops is not None:
        primary_unit_count = _num(get_nested(resource, "result_metrics.duration_sec.sum"))
        if primary_unit_count is None:
            avg_audio_duration = _num(get_nested(resource, "result_metrics.duration_sec.avg"))
            primary_unit_count = (
                resource_ops * avg_audio_duration if avg_audio_duration is not None else None
            )
        primary_count_unit = "audio_second"

    energy_per_primary_unit = None
    if energy_j is not None and primary_unit_count and primary_unit_count > 0:
        energy_per_primary_unit = energy_j / primary_unit_count

    throughput_per_watt = None
    if resource_rate is not None and power_w is not None and power_w > 0:
        throughput_per_watt = resource_rate / power_w

    hourly_cost = None
    if cm["hardware_price"] is not None and cm["amortization_hours"]:
        hourly_cost = cm["hardware_price"] / cm["amortization_hours"]
    if power_w is not None and cm["electricity_per_kwh"] is not None:
        hourly_cost = (hourly_cost or 0.0) + (power_w / 1000.0) * cm["electricity_per_kwh"]

    cost_per_1k_primary = None
    if hourly_cost is not None and resource_rate is not None and resource_rate > 0:
        cost_per_1k_primary = hourly_cost / (resource_rate * 3600.0) * 1000.0

    warnings = []
    if power_w is None:
        warnings.append("power telemetry unavailable; efficiency/cost energy fields are partial")
    if metric_value is None:
        warnings.append("primary workload metric unavailable")
    if resource_rate is None:
        warnings.append("resource-phase primary rate unavailable; efficiency fields are partial")
    throughput_error_rate = _num(get_nested(
        result, "benchmark_results.throughput.error_rate")) or 0.0
    if throughput_error_rate > 0:
        warnings.append("throughput includes failed attempts; inspect quality fields")
    if resource.get("ram_scope") == "legacy_unspecified":
        warnings.append("legacy RAM metric has no workload or host scope")
    if energy_coverage is not None and energy_coverage < 0.9:
        warnings.append("energy telemetry covered less than 90% of the resource phase")
    if energy_derived:
        warnings.append("energy estimated from average power and resource measurement duration")
    if resource.get("thermal_throttling_observed"):
        warnings.append("thermal throttling was active during the resource phase")

    return {
        "schema_version": "scorecard-2026.07-v2",
        "front": front,
        "mode": _mode(result),
        "primary_metric": metric,
        "performance_score": _performance_score(front, metric, hardware_config),
        "efficiency": {
            "avg_power_w": round(power_w, 3) if power_w is not None else None,
            "energy_j": round(energy_j, 3) if energy_j is not None else None,
            "energy_source": "derived" if energy_derived else resource.get("energy_source"),
            "energy_coverage_ratio": energy_coverage,
            "energy_per_op_j": round(energy_per_op, 6) if energy_per_op is not None else None,
            "primary_unit_count": round(primary_unit_count, 3) if primary_unit_count is not None else None,
            "primary_count_unit": primary_count_unit,
            "energy_per_primary_unit_j": round(energy_per_primary_unit, 6) if energy_per_primary_unit is not None else None,
            "primary_metric_per_watt": round(throughput_per_watt, 6) if throughput_per_watt is not None else None,
            "resource_primary_rate": round(resource_rate, 6) if resource_rate is not None else None,
        },
        "cost": {
            "currency": cm["currency"],
            "hourly_cost": round(hourly_cost, 6) if hourly_cost is not None else None,
            "cost_per_1k_primary_units": round(cost_per_1k_primary, 8) if cost_per_1k_primary is not None else None,
        },
        "comparison_policy": {
            "same_front_required": True,
            "same_model_required": True,
            "same_profile_required": True,
            "same_dataset_required": True,
            "architecture_agnostic": True,
        },
        "measurement_basis": {
            "performance": "throughput_phase",
            "energy_and_efficiency": "resource_usage_phase",
            "resource_duration_sec": resource_duration,
            "resource_operation_count": resource_ops,
            "ram_scope": resource.get("ram_scope"),
            "gpu_scope": resource.get("gpu_scope"),
            "vram_scope": resource.get("vram_scope"),
            "power_scope": resource.get("power_scope"),
            "energy_coverage_ratio": energy_coverage,
        },
        "warnings": warnings,
    }
