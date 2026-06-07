"""
scorecard.py — Architecture-neutral hardware benchmark scorecard.

Scores are designed for comparing many hardware targets while staying honest
about missing telemetry. They avoid architecture-specific assumptions and keep
the raw primary metric visible beside derived efficiency/cost fields.
"""

from typing import Any, Dict, Optional

from shared.methodology import get_nested


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
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


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

    if front == "slm":
        tps = _num(get_nested(result_metrics, "tokens_per_sec.avg"))
        if tps is not None:
            return {"name": "tokens_per_sec", "value": tps, "unit": "tokens/s", "higher_is_better": True}

    if front == "audio" and _mode(result) == "asr":
        rtf = _num(get_nested(result, "pipeline_results.avg_rtf"))
        if rtf and rtf > 0:
            return {"name": "audio_realtime_factor_inverse", "value": round(1.0 / rtf, 4),
                    "unit": "x real-time", "higher_is_better": True}

    ops = _num(tp.get("ops_per_sec"))
    unit = "frames/s" if front == "vision" else "ops/s"
    return {"name": "ops_per_sec", "value": ops, "unit": unit,
            "higher_is_better": True}


def _peak_or_avg_power(result: dict) -> Optional[float]:
    res = result.get("benchmark_results", {}).get("resource_usage", {})
    avg_power = _num(res.get("avg_power_w"))
    if avg_power is not None:
        return avg_power
    samples = res.get("timeseries", [])
    vals = [_num(s.get("power_w") or s.get("gpu_power_w")) for s in samples if isinstance(s, dict)]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


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
    energy_j = _num(get_nested(result, "benchmark_results.resource_usage.total_energy_j"))
    total_ops = _num(get_nested(result, "benchmark_results.throughput.ops_total"))
    duration_s = _num(get_nested(result, "benchmark_results.throughput.duration_sec"))

    if energy_j is None and power_w is not None and duration_s is not None:
        energy_j = power_w * duration_s

    energy_per_primary = None
    if energy_j is not None and total_ops and total_ops > 0:
        energy_per_primary = energy_j / total_ops

    primary_unit_count = total_ops
    if metric.get("name") == "tokens_per_sec" and total_ops:
        avg_tokens = _num(get_nested(result, "benchmark_results.throughput.result_metrics.total_tokens.avg"))
        if avg_tokens is not None:
            primary_unit_count = total_ops * avg_tokens

    energy_per_primary_unit = None
    if energy_j is not None and primary_unit_count and primary_unit_count > 0:
        energy_per_primary_unit = energy_j / primary_unit_count

    throughput_per_watt = None
    if metric_value is not None and power_w and power_w > 0:
        throughput_per_watt = metric_value / power_w

    hourly_cost = None
    if cm["hardware_price"] is not None and cm["amortization_hours"]:
        hourly_cost = cm["hardware_price"] / cm["amortization_hours"]
    if power_w is not None and cm["electricity_per_kwh"] is not None:
        hourly_cost = (hourly_cost or 0.0) + (power_w / 1000.0) * cm["electricity_per_kwh"]

    cost_per_1k_primary = None
    if hourly_cost is not None and metric_value and metric_value > 0:
        cost_per_1k_primary = hourly_cost / (metric_value * 3600.0) * 1000.0

    warnings = []
    if power_w is None:
        warnings.append("power telemetry unavailable; efficiency/cost energy fields are partial")
    if metric_value is None:
        warnings.append("primary workload metric unavailable")
    if get_nested(result, "benchmark_results.throughput.error_rate", 0) > 0:
        warnings.append("throughput includes failed attempts; inspect quality fields")

    return {
        "schema_version": "scorecard-2026.05-v1",
        "front": front,
        "mode": _mode(result),
        "primary_metric": metric,
        "performance_score": _performance_score(front, metric, hardware_config),
        "efficiency": {
            "avg_power_w": round(power_w, 3) if power_w is not None else None,
            "energy_j": round(energy_j, 3) if energy_j is not None else None,
            "energy_per_op_j": round(energy_per_primary, 6) if energy_per_primary is not None else None,
            "primary_unit_count": round(primary_unit_count, 3) if primary_unit_count is not None else None,
            "energy_per_primary_unit_j": round(energy_per_primary_unit, 6) if energy_per_primary_unit is not None else None,
            "primary_metric_per_watt": round(throughput_per_watt, 6) if throughput_per_watt is not None else None,
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
        "warnings": warnings,
    }
