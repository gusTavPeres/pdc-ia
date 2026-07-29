"""Metadados e utilitários comuns aos resultados de benchmark."""

import math
import platform
import statistics
from typing import Any, Dict, Iterable, List, Optional

METHODOLOGY_VERSION = "2026.07-v2"


def base_methodology(profile_name: Optional[str], profile_config: dict) -> dict:
    """Monta a metodologia registrada em cada resultado."""
    return {
        "version": METHODOLOGY_VERSION,
        "profile": profile_name or "unknown",
        "principles": [
            "same workload inputs for comparable runs",
            "warmup excluded from measured throughput",
            "latency reported with percentiles from successful samples only",
            "failed inference calls are counted and reported",
            "resource samples are time-series snapshots during active load",
            "resource metrics identify whether they describe the workload or the host device",
            "energy and operation counts use the same measurement window",
            "scores are relative indicators and keep missing energy/cost explicit",
        ],
        "architecture_policy": {
            "cpu_arch": platform.machine(),
            "portable_across": ["x86_64", "aarch64", "arm64"],
            "assumptions": [
                "do not require NVIDIA GPU metrics for validity",
                "device-specific power counters are optional",
                "compare only runs with the same front, model, profile, dataset, and mode",
            ],
        },
        "profile_config": profile_config,
    }


def benchmark_metadata(front: Optional[str], model: Any, mode: Optional[str],
                       dataset: Optional[str], execution: Optional[dict] = None,
                       profile: Optional[str] = None, run_id: Optional[str] = None) -> dict:
    return {
        "run_id": run_id,
        "front": front,
        "model": model,
        "mode": mode,
        "dataset": dataset,
        "profile": profile,
        "execution": execution or {},
        "comparability_key": {
            "methodology_version": METHODOLOGY_VERSION,
            "front": front,
            "model": model,
            "mode": mode,
            "dataset": dataset,
            "profile": profile,
            "execution": execution or {},
        },
    }


def summarize_errors(errors: Iterable[str], limit: int = 5,
                     total_count: Optional[int] = None) -> dict:
    errors = list(errors)
    counts: Dict[str, int] = {}
    for err in errors:
        key = str(err)[:180]
        counts[key] = counts.get(key, 0) + 1
    return {
        "count": total_count if total_count is not None else len(errors),
        "sampled_count": len(errors),
        "unique": len(counts),
        "samples": sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit],
    }


def summarize_numeric_result_fields(results: Iterable[Any]) -> dict:
    """Resume campos numéricos retornados pelas inferências."""
    fields: Dict[str, List[float]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                number = float(value)
                if math.isfinite(number):
                    fields.setdefault(key, []).append(number)

    summary = {}
    for key, values in fields.items():
        if not values:
            continue
        summary[key] = {
            "avg": round(statistics.mean(values), 4),
            "sum": round(sum(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "samples": len(values),
        }
    return summary


class NumericResultAccumulator:
    """Mantém estatísticas numéricas sem armazenar todas as respostas."""

    def __init__(self):
        self._fields = {}

    def update(self, item: Any):
        if not isinstance(item, dict):
            return
        for key, value in item.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if not math.isfinite(number):
                continue
            field = self._fields.setdefault(key, {
                "sum": 0.0,
                "min": number,
                "max": number,
                "samples": 0,
            })
            field["sum"] += number
            field["min"] = min(field["min"], number)
            field["max"] = max(field["max"], number)
            field["samples"] += 1

    def summary(self) -> dict:
        return {
            key: {
                "avg": round(field["sum"] / field["samples"], 4),
                "sum": round(field["sum"], 4),
                "min": round(field["min"], 4),
                "max": round(field["max"], 4),
                "samples": field["samples"],
            }
            for key, field in self._fields.items()
            if field["samples"]
        }


def get_nested(data: dict, path: str, default=None):
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def result_quality(success: int, errors: int) -> dict:
    total = success + errors
    error_rate = errors / total if total else 0.0
    return {
        "success_count": success,
        "error_count": errors,
        "total_attempts": total,
        "error_rate": round(error_rate, 4),
        "valid": success > 0 and error_rate < 0.05,
        "warnings": [] if error_rate < 0.05 else ["high inference error rate"],
    }
