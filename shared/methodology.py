"""
methodology.py — Shared benchmark methodology metadata and result helpers.

The schema here is intentionally architecture-neutral. It describes the
workload, sampling policy, errors, and comparability constraints without
assuming x86, ARM, NVIDIA, CPU-only, or accelerator-specific behavior.
"""

import platform
import statistics
from typing import Any, Dict, Iterable, List, Optional


METHODOLOGY_VERSION = "2026.05-v1"


def base_methodology(profile_name: Optional[str], profile_config: dict) -> dict:
    """Return methodology metadata embedded in every benchmark result."""
    return {
        "version": METHODOLOGY_VERSION,
        "profile": profile_name or "unknown",
        "principles": [
            "same workload inputs for comparable runs",
            "warmup excluded from measured throughput",
            "latency reported with percentiles from successful samples only",
            "failed inference calls are counted and reported",
            "resource samples are time-series snapshots during active load",
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
                       dataset: Optional[str]) -> dict:
    return {
        "front": front,
        "model": model,
        "mode": mode,
        "dataset": dataset,
        "comparability_key": {
            "front": front,
            "model": model,
            "mode": mode,
            "dataset": dataset,
        },
    }


def summarize_errors(errors: Iterable[str], limit: int = 5) -> dict:
    errors = list(errors)
    counts: Dict[str, int] = {}
    for err in errors:
        key = str(err)[:180]
        counts[key] = counts.get(key, 0) + 1
    return {
        "count": len(errors),
        "unique": len(counts),
        "samples": sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit],
    }


def summarize_numeric_result_fields(results: Iterable[Any]) -> dict:
    """
    Aggregate numeric fields from inference return dicts.

    This captures domain metrics such as tokens_per_sec, rtf, duration_sec, or
    time_to_first_token_ms without coupling the shared suite to one workload.
    """
    fields: Dict[str, List[float]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                fields.setdefault(key, []).append(float(value))

    summary = {}
    for key, values in fields.items():
        if not values:
            continue
        summary[key] = {
            "avg": round(statistics.mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "samples": len(values),
        }
    return summary


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
