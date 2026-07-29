"""Validação dos perfis de benchmark."""

import math
from numbers import Real


def _number(value, label, minimum=None, maximum=None, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} deve ser numérico")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} deve ser finito")
    if strictly_positive and value <= 0:
        raise ValueError(f"{label} deve ser maior que zero")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} deve ser maior ou igual a {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} deve ser menor ou igual a {maximum}")


def _integer(value, label, minimum=None, maximum=None, strictly_positive=False):
    _number(value, label, minimum=minimum, maximum=maximum,
            strictly_positive=strictly_positive)
    if int(value) != value:
        raise ValueError(f"{label} deve ser inteiro")


def _enabled(section, label, default):
    value = section.get("enabled", default)
    if not isinstance(value, bool):
        raise ValueError(f"{label}.enabled deve ser booleano")
    return value


def validate_profile_config(profile: dict, name: str = "profile") -> dict:
    if not isinstance(profile, dict) or not profile:
        raise ValueError(f"{name} deve ser um objeto não vazio")

    throughput = profile.get("throughput")
    if throughput is not None:
        if not isinstance(throughput, dict):
            raise ValueError(f"{name}.throughput deve ser um objeto")
        if _enabled(throughput, f"{name}.throughput", True):
            _number(throughput.get("duration_sec"),
                    f"{name}.throughput.duration_sec", maximum=86_400,
                    strictly_positive=True)
            _integer(throughput.get("warmup", 0),
                     f"{name}.throughput.warmup", minimum=0, maximum=10_000)

    latency = profile.get("latency")
    if latency is not None:
        if not isinstance(latency, dict):
            raise ValueError(f"{name}.latency deve ser um objeto")
        if _enabled(latency, f"{name}.latency", True):
            _integer(latency.get("samples"), f"{name}.latency.samples",
                     maximum=100_000, strictly_positive=True)
            percentiles = latency.get("percentiles")
            if not isinstance(percentiles, list) or not percentiles:
                raise ValueError(f"{name}.latency.percentiles deve ser uma lista não vazia")
            for percentile in percentiles:
                _number(percentile, f"{name}.latency.percentiles",
                        minimum=0, maximum=100)

    resources = profile.get("resource_usage")
    if resources is not None:
        if not isinstance(resources, dict):
            raise ValueError(f"{name}.resource_usage deve ser um objeto")
        if _enabled(resources, f"{name}.resource_usage", True):
            _number(resources.get("duration_sec"),
                    f"{name}.resource_usage.duration_sec", maximum=86_400,
                    strictly_positive=True)
            _number(resources.get("interval_sec"),
                    f"{name}.resource_usage.interval_sec", minimum=0.05,
                    maximum=3_600, strictly_positive=True)

    sustained = profile.get("sustained_load")
    if sustained is not None:
        if not isinstance(sustained, dict):
            raise ValueError(f"{name}.sustained_load deve ser um objeto")
        _enabled(sustained, f"{name}.sustained_load", True)
        _integer(sustained.get("duration_min", 0),
                 f"{name}.sustained_load.duration_min", minimum=0, maximum=1_440)

    memory = profile.get("memory_bandwidth")
    if memory is not None:
        if not isinstance(memory, dict):
            raise ValueError(f"{name}.memory_bandwidth deve ser um objeto")
        if _enabled(memory, f"{name}.memory_bandwidth", False):
            patterns = memory.get("patterns")
            if not isinstance(patterns, list) or not patterns:
                raise ValueError(f"{name}.memory_bandwidth.patterns deve ser uma lista")
            unknown = set(patterns) - {"sequential", "random"}
            if unknown:
                raise ValueError(f"{name}.memory_bandwidth.patterns contém: {sorted(unknown)}")
            sizes = memory.get("block_sizes_kb")
            if not isinstance(sizes, list) or not sizes:
                raise ValueError(f"{name}.memory_bandwidth.block_sizes_kb deve ser uma lista")
            for size in sizes:
                _integer(size, f"{name}.memory_bandwidth.block_sizes_kb",
                         maximum=262_144, strictly_positive=True)
    return profile


def validate_benchmark_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("A configuração deve ser um objeto YAML")
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("profiles deve conter pelo menos um perfil")
    for name, profile in profiles.items():
        validate_profile_config(profile, f"profiles.{name}")
    default_profile = config.get("default_profile")
    if default_profile and default_profile not in profiles:
        raise ValueError("default_profile não existe em profiles")
    return config
