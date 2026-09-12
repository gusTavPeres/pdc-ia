"""Telemetria de dispositivos exposta pelo host Linux."""

from __future__ import annotations

import glob
import json
import math
import os
import platform
import re
import tempfile
import time
from typing import Any, Optional

HOST_TELEMETRY_SCHEMA = "host-telemetry-2026.08-v1"
PROVIDER_TELEMETRY_SCHEMA = "accelerator-telemetry-2026.08-v1"


def finite_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _json_document_safe(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, list):
        return all(_json_document_safe(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_document_safe(item)
            for key, item in value.items()
        )
    return False


def read_text(path: str) -> Optional[str]:
    # Ponto único de toda leitura de sysfs (thermal, hwmon, devfreq, cpufreq).
    # Sensores como os thermal_zone do Jetson retornam EAGAIN e o codec pode
    # levantar TypeError, não só OSError. Um coletor de telemetria nunca pode
    # derrubar o benchmark: qualquer falha de leitura vira campo ausente.
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().replace("\x00", " ").strip()
    except Exception:
        return None


def _normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def detect_platform(sys_root: str = "/sys", proc_root: str = "/proc") -> dict:
    model = None
    for path in (
        os.path.join(sys_root, "firmware/devicetree/base/model"),
        os.path.join(proc_root, "device-tree/model"),
    ):
        model = read_text(path)
        if model:
            break

    compatible = None
    for path in (
        os.path.join(sys_root, "firmware/devicetree/base/compatible"),
        os.path.join(proc_root, "device-tree/compatible"),
    ):
        compatible = read_text(path)
        if compatible:
            break

    identity = " ".join(filter(None, (model, compatible))).lower()
    architecture = platform.machine()
    if "raspberry pi" in identity or "raspberrypi" in identity:
        family = "raspberry_pi"
    elif "jetson" in identity or "nvidia,tegra" in identity:
        family = "jetson"
    elif architecture.lower() in {"aarch64", "arm64", "armv7l", "armv8l"}:
        family = "linux_arm"
    else:
        family = "linux"
    return {
        "family": family,
        "model": model,
        "compatible": compatible,
        "architecture": architecture,
    }


_BOARD_RAIL_PRIORITY = (
    "vdd_in",
    "vin_sys_5v0",
    "pom_5v_in",
    "sys5v",
    "vdd_sys",
    "board_power",
    "system_power",
    "input_power",
)


def _select_board_rail(rails: list[dict]) -> Optional[dict]:
    by_label = {_normalize_label(item.get("label")): item for item in rails}
    for label in _BOARD_RAIL_PRIORITY:
        if label in by_label:
            return by_label[label]
    return None


def parse_tegrastats(line: str) -> dict:
    """Converte uma linha do tegrastats para o contrato comum do agente."""
    result = {
        "sources": ["tegrastats"],
        "platform": {"family": "jetson"},
        "cpu": {},
        "memory": {},
        "gpu": {},
        "memory_controller": {},
        "power": {"rails": []},
        "temperatures": [],
        "accelerators": [],
    }
    if not isinstance(line, str) or not line.strip():
        return result

    ram = re.search(r"\bRAM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB\b", line)
    if ram:
        result["memory"] = {
            "used_mb": float(ram.group(1)),
            "total_mb": float(ram.group(2)),
            "kind": "unified_system_memory",
        }

    cpu = re.search(r"\bCPU\s*\[([^]]+)\]", line)
    if cpu:
        utilizations = [
            float(value)
            for value in re.findall(r"(\d+(?:\.\d+)?)%(?:@\d+)?", cpu.group(1))
        ]
        frequencies = [
            float(value)
            for value in re.findall(r"%@(\d+(?:\.\d+)?)", cpu.group(1))
        ]
        result["cpu"] = {
            "utilization_pct": round(sum(utilizations) / len(utilizations), 3)
            if utilizations else None,
            "frequency_mhz": max(frequencies) if frequencies else None,
            "cores_reported": len(utilizations),
        }

    gpu = re.search(
        r"\bGR3D(?:_FREQ)?\s+(\d+(?:\.\d+)?)%"
        r"(?:@(?:\[)?(\d+(?:\.\d+)?))?",
        line,
    )
    if gpu:
        result["gpu"] = {
            "name": "NVIDIA Jetson integrated GPU",
            "utilization_pct": float(gpu.group(1)),
            "frequency_mhz": float(gpu.group(2)) if gpu.group(2) else None,
            "memory_kind": "unified_system_memory",
            "scope": "visible_devices",
            "source": "tegrastats_gr3d",
        }
        result["accelerators"].append({
            "id": "jetson-gpu",
            "kind": "gpu",
            **result["gpu"],
        })

    emc = re.search(
        r"\bEMC(?:_FREQ)?\s+(\d+(?:\.\d+)?)%(?:@(\d+(?:\.\d+)?))?",
        line,
    )
    if emc:
        result["memory_controller"] = {
            "utilization_pct": float(emc.group(1)),
            "frequency_mhz": float(emc.group(2)) if emc.group(2) else None,
        }
    else:
        emc_frequency = re.search(r"\bEMC(?:_FREQ)?\s+@(\d+(?:\.\d+)?)", line)
        if emc_frequency:
            result["memory_controller"] = {
                "utilization_pct": None,
                "frequency_mhz": float(emc_frequency.group(1)),
            }

    for match in re.finditer(
            r"\b(NVDLA\d*)\s+(\d+(?:\.\d+)?)%(?:@(\d+(?:\.\d+)?))?", line):
        result["accelerators"].append({
            "id": f"jetson-{match.group(1).lower()}",
            "kind": "npu",
            "name": match.group(1),
            "utilization_pct": float(match.group(2)),
            "frequency_mhz": float(match.group(3)) if match.group(3) else None,
            "scope": "visible_devices",
            "source": "tegrastats_nvdla",
        })

    for match in re.finditer(
            r"\b(PVA\d*(?:_FREQ)?)\s+\[([^]]+)](?:@(\d+(?:\.\d+)?))?", line):
        values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)%", match.group(2))]
        name = match.group(1).removesuffix("_FREQ")
        result["accelerators"].append({
            "id": f"jetson-{name.lower()}",
            "kind": "accelerator",
            "name": name,
            "utilization_pct": max(values) if values else None,
            "frequency_mhz": float(match.group(3)) if match.group(3) else None,
            "scope": "visible_devices",
            "source": "tegrastats_pva",
        })

    known_ids = {item.get("id") for item in result["accelerators"]}
    for match in re.finditer(r"\b(NVDLA\d*)\s+@(\d+(?:\.\d+)?)", line):
        device_id = f"jetson-{match.group(1).lower()}"
        if device_id in known_ids:
            continue
        result["accelerators"].append({
            "id": device_id,
            "kind": "npu",
            "name": match.group(1),
            "utilization_pct": None,
            "frequency_mhz": float(match.group(2)),
            "scope": "visible_devices",
            "source": "tegrastats_nvdla",
        })
        known_ids.add(device_id)

    for match in re.finditer(r"\b(PVA\d*)(?:_FREQ)?\s+@(\d+(?:\.\d+)?)", line):
        device_id = f"jetson-{match.group(1).lower()}"
        if device_id in known_ids:
            continue
        result["accelerators"].append({
            "id": device_id,
            "kind": "accelerator",
            "name": match.group(1),
            "utilization_pct": None,
            "frequency_mhz": float(match.group(2)),
            "scope": "visible_devices",
            "source": "tegrastats_pva",
        })
        known_ids.add(device_id)

    fixed_engine = re.compile(
        r"\b(VIC(?:_LOAD|_FREQ)?|OFA|NVENC\d*|NVDEC\d*|NVJPG\d*)\s+"
        r"(\d+(?:\.\d+)?)%(?:@(\d+(?:\.\d+)?))?"
    )
    for match in fixed_engine.finditer(line):
        name = match.group(1).removesuffix("_LOAD").removesuffix("_FREQ")
        result["accelerators"].append({
            "id": f"jetson-{name.lower()}",
            "kind": "accelerator",
            "name": name,
            "utilization_pct": float(match.group(2)),
            "frequency_mhz": float(match.group(3)) if match.group(3) else None,
            "scope": "visible_devices",
            "source": "tegrastats_engine",
        })

    for match in re.finditer(r"(?<![\w])([A-Za-z][\w-]*)@(-?\d+(?:\.\d+)?)C\b", line):
        value = finite_number(match.group(2))
        if value is None or not -50 <= value <= 250:
            continue
        result["temperatures"].append({
            "label": match.group(1),
            "temperature_c": value,
            "source": "tegrastats",
        })

    rails: dict[str, dict] = {}
    explicit_power = re.compile(
        r"\b([A-Z][A-Z0-9_]*)\s+(\d+(?:\.\d+)?)mW/(\d+(?:\.\d+)?)mW\b"
    )
    legacy_power = re.compile(
        r"\b([A-Z][A-Z0-9_]*)\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)(?![\dA-Za-z])"
    )
    for expression in (explicit_power, legacy_power):
        for match in expression.finditer(line):
            label = match.group(1)
            if label in rails:
                continue
            rails[label] = {
                "label": label,
                "power_w": float(match.group(2)) / 1000.0,
                "average_power_w": float(match.group(3)) / 1000.0,
                "source": "tegrastats",
            }
    result["power"]["rails"] = list(rails.values())
    board_rail = _select_board_rail(result["power"]["rails"])
    if board_rail:
        result["power"].update({
            "board_w": board_rail["power_w"],
            "board_source": f"tegrastats:{board_rail['label']}",
            "scope": "host_board",
        })

    gpu_temperatures = [
        item["temperature_c"] for item in result["temperatures"]
        if "gpu" in _normalize_label(item.get("label"))
    ]
    if gpu_temperatures and result["gpu"]:
        result["gpu"]["temperature_c"] = max(gpu_temperatures)
        result["accelerators"][0]["temperature_c"] = max(gpu_temperatures)
    return result


def _classify_accelerator(value: str) -> Optional[str]:
    text = value.lower()
    if any(marker in text for marker in (
            "npu", "rknpu", "ethos", "hailo", "apex", "neural", "ivpu", "xdna")):
        return "npu"
    if any(marker in text for marker in (
            "gpu", "nvgpu", "mali", "panfrost", "adreno", "etnaviv", "vivante", "3d")):
        return "gpu"
    if any(marker in text for marker in ("dsp", "vpu")):
        return "accelerator"
    return None


class SysfsTelemetryCollector:
    """Lê interfaces padronizadas do kernel sem depender de um fabricante."""

    def __init__(self, sys_root: str = "/sys", proc_root: str = "/proc"):
        self.sys_root = os.path.abspath(sys_root)
        self.proc_root = os.path.abspath(proc_root)
        self.platform = detect_platform(self.sys_root, self.proc_root)
        self._devfreq_counters: dict[str, tuple[float, float]] = {}

    def _thermal_sensors(self) -> list[dict]:
        sensors = []
        pattern = os.path.join(self.sys_root, "class/thermal/thermal_zone*/temp")
        for path in sorted(glob.glob(pattern)):
            value = finite_number(read_text(path))
            if value is None:
                continue
            value = value / 1000.0 if abs(value) > 500 else value
            if not -50 <= value <= 250:
                continue
            base = os.path.dirname(path)
            sensors.append({
                "label": read_text(os.path.join(base, "type")) or os.path.basename(base),
                "temperature_c": round(value, 3),
                "source": "thermal_sysfs",
            })
        return sensors

    def _devfreq_utilization(self, base: str) -> Optional[float]:
        busy = finite_number(read_text(os.path.join(base, "busy_time")))
        total = finite_number(read_text(os.path.join(base, "total_time")))
        previous = self._devfreq_counters.get(base)
        if busy is not None and total is not None:
            self._devfreq_counters[base] = (busy, total)
            if previous:
                busy_delta = busy - previous[0]
                total_delta = total - previous[1]
                if total_delta > 0 and 0 <= busy_delta <= total_delta:
                    return min(100.0, busy_delta / total_delta * 100.0)

        load_text = read_text(os.path.join(base, "load")) or ""
        match = re.search(r"-?\d+(?:\.\d+)?", load_text)
        raw = finite_number(match.group(0)) if match else None
        if raw is None or raw < 0:
            return None
        if raw <= 100:
            return raw
        if raw <= 1000:
            return raw / 10.0
        return None

    def _devfreq_devices(self) -> list[dict]:
        devices = []
        pattern = os.path.join(self.sys_root, "class/devfreq/*")
        for base in sorted(glob.glob(pattern)):
            name = read_text(os.path.join(base, "name")) or os.path.basename(base)
            identity = f"{name} {os.path.realpath(base)}"
            kind = _classify_accelerator(identity)
            if kind is None:
                continue
            frequency = finite_number(read_text(os.path.join(base, "cur_freq")))
            if frequency is not None:
                frequency = frequency / 1_000_000.0 if frequency > 100_000 else frequency
            utilization = self._devfreq_utilization(base)
            devices.append({
                "id": f"devfreq:{os.path.basename(base)}",
                "kind": kind,
                "name": name,
                "utilization_pct": round(utilization, 3) if utilization is not None else None,
                "frequency_mhz": round(frequency, 3) if frequency is not None else None,
                "scope": "visible_devices",
                "source": "linux_devfreq",
            })
        return devices

    def _hwmon(self) -> tuple[list[dict], list[dict], list[dict]]:
        temperatures = []
        powers = []
        energies = []
        pattern = os.path.join(self.sys_root, "class/hwmon/hwmon*")
        for base in sorted(glob.glob(pattern)):
            chip = read_text(os.path.join(base, "name")) or os.path.basename(base)
            for path in sorted(glob.glob(os.path.join(base, "temp[0-9]*_input"))):
                value = finite_number(read_text(path))
                if value is None:
                    continue
                value = value / 1000.0 if abs(value) > 500 else value
                if not -50 <= value <= 250:
                    continue
                prefix = path[:-len("_input")]
                temperatures.append({
                    "label": read_text(prefix + "_label") or f"{chip}:{os.path.basename(prefix)}",
                    "temperature_c": round(value, 3),
                    "source": f"hwmon:{chip}",
                })
            for path in sorted(glob.glob(os.path.join(base, "power[0-9]*_input"))):
                value = finite_number(read_text(path))
                if value is None:
                    continue
                watts = value / 1_000_000.0
                if not 0 <= watts <= 100_000:
                    continue
                prefix = path[:-len("_input")]
                powers.append({
                    "label": read_text(prefix + "_label") or f"{chip}:{os.path.basename(prefix)}",
                    "power_w": round(watts, 6),
                    "source": f"hwmon:{chip}",
                })
            for path in sorted(glob.glob(os.path.join(base, "energy[0-9]*_input"))):
                value = finite_number(read_text(path))
                if value is None:
                    continue
                prefix = path[:-len("_input")]
                energies.append({
                    "label": read_text(prefix + "_label") or f"{chip}:{os.path.basename(prefix)}",
                    "energy_j": round(value / 1_000_000.0, 6),
                    "source": f"hwmon:{chip}",
                })
        return temperatures, powers, energies

    def _cpu_frequency(self) -> Optional[float]:
        values = []
        pattern = os.path.join(
            self.sys_root, "devices/system/cpu/cpufreq/policy*/scaling_cur_freq")
        for path in glob.glob(pattern):
            value = finite_number(read_text(path))
            if value is not None and value >= 0:
                values.append(value / 1000.0)
        return max(values) if values else None

    def _thermal_throttling(self) -> dict:
        devices_by_name = {}
        pattern = os.path.join(self.sys_root, "class/thermal/cooling_device*")
        for base in sorted(glob.glob(pattern)):
            current = finite_number(read_text(os.path.join(base, "cur_state")))
            maximum = finite_number(read_text(os.path.join(base, "max_state")))
            if current is None:
                continue
            name = read_text(os.path.join(base, "type")) or os.path.basename(base)
            normalized = _normalize_label(name)
            limits_compute = "fan" not in normalized and any(
                marker in normalized for marker in (
                    "cpu", "processor", "cpufreq", "gpu", "devfreq", "thermal"))
            device = {
                "name": name,
                "state": int(current),
                "max_state": int(maximum) if maximum is not None else None,
                "limits_compute": limits_compute,
                "count": 1,
            }
            existing = devices_by_name.get(name)
            if existing:
                existing["state"] = max(existing["state"], device["state"])
                maximums = [
                    value for value in (existing["max_state"], device["max_state"])
                    if value is not None
                ]
                existing["max_state"] = max(maximums) if maximums else None
                existing["limits_compute"] = (
                    existing["limits_compute"] or device["limits_compute"])
                existing["count"] += 1
            else:
                devices_by_name[name] = device
        devices = list(devices_by_name.values())
        return {
            "active": any(item["state"] > 0 and item["limits_compute"] for item in devices),
            "devices": devices,
            "source": "thermal_cooling_sysfs",
        }

    def snapshot(self) -> dict:
        thermal = self._thermal_sensors()
        hwmon_thermal, powers, energies = self._hwmon()
        temperatures = thermal + hwmon_thermal
        accelerators = self._devfreq_devices()
        gpu_devices = [item for item in accelerators if item.get("kind") == "gpu"]

        cpu_temperatures = [
            item["temperature_c"] for item in temperatures
            if any(marker in _normalize_label(
                f"{item.get('label', '')} {item.get('source', '')}")
                for marker in (
                    "cpu", "soc", "package", "core", "thermal", "k10temp", "tctl", "tdie"))
        ]
        gpu_temperatures = [
            item["temperature_c"] for item in temperatures
            if "gpu" in _normalize_label(item.get("label"))
        ]
        cpu_frequency = self._cpu_frequency()
        board_rail = _select_board_rail(powers)
        gpu = {}
        if gpu_devices:
            utilization = [finite_number(item.get("utilization_pct")) for item in gpu_devices]
            utilization = [value for value in utilization if value is not None]
            frequencies = [finite_number(item.get("frequency_mhz")) for item in gpu_devices]
            frequencies = [value for value in frequencies if value is not None]
            gpu = {
                "name": " / ".join(str(item.get("name") or item.get("id"))
                                     for item in gpu_devices),
                "utilization_pct": max(utilization) if utilization else None,
                "frequency_mhz": max(frequencies) if frequencies else None,
                "temperature_c": max(gpu_temperatures) if gpu_temperatures else None,
                "scope": "visible_devices",
                "source": "linux_devfreq",
            }

        power = {"sensors": powers, "energy_sensors": energies}
        if board_rail:
            power.update({
                "board_w": board_rail["power_w"],
                "board_source": f"{board_rail['source']}:{board_rail['label']}",
                "scope": "host_board",
            })
        return {
            "sources": ["linux_sysfs"],
            "platform": self.platform,
            "cpu": {
                "temperature_c": max(cpu_temperatures) if cpu_temperatures else None,
                "frequency_mhz": cpu_frequency,
            },
            "gpu": gpu,
            "power": power,
            "throttling": self._thermal_throttling(),
            "temperatures": temperatures,
            "accelerators": accelerators,
        }


def merge_telemetry(base: dict, overlay: dict) -> dict:
    """Combina fontes mantendo listas de sensores e preenchendo campos mais específicos."""
    result = dict(base or {})
    overlay = overlay if isinstance(overlay, dict) else {}
    for key in (
            "platform", "cpu", "memory", "gpu", "memory_controller", "power", "throttling"):
        current = result.get(key) if isinstance(result.get(key), dict) else {}
        incoming = overlay.get(key) if isinstance(overlay.get(key), dict) else {}
        merged = dict(current)
        merged.update({name: value for name, value in incoming.items() if value is not None})
        if merged:
            result[key] = merged

    sources = list(dict.fromkeys([
        *[str(value) for value in result.get("sources", []) if value],
        *[str(value) for value in overlay.get("sources", []) if value],
    ]))
    result["sources"] = sources
    for key in ("temperatures", "accelerators"):
        items = []
        positions = {}
        for item in [*result.get(key, []), *overlay.get(key, [])]:
            if not isinstance(item, dict):
                continue
            identity = str(item.get("id") or item.get("label") or item.get("name") or item)
            if identity in positions:
                merged = dict(items[positions[identity]])
                merged.update({name: value for name, value in item.items()
                               if value is not None})
                items[positions[identity]] = merged
            else:
                positions[identity] = len(items)
                items.append(dict(item))
        result[key] = items
    return result


def load_provider_snapshots(directory: str, max_age_sec: float = 5.0) -> list[dict]:
    snapshots = []
    now = time.time()
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            if os.path.getsize(path) > 1024 * 1024:
                continue
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (not isinstance(payload, dict) or not _json_document_safe(payload)
                or payload.get("schema_version") != PROVIDER_TELEMETRY_SCHEMA):
            continue
        timestamp = finite_number(payload.get("timestamp_epoch"))
        if timestamp is None or now - timestamp > max_age_sec or timestamp - now > 60:
            continue
        provider = str(payload.get("provider") or os.path.basename(path))
        payload = dict(payload)
        provider_sources = payload.get("sources", [])
        provider_sources = provider_sources if isinstance(provider_sources, list) else []
        payload["sources"] = list(dict.fromkeys([
            *provider_sources,
            f"external:{provider}",
        ]))
        snapshots.append(payload)
    return snapshots


def load_host_telemetry(path: str, max_age_sec: float = 5.0,
                        now: Optional[float] = None) -> Optional[dict]:
    try:
        if os.path.getsize(path) > 4 * 1024 * 1024:
            return None
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (not isinstance(payload, dict) or not _json_document_safe(payload)
            or payload.get("schema_version") != HOST_TELEMETRY_SCHEMA):
        return None
    timestamp = finite_number(payload.get("timestamp_epoch"))
    current = time.time() if now is None else now
    if timestamp is None or current - timestamp > max_age_sec or timestamp - current > 60:
        return None
    return payload


def write_host_telemetry(path: str, payload: dict) -> None:
    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    document = dict(payload)
    document["schema_version"] = HOST_TELEMETRY_SCHEMA
    document.setdefault("timestamp_epoch", time.time())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=".telemetry-", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(document, handle, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    except Exception:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise
