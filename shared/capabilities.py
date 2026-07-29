"""Detecção portátil de hardware e sensores disponíveis."""

import csv
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime
from typing import Optional

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


HARDWARE_SNAPSHOT_SCHEMA = "hardware-snapshot-2026.07-v1"
DEFAULT_HARDWARE_SNAPSHOT_PATH = "/app/data/hardware/host.json"


def _run_cmd(cmd: list) -> Optional[str]:
    """Executa comando e retorna stdout, ou None se falhar."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def _detect_cpu() -> dict:
    """
    Detecta informações do CPU.
    Lê /proc/cpuinfo para modelo e flags, psutil para núcleos e frequência.
    """
    cpu = {
        "model": None,
        "arch": None,
        "cores_physical": None,
        "cores_logical": None,
        "freq_mhz": None,
        "cache_kb": None,
        "flags": {
            "avx": False,
            "avx2": False,
            "avx512": False,
            "vnni": False,
            "fma": False,
            "amx": False,
            "neon": False,
            "dotprod": False,
            "i8mm": False,
            "sve": False,
            "sve2": False,
        }
    }

    # Arquitetura (ex: x86_64, aarch64)
    try:
        cpu["arch"] = platform.machine()
    except Exception:
        pass

    # Modelo e flags via /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            content = f.read()
        feature_set = set()
        for line in content.splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            key = key.strip().lower()
            value = value.strip()
            if cpu["model"] is None and key in (
                    "model name", "hardware", "processor", "cpu model"):
                if value and not value.isdigit():
                    cpu["model"] = value
            if key in ("flags", "features") and not feature_set:
                feature_set = set(value.lower().split())

        cpu["flags"].update({
            "avx": "avx" in feature_set,
            "avx2": "avx2" in feature_set,
            "avx512": any(flag.startswith("avx512") for flag in feature_set),
            "vnni": "avx512_vnni" in feature_set or "avx_vnni" in feature_set,
            "fma": "fma" in feature_set,
            "amx": any(flag.startswith("amx_") for flag in feature_set),
            "neon": "neon" in feature_set or "asimd" in feature_set,
            "dotprod": "dotprod" in feature_set or "asimddp" in feature_set,
            "i8mm": "i8mm" in feature_set,
            "sve": "sve" in feature_set,
            "sve2": "sve2" in feature_set,
        })
    except Exception:
        pass

    # Núcleos e frequência via psutil
    if _PSUTIL:
        try:
            cpu["cores_physical"] = psutil.cpu_count(logical=False)
            cpu["cores_logical"] = psutil.cpu_count(logical=True)
            freq = psutil.cpu_freq()
            if freq:
                cpu["freq_mhz"] = round(freq.current, 1)
        except Exception:
            pass

    # Cache L3 via lscpu
    try:
        lscpu_out = _run_cmd(["lscpu"])
        if lscpu_out:
            for line in lscpu_out.splitlines():
                if "L3 cache" in line or "L3 Cache" in line:
                    # Ex: "L3 cache:                        8 MiB"
                    val = line.split(":")[1].strip()
                    # Converte para KB
                    if "MiB" in val or "MB" in val:
                        num = float(re.findall(r"[\d.]+", val)[0])
                        cpu["cache_kb"] = int(num * 1024)
                    elif "KiB" in val or "KB" in val:
                        num = float(re.findall(r"[\d.]+", val)[0])
                        cpu["cache_kb"] = int(num)
    except Exception:
        pass

    return cpu


def _detect_ram() -> dict:
    """
    Detecta informações de memória RAM.
    Usa psutil para total/disponível, dmidecode para tipo/velocidade (requer sudo).
    """
    ram = {
        "total_gb": None,
        "available_gb": None,
        "type": None,       # DDR4, DDR5, etc.
        "speed_mhz": None,
    }

    if _PSUTIL:
        try:
            mem = psutil.virtual_memory()
            ram["total_gb"] = round(mem.total / (1024**3), 2)
            ram["available_gb"] = round(mem.available / (1024**3), 2)
        except Exception:
            pass

    # Tipo e velocidade via dmidecode (precisa de root/sudo)
    try:
        out = _run_cmd(["dmidecode", "-t", "memory"])
        if out:
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Type:") and ram["type"] is None:
                    t = line.split(":")[1].strip()
                    if t not in ("Unknown", "Other", ""):
                        ram["type"] = t
                if line.startswith("Speed:") and ram["speed_mhz"] is None:
                    val = line.split(":")[1].strip()
                    match = re.findall(r"(\d+)", val)
                    if match:
                        ram["speed_mhz"] = int(match[0])
    except Exception:
        pass

    return ram


def _detect_gpu() -> dict:
    """Detecta as GPUs NVIDIA visíveis e mantém os campos legados agregados."""
    gpu = {
        "available": False,
        "model": None,
        "vram_mb": None,
        "cuda_version": None,
        "compute_capability": None,
        "devices": [],
    }

    if _PYNVML:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            devices = []
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                compute_capability = None
                try:
                    capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{capability[0]}.{capability[1]}"
                except Exception:
                    pass
                devices.append({
                    "index": index,
                    "model": str(name),
                    "vram_mb": round(memory.total / (1024 ** 2)),
                    "compute_capability": compute_capability,
                })
            if devices:
                gpu["available"] = True
                gpu["model"] = " / ".join(device["model"] for device in devices)
                gpu["vram_mb"] = sum(device["vram_mb"] for device in devices)
                major = cuda_version // 1000
                minor = (cuda_version % 1000) // 10
                gpu["cuda_version"] = f"{major}.{minor}"
                gpu["compute_capability"] = devices[0]["compute_capability"]
                gpu["devices"] = devices
            pynvml.nvmlShutdown()
            return gpu
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    try:
        out = _run_cmd(["nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits"])
        if out:
            devices = []
            for index, row in enumerate(csv.reader(out.splitlines())):
                parts = [part.strip() for part in row]
                if len(parts) < 2:
                    continue
                try:
                    vram_mb = round(float(parts[1]))
                except ValueError:
                    vram_mb = None
                devices.append({
                    "index": index,
                    "model": parts[0],
                    "vram_mb": vram_mb,
                    "compute_capability": None,
                    "driver_version": parts[2] if len(parts) > 2 else None,
                })
            if devices:
                gpu["available"] = True
                gpu["model"] = " / ".join(device["model"] for device in devices)
                memories = [device["vram_mb"] for device in devices
                            if device["vram_mb"] is not None]
                gpu["vram_mb"] = sum(memories) if memories else None
                gpu["devices"] = devices
    except Exception:
        pass

    try:
        nvcc_out = _run_cmd(["nvcc", "--version"])
        if nvcc_out:
            match = re.search(r"release ([\d.]+)", nvcc_out)
            if match:
                gpu["cuda_version"] = match.group(1)
    except Exception:
        pass

    return gpu


def _base_block_device_name(device: str) -> str:
    name = os.path.basename(device)
    partitioned = re.fullmatch(r"(nvme\d+n\d+|mmcblk\d+|loop\d+)p\d+", name)
    if partitioned:
        return partitioned.group(1)
    if re.fullmatch(r"nvme\d+n\d+|mmcblk\d+|loop\d+|dm-\d+|md\d+", name):
        return name
    return re.sub(r"\d+$", "", name)


def _detect_storage() -> dict:
    """
    Detecta tipo do disco onde /home está montado (SSD NVMe, SSD SATA, HDD).
    Usa lsblk para obter informações do dispositivo.
    """
    storage = {
        "type": None,       # "NVMe SSD", "SSD", "HDD"
        "device": None,
    }

    try:
        # Descobre o dispositivo que contém /home (ou /)
        out = _run_cmd(["df", "/home"])
        if not out:
            out = _run_cmd(["df", "/"])
        if out:
            device = out.splitlines()[-1].split()[0]
            storage["device"] = device

            parent = _run_cmd(["lsblk", "-no", "PKNAME", device])
            base_dev = parent.splitlines()[0].strip() if parent else \
                _base_block_device_name(device)

            # Verifica se é NVMe
            if "nvme" in base_dev.lower():
                storage["type"] = "NVMe SSD"
            else:
                # Verifica rotacional via sysfs
                rotational_path = f"/sys/block/{base_dev}/queue/rotational"
                try:
                    with open(rotational_path) as f:
                        rotational = f.read().strip()
                    storage["type"] = "HDD" if rotational == "1" else "SSD"
                except Exception:
                    # Tenta via lsblk
                    lsblk_out = _run_cmd(["lsblk", "-d", "-o", "NAME,ROTA", device])
                    if lsblk_out:
                        for line in lsblk_out.splitlines()[1:]:
                            parts = line.split()
                            if len(parts) >= 2:
                                storage["type"] = "HDD" if parts[1] == "1" else "SSD"
    except Exception:
        pass

    return storage


def _detect_temperatures() -> dict:
    """
    Lê temperaturas de /sys/class/thermal/thermal_zone*/temp.
    Retorna média e máxima das zonas disponíveis.
    """
    temps = {
        "cpu_avg_c": None,
        "cpu_max_c": None,
        "zones": [],
    }

    try:
        thermal_base = "/sys/class/thermal"
        if os.path.exists(thermal_base):
            readings = []
            for zone in sorted(os.listdir(thermal_base)):
                if not zone.startswith("thermal_zone"):
                    continue
                temp_path = os.path.join(thermal_base, zone, "temp")
                type_path = os.path.join(thermal_base, zone, "type")
                try:
                    with open(temp_path) as f:
                        temp_c = int(f.read().strip()) / 1000.0
                    zone_type = "unknown"
                    try:
                        with open(type_path) as f:
                            zone_type = f.read().strip()
                    except Exception:
                        pass
                    if not any(marker in zone_type.lower() for marker in (
                            "cpu", "soc", "package", "x86_pkg", "coretemp", "k10temp")):
                        continue
                    readings.append(temp_c)
                    temps["zones"].append({"zone": zone, "type": zone_type, "temp_c": temp_c})
                except Exception:
                    continue

            if readings:
                temps["cpu_avg_c"] = round(sum(readings) / len(readings), 1)
                temps["cpu_max_c"] = round(max(readings), 1)
    except Exception:
        pass

    # Fallback: psutil sensors
    if temps["cpu_avg_c"] is None and _PSUTIL:
        try:
            sensors = psutil.sensors_temperatures()
            if sensors:
                all_temps = []
                for chip in ("coretemp", "k10temp", "cpu_thermal", "soc_thermal", "zenpower"):
                    entries = sensors.get(chip, [])
                    for entry in entries:
                        if entry.current and entry.current > 0:
                            all_temps.append(entry.current)
                if all_temps:
                    temps["cpu_avg_c"] = round(sum(all_temps) / len(all_temps), 1)
                    temps["cpu_max_c"] = round(max(all_temps), 1)
        except Exception:
            pass

    return temps


def _detect_power() -> dict:
    """
    Lê consumo de energia via RAPL (Running Average Power Limit).
    Interface: /sys/class/powercap/intel-rapl/*/energy_uj
    Faz duas leituras com intervalo para calcular watts.
    """
    power = {
        "package_watts": None,
        "dram_watts": None,
    }

    try:
        import time
        rapl_base = "/sys/class/powercap"
        if not os.path.exists(rapl_base):
            return power

        def read_energy(path: str) -> Optional[int]:
            """Lê energia em microjoules de um arquivo RAPL."""
            try:
                with open(path) as f:
                    return int(f.read().strip())
            except Exception:
                return None

        # Procura domínios RAPL
        package_paths = []
        dram_paths = []

        for entry in os.listdir(rapl_base):
            if not entry.startswith("intel-rapl"):
                continue
            base = os.path.join(rapl_base, entry)
            # Domínio raiz = package
            energy_path = os.path.join(base, "energy_uj")
            name_path = os.path.join(base, "name")
            if os.path.exists(energy_path):
                name = ""
                try:
                    with open(name_path) as f:
                        name = f.read().strip()
                except Exception:
                    pass
                if "package" in name or name == "":
                    package_paths.append(energy_path)
                elif "dram" in name:
                    dram_paths.append(energy_path)

            # Sub-domínios (DRAM, uncore, etc.)
            try:
                for sub in os.listdir(base):
                    sub_energy = os.path.join(base, sub, "energy_uj")
                    sub_name_p = os.path.join(base, sub, "name")
                    if os.path.exists(sub_energy):
                        sub_name = ""
                        try:
                            with open(sub_name_p) as f:
                                sub_name = f.read().strip()
                        except Exception:
                            pass
                        if "dram" in sub_name:
                            dram_paths.append(sub_energy)
            except Exception:
                pass

        if not package_paths:
            return power

        # Leitura 1
        e1_pkg = [read_energy(p) for p in package_paths]
        e1_dram = [read_energy(p) for p in dram_paths] if dram_paths else []
        t1 = time.time()

        time.sleep(0.5)

        # Leitura 2
        e2_pkg = [read_energy(p) for p in package_paths]
        e2_dram = [read_energy(p) for p in dram_paths] if dram_paths else []
        dt = time.time() - t1

        # Calcula watts (energia em uJ → W)
        valid_pkg = [(e2 - e1) for e1, e2 in zip(e1_pkg, e2_pkg)
                     if e1 is not None and e2 is not None and e2 >= e1]
        if valid_pkg:
            power["package_watts"] = round(sum(valid_pkg) / dt / 1e6, 2)

        valid_dram = [(e2 - e1) for e1, e2 in zip(e1_dram, e2_dram)
                      if e1 is not None and e2 is not None and e2 >= e1]
        if valid_dram:
            power["dram_watts"] = round(sum(valid_dram) / dt / 1e6, 2)

    except Exception:
        pass

    return power


def _read_sysfs(path: str) -> Optional[str]:
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


_NPU_DRIVER_MARKERS = ("ivpu", "xdna", "hailo", "apex", "rknpu", "ethos", "neuron", "npu")


def _pci_accelerator_category(class_code: str, driver: Optional[str]) -> Optional[str]:
    if driver and any(marker in driver.lower() for marker in _NPU_DRIVER_MARKERS):
        return "npu"
    if class_code.startswith("0x03"):
        return "gpu"
    if class_code.startswith("0x12"):
        return "accelerator"
    return None


def _detect_accelerators() -> dict:
    devices = []
    for base in sorted(glob.glob("/sys/bus/pci/devices/*")):
        class_code = (_read_sysfs(os.path.join(base, "class")) or "").lower()
        driver_path = os.path.join(base, "driver")
        driver = os.path.basename(os.path.realpath(driver_path)) \
            if os.path.exists(driver_path) else None
        category = _pci_accelerator_category(class_code, driver)
        if category is None:
            continue
        devices.append({
            "kind": "pci",
            "category": category,
            "address": os.path.basename(base),
            "vendor_id": _read_sysfs(os.path.join(base, "vendor")),
            "device_id": _read_sysfs(os.path.join(base, "device")),
            "driver": driver,
        })

    node_patterns = {
        "npu": (
            "/dev/accel/accel*", "/dev/apex_*", "/dev/hailo*",
            "/dev/rknpu*", "/dev/ethos*", "/dev/neuron*",
        ),
        "gpu": ("/dev/dri/renderD*",),
    }
    known_nodes = set()
    for category, patterns in node_patterns.items():
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                real_path = os.path.realpath(path)
                if real_path in known_nodes:
                    continue
                known_nodes.add(real_path)
                devices.append({
                    "kind": "device_node",
                    "category": category,
                    "path": path,
                    "driver": None,
                })
    return {"available": bool(devices), "devices": devices}


def _detect_serial() -> dict:
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    if by_id:
        devices = [{"path": path, "device": os.path.realpath(path)} for path in by_id]
    else:
        paths = sorted(set(
            glob.glob("/dev/ttyACM*") +
            glob.glob("/dev/ttyUSB*") +
            glob.glob("/dev/ttyAMA*")
        ))
        devices = [{"path": path, "device": path} for path in paths]
    return {"available": bool(devices), "devices": devices}


def hardware_fingerprint(capabilities: dict) -> str:
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    cpu = capabilities.get("cpu") if isinstance(capabilities.get("cpu"), dict) else {}
    ram = capabilities.get("ram") if isinstance(capabilities.get("ram"), dict) else {}
    gpu = capabilities.get("gpu") if isinstance(capabilities.get("gpu"), dict) else {}
    storage = capabilities.get("storage") \
        if isinstance(capabilities.get("storage"), dict) else {}
    accelerators = capabilities.get("accelerators")
    accelerators = accelerators if isinstance(accelerators, dict) else {}
    serial = capabilities.get("serial") if isinstance(capabilities.get("serial"), dict) else {}

    identity = {
        "cpu": {key: cpu.get(key) for key in (
            "model", "arch", "cores_physical", "cores_logical", "cache_kb")},
        "ram": {key: ram.get(key) for key in ("total_gb", "type", "speed_mhz")},
        "gpu": {key: gpu.get(key) for key in (
            "model", "vram_mb", "compute_capability")},
        "accelerators": [
            {key: device.get(key) for key in (
                "kind", "category", "address", "path", "vendor_id", "device_id", "driver")}
            for device in accelerators.get("devices", []) if isinstance(device, dict)
        ],
        "serial": [
            {key: device.get(key) for key in ("path", "device")}
            for device in serial.get("devices", []) if isinstance(device, dict)
        ],
        "storage": {key: storage.get(key) for key in ("type", "device")},
    }
    encoded = json.dumps(identity, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def save_hardware_snapshot(capabilities: dict, path: str) -> dict:
    if not isinstance(capabilities, dict) or not capabilities:
        raise ValueError("inventário de hardware inválido")
    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    payload = {
        "schema_version": HARDWARE_SNAPSHOT_SCHEMA,
        "captured_at": datetime.now().astimezone().isoformat(),
        "hardware_id": hardware_fingerprint(capabilities),
        "hardware": capabilities,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=".hardware-", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
        return payload
    except Exception:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def load_hardware_snapshot(path: Optional[str] = None) -> Optional[dict]:
    target = path or os.environ.get(
        "PDC_HOST_HARDWARE_PATH", DEFAULT_HARDWARE_SNAPSHOT_PATH)
    try:
        if os.path.getsize(target) > 10 * 1024 * 1024:
            return None
        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != HARDWARE_SNAPSHOT_SCHEMA:
        return None
    if not isinstance(payload.get("hardware"), dict) or not payload["hardware"]:
        return None
    return payload


def merge_host_capabilities(runtime_capabilities: dict,
                            snapshot: Optional[dict] = None) -> dict:
    runtime = runtime_capabilities if isinstance(runtime_capabilities, dict) else {}
    runtime_visibility = runtime.get("runtime_visibility")
    runtime_source = runtime_visibility \
        if isinstance(runtime_visibility, dict) and runtime_visibility else runtime
    snapshot = snapshot if snapshot is not None else load_hardware_snapshot()
    detected_at = datetime.now().astimezone().isoformat()
    if not snapshot:
        merged = dict(runtime)
        merged["provenance"] = {
            "inventory_source": "runtime_detection",
            "detected_at": detected_at,
            "hardware_id": hardware_fingerprint(runtime),
        }
        return merged

    host = snapshot["hardware"]
    merged = dict(host)
    runtime_gpu = runtime_source.get("gpu") \
        if isinstance(runtime_source.get("gpu"), dict) else {}
    host_gpu = merged.get("gpu") if isinstance(merged.get("gpu"), dict) else {}
    if runtime_gpu.get("available") and not host_gpu.get("available"):
        merged["gpu"] = runtime_gpu

    merged["runtime_visibility"] = {
        key: runtime_source.get(key)
        for key in ("gpu", "accelerators", "serial", "storage")
        if key in runtime_source
    }
    merged["provenance"] = {
        "inventory_source": "host_snapshot",
        "snapshot_schema_version": snapshot.get("schema_version"),
        "captured_at": snapshot.get("captured_at"),
        "detected_at": detected_at,
        "hardware_id": snapshot.get("hardware_id") or hardware_fingerprint(host),
    }
    return merged


class CapabilityDetector:
    """Detecta hardware dentro ou fora de containers."""

    def detect(self, use_host_snapshot: bool = True) -> dict:
        """Retorna as capacidades disponíveis sem exigir cada sensor."""
        detected = {
            "cpu": _detect_cpu(),
            "ram": _detect_ram(),
            "gpu": _detect_gpu(),
            "accelerators": _detect_accelerators(),
            "serial": _detect_serial(),
            "storage": _detect_storage(),
            "temperature": _detect_temperatures(),
            "power": _detect_power(),
        }
        return merge_host_capabilities(detected) if use_host_snapshot \
            else merge_host_capabilities(detected, snapshot={})
