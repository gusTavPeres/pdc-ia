"""
capabilities.py — Detecção de hardware do sistema.

Detecta CPU, RAM, GPU, storage, temperatura e consumo de energia.
Funciona dentro e fora de containers Docker.
Nunca falha: retorna null para tudo que não conseguir detectar.
"""

import platform
import subprocess
import os
import re
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
        # Pega modelo do primeiro processador
        for line in content.splitlines():
            if "model name" in line.lower() and cpu["model"] is None:
                cpu["model"] = line.split(":")[1].strip()
            if "flags" in line.lower() and ":" in line:
                flags_str = line.split(":")[1].strip().lower()
                flags = set(flags_str.split())
                cpu["flags"]["avx"] = "avx" in flags
                cpu["flags"]["avx2"] = "avx2" in flags
                cpu["flags"]["avx512"] = any(f.startswith("avx512") for f in flags)
                cpu["flags"]["vnni"] = "avx512_vnni" in flags or "avx_vnni" in flags
                cpu["flags"]["fma"] = "fma" in flags
                break  # Só precisa do primeiro core
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
    """
    Detecta GPU NVIDIA via pynvml (prioritário) ou nvidia-smi (fallback).
    Retorna null para tudo se não houver GPU NVIDIA.
    """
    gpu = {
        "available": False,
        "model": None,
        "vram_mb": None,
        "cuda_version": None,
        "compute_capability": None,
    }

    # Tentativa 1: pynvml (API direta NVIDIA)
    if _PYNVML:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu["available"] = True
                gpu["model"] = pynvml.nvmlDeviceGetName(handle)
                # pynvml pode retornar bytes em versões antigas
                if isinstance(gpu["model"], bytes):
                    gpu["model"] = gpu["model"].decode()
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu["vram_mb"] = round(mem_info.total / (1024**2))
                # Versão CUDA do driver
                cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion()
                major = cuda_ver // 1000
                minor = (cuda_ver % 1000) // 10
                gpu["cuda_version"] = f"{major}.{minor}"
                # Compute capability
                cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                gpu["compute_capability"] = f"{cc[0]}.{cc[1]}"
            pynvml.nvmlShutdown()
            return gpu
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # Tentativa 2: parsear saída do nvidia-smi
    try:
        out = _run_cmd(["nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits"])
        if out:
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 2:
                gpu["available"] = True
                gpu["model"] = parts[0]
                try:
                    gpu["vram_mb"] = int(parts[1])
                except Exception:
                    pass
    except Exception:
        pass

    # Versão CUDA via nvcc ou nvidia-smi
    try:
        nvcc_out = _run_cmd(["nvcc", "--version"])
        if nvcc_out:
            match = re.search(r"release ([\d.]+)", nvcc_out)
            if match:
                gpu["cuda_version"] = match.group(1)
    except Exception:
        pass

    return gpu


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

            # Remove números do final (ex: /dev/sda1 → sda)
            base_dev = re.sub(r'\d+$', '', device.split("/")[-1])
            # Remove 'p' antes dos números em NVMe (ex: nvme0n1p1 → nvme0n1)
            base_dev = re.sub(r'p\d+$', '', base_dev)

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
                for chip, entries in sensors.items():
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


class CapabilityDetector:
    """
    Detecta capacidades de hardware do sistema.

    Funciona dentro e fora de containers Docker.
    Nunca levanta exceção — retorna None para tudo que não conseguir detectar.

    Exemplo de uso:
        detector = CapabilityDetector()
        caps = detector.detect()
        print(caps["cpu"]["model"])
        print(caps["gpu"]["vram_mb"])
    """

    def detect(self) -> dict:
        """
        Executa todas as detecções e retorna dicionário completo.

        Retorna:
            dict com chaves: cpu, ram, gpu, storage, temperature, power
        """
        return {
            "cpu": _detect_cpu(),
            "ram": _detect_ram(),
            "gpu": _detect_gpu(),
            "storage": _detect_storage(),
            "temperature": _detect_temperatures(),
            "power": _detect_power(),
        }
