"""
metrics_collector.py — Coleta de métricas do sistema em thread separada.

Coleta CPU, RAM, GPU e temperatura continuamente em background.
"""

import time
import threading
from typing import List, Optional

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


class MetricsCollector:
    """
    Coleta métricas do sistema em thread separada.

    Usa psutil (CPU, RAM), pynvml (GPU), sysfs (temperatura).
    Thread-safe: dados coletados são armazenados internamente e
    podem ser lidos a qualquer momento com get_data().

    Exemplo de uso:
        collector = MetricsCollector(interval_sec=1)
        collector.start()

        # ... executar inferências ...

        collector.stop()
        timeseries = collector.get_data()
        print(f"Pico de RAM: {max(s['ram_mb'] for s in timeseries)} MB")
    """

    def __init__(self, interval_sec: float = 1.0):
        """
        Inicializa o coletor.

        Args:
            interval_sec: Intervalo entre coletas em segundos.
        """
        self._interval = interval_sec
        self._data: List[dict] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

        # Inicializa pynvml uma vez
        self._nvml_available = False
        if _PYNVML:
            try:
                pynvml.nvmlInit()
                pynvml.nvmlDeviceGetCount()  # Verifica se há GPUs
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_available = True
            except Exception:
                self._nvml_available = False

    def start(self):
        """Inicia a coleta de métricas em background."""
        if self._thread is not None and self._thread.is_alive():
            return  # Já está rodando

        self._stop_event.clear()
        self._data.clear()
        self._start_time = time.time()

        self._thread = threading.Thread(
            target=self._collect_loop,
            name="MetricsCollector",
            daemon=True,  # Thread daemon: encerra quando o programa principal encerrar
        )
        self._thread.start()

    def stop(self):
        """Para a coleta de métricas e aguarda a thread encerrar."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 3)

        # Encerra pynvml
        if self._nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_available = False

    def get_data(self) -> List[dict]:
        """
        Retorna cópia thread-safe dos dados coletados.

        Retorna:
            Lista de dicts com: timestamp, ram_mb, cpu_pct, gpu_pct,
            gpu_mem_mb, temp_c
        """
        with self._lock:
            return list(self._data)

    def get_summary(self) -> dict:
        """
        Retorna sumário estatístico dos dados coletados até agora.

        Retorna:
            dict com: peak_ram_mb, avg_cpu_pct, avg_gpu_pct, peak_temp_c,
            samples_count, duration_sec
        """
        data = self.get_data()
        if not data:
            return {}

        ram_values = [s["ram_mb"] for s in data if s.get("ram_mb") is not None]
        cpu_values = [s["cpu_pct"] for s in data if s.get("cpu_pct") is not None]
        gpu_values = [s["gpu_pct"] for s in data if s.get("gpu_pct") is not None]
        temp_values = [s["temp_c"] for s in data if s.get("temp_c") is not None]

        return {
            "peak_ram_mb": round(max(ram_values), 1) if ram_values else None,
            "avg_cpu_pct": round(sum(cpu_values) / len(cpu_values), 1) if cpu_values else None,
            "avg_gpu_pct": round(sum(gpu_values) / len(gpu_values), 1) if gpu_values else None,
            "peak_temp_c": round(max(temp_values), 1) if temp_values else None,
            "samples_count": len(data),
            "duration_sec": round(data[-1]["timestamp"] - data[0]["timestamp"], 2) if len(data) > 1 else 0,
        }

    def _collect_loop(self):
        """Loop interno da thread de coleta."""
        while not self._stop_event.is_set():
            snapshot = self._collect_snapshot()
            with self._lock:
                self._data.append(snapshot)
            self._stop_event.wait(timeout=self._interval)

    def _collect_snapshot(self) -> dict:
        """Coleta uma snapshot das métricas atuais."""
        snapshot = {
            "timestamp": round(time.time() - self._start_time, 2),
            "ram_mb": None,
            "cpu_pct": None,
            "gpu_pct": None,
            "gpu_mem_mb": None,
            "temp_c": None,
        }

        # CPU e RAM via psutil
        if _PSUTIL:
            try:
                mem = psutil.virtual_memory()
                snapshot["ram_mb"] = round((mem.total - mem.available) / (1024**2), 1)
                # cpu_percent com interval=None usa medição desde a última chamada
                snapshot["cpu_pct"] = psutil.cpu_percent(interval=None)
            except Exception:
                pass

        # GPU via pynvml (se disponível)
        if self._nvml_available:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                snapshot["gpu_pct"] = util.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                snapshot["gpu_mem_mb"] = round(mem_info.used / (1024**2), 1)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                snapshot["temp_c"] = temp
            except Exception:
                pass

        # Temperatura via sysfs (fallback se não tiver GPU)
        if snapshot["temp_c"] is None:
            try:
                import os
                thermal_base = "/sys/class/thermal"
                readings = []
                if os.path.exists(thermal_base):
                    for zone in os.listdir(thermal_base):
                        if not zone.startswith("thermal_zone"):
                            continue
                        try:
                            with open(f"{thermal_base}/{zone}/temp") as f:
                                readings.append(int(f.read().strip()) / 1000.0)
                        except Exception:
                            continue
                if readings:
                    snapshot["temp_c"] = round(max(readings), 1)
            except Exception:
                pass

        return snapshot
