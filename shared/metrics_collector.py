"""Coleta portátil de recursos durante benchmarks."""

import glob
import math
import os
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple

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


RESOURCE_SCHEMA_VERSION = "resources-2026.07-v2"
_MIB = 1024 ** 2


def _finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _time_weighted_average(samples: List[dict], key: str) -> Optional[float]:
    values = []
    for sample in samples:
        value = _finite_number(sample.get(key))
        if value is not None:
            values.append(value)
    if not values:
        return None

    area = 0.0
    observed = 0.0
    for previous, current in zip(samples, samples[1:]):
        t0 = _finite_number(previous.get("timestamp"))
        t1 = _finite_number(current.get("timestamp"))
        v0 = _finite_number(previous.get(key))
        v1 = _finite_number(current.get(key))
        if None in (t0, t1, v0, v1):
            continue
        delta = t1 - t0
        if delta <= 0:
            continue
        area += (v0 + v1) * 0.5 * delta
        observed += delta
    return area / observed if observed > 0 else sum(values) / len(values)


def _integrate_metric(samples: List[dict], key: str) -> Tuple[Optional[float], float]:
    total = 0.0
    observed = 0.0
    for previous, current in zip(samples, samples[1:]):
        t0 = _finite_number(previous.get("timestamp"))
        t1 = _finite_number(current.get("timestamp"))
        v0 = _finite_number(previous.get(key))
        v1 = _finite_number(current.get(key))
        if None in (t0, t1, v0, v1):
            continue
        delta = t1 - t0
        if delta <= 0:
            continue
        total += (v0 + v1) * 0.5 * delta
        observed += delta
    return (total if observed > 0 else None), observed


def _interval_weighted_average(samples: List[dict], key: str) -> Optional[float]:
    total = 0.0
    observed = 0.0
    fallback = []
    for sample in samples:
        value = _finite_number(sample.get(key))
        if value is not None:
            fallback.append(value)
    for previous, current in zip(samples, samples[1:]):
        t0 = _finite_number(previous.get("timestamp"))
        t1 = _finite_number(current.get("timestamp"))
        value = _finite_number(current.get(key))
        if None in (t0, t1, value):
            continue
        delta = t1 - t0
        if delta <= 0:
            continue
        total += value * delta
        observed += delta
    if observed > 0:
        return total / observed
    return sum(fallback) / len(fallback) if fallback else None


def _gpu_metric_scope(samples: List[dict], value_key: str, scope_key: str) -> str:
    scopes = {
        sample.get(scope_key)
        for sample in samples
        if _finite_number(sample.get(value_key)) is not None
        and sample.get(scope_key) not in (None, "", "unavailable")
    }
    if not scopes:
        return "unavailable"
    if scopes == {"workload_processes"}:
        return "workload_processes"
    return "visible_devices"


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path) as handle:
            return handle.read().strip()
    except (OSError, ValueError):
        return None


def _is_containerized() -> bool:
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    cgroup = (_read_text("/proc/self/cgroup") or "").lower()
    return any(marker in cgroup for marker in ("docker", "containerd", "kubepods", "libpod"))


def _parse_key_values(path: str) -> Dict[str, int]:
    values = {}
    text = _read_text(path)
    if not text:
        return values
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


class _CgroupReader:
    def __init__(self, mode: str, memory_path: str, cpu_path: str):
        self.mode = mode
        self.memory_path = memory_path
        self.cpu_path = cpu_path
        self._last_cpu_usage = None
        self._last_cpu_time = None
        self._start_cpu_usage = None
        self._start_cpu_time = None
        self._latest_cpu_usage = None
        self._latest_cpu_time = None
        self._skip_cpu_sample = True

    @classmethod
    def discover(cls):
        entries = []
        text = _read_text("/proc/self/cgroup") or ""
        for line in text.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3:
                entries.append((parts[1], parts[2]))

        for controllers, relative in entries:
            if controllers:
                continue
            if _is_containerized() and relative.strip("/") == "":
                reader = cls._discover_host_mounted_v2()
                if reader is not None:
                    return reader
            base = os.path.join("/sys/fs/cgroup", relative.lstrip("/"))
            if os.path.isfile(os.path.join(base, "memory.current")):
                return cls("v2", base, base)

        reader = cls._discover_host_mounted_v2()
        if reader is not None:
            return reader

        controller_paths = {}
        for controllers, relative in entries:
            for controller in controllers.split(","):
                controller_paths[controller] = relative
        memory_relative = controller_paths.get("memory")
        cpu_relative = controller_paths.get("cpuacct") or controller_paths.get("cpu")
        if memory_relative and cpu_relative:
            if (_is_containerized() and memory_relative.strip("/") == ""
                    and cpu_relative.strip("/") == ""):
                reader = cls._discover_host_mounted_v1()
                if reader is not None:
                    return reader
            memory_path = os.path.join(
                "/sys/fs/cgroup/memory", memory_relative.lstrip("/"))
            cpu_bases = ("cpu,cpuacct", "cpuacct,cpu", "cpuacct", "cpu")
            for controller in cpu_bases:
                cpu_path = os.path.join(
                    "/sys/fs/cgroup", controller, cpu_relative.lstrip("/"))
                if os.path.isfile(os.path.join(cpu_path, "cpuacct.usage")):
                    return cls("v1", memory_path, cpu_path)
        return None

    @classmethod
    def _discover_host_mounted_v2(cls):
        hostname = socket.gethostname().split(".", 1)[0]
        candidates = [
            f"/sys/fs/cgroup/system.slice/docker-{hostname}*.scope",
            f"/sys/fs/cgroup/docker/{hostname}*",
            f"/sys/fs/cgroup/system.slice/cri-containerd-{hostname}*.scope",
        ]
        for pattern in candidates:
            for base in glob.glob(pattern):
                if os.path.isfile(os.path.join(base, "memory.current")):
                    return cls("v2", base, base)
        return None

    @classmethod
    def _discover_host_mounted_v1(cls):
        hostname = socket.gethostname().split(".", 1)[0]
        memory_patterns = (
            f"/sys/fs/cgroup/memory/system.slice/docker-{hostname}*.scope",
            f"/sys/fs/cgroup/memory/docker/{hostname}*",
            f"/sys/fs/cgroup/memory/{hostname}*",
        )
        cpu_patterns = []
        for controller in ("cpu,cpuacct", "cpuacct,cpu", "cpuacct", "cpu"):
            cpu_patterns.extend((
                f"/sys/fs/cgroup/{controller}/system.slice/docker-{hostname}*.scope",
                f"/sys/fs/cgroup/{controller}/docker/{hostname}*",
                f"/sys/fs/cgroup/{controller}/{hostname}*",
            ))
        memory_paths = [path for pattern in memory_patterns for path in glob.glob(pattern)]
        cpu_paths = [path for pattern in cpu_patterns for path in glob.glob(pattern)]
        for memory_path in memory_paths:
            if not os.path.isfile(os.path.join(memory_path, "memory.usage_in_bytes")):
                continue
            for cpu_path in cpu_paths:
                if os.path.isfile(os.path.join(cpu_path, "cpuacct.usage")):
                    return cls("v1", memory_path, cpu_path)
        return None

    def reset(self, now: float):
        self._last_cpu_usage = self._read_cpu_usage()
        self._last_cpu_time = now
        self._start_cpu_usage = self._last_cpu_usage
        self._start_cpu_time = now
        self._latest_cpu_usage = self._last_cpu_usage
        self._latest_cpu_time = now
        self._skip_cpu_sample = True

    def memory(self) -> dict:
        if self.mode == "v2":
            current = _finite_number(_read_text(os.path.join(self.memory_path, "memory.current")))
            stat = _parse_key_values(os.path.join(self.memory_path, "memory.stat"))
            inactive_file = float(stat.get("inactive_file", 0))
            limit_raw = _read_text(os.path.join(self.memory_path, "memory.max"))
        else:
            current = _finite_number(_read_text(
                os.path.join(self.memory_path, "memory.usage_in_bytes")))
            stat = _parse_key_values(os.path.join(self.memory_path, "memory.stat"))
            inactive_file = float(stat.get("total_inactive_file", stat.get("inactive_file", 0)))
            limit_raw = _read_text(os.path.join(
                self.memory_path, "memory.limit_in_bytes"))

        if current is None:
            return {}
        working_set = max(0.0, current - inactive_file)
        limit = _finite_number(limit_raw) if limit_raw != "max" else None
        return {
            "working_set_mb": working_set / _MIB,
            "total_mb": current / _MIB,
            "cache_mb": inactive_file / _MIB,
            "limit_mb": limit / _MIB if limit is not None and limit < 2 ** 60 else None,
        }

    def _read_cpu_usage(self) -> Optional[float]:
        if self.mode == "v2":
            stat = _parse_key_values(os.path.join(self.cpu_path, "cpu.stat"))
            usage_usec = stat.get("usage_usec")
            return usage_usec / 1_000_000.0 if usage_usec is not None else None
        usage_ns = _finite_number(_read_text(os.path.join(self.cpu_path, "cpuacct.usage")))
        return usage_ns / 1_000_000_000.0 if usage_ns is not None else None

    def cpu_percent(self, now: float) -> Optional[float]:
        usage = self._read_cpu_usage()
        previous_usage = self._last_cpu_usage
        previous_time = self._last_cpu_time
        self._last_cpu_usage = usage
        self._last_cpu_time = now
        if usage is not None:
            self._latest_cpu_usage = usage
            self._latest_cpu_time = now
        if self._skip_cpu_sample:
            self._skip_cpu_sample = False
            return None
        if usage is None or previous_usage is None or previous_time is None:
            return None
        elapsed = now - previous_time
        used = usage - previous_usage
        if elapsed <= 0 or used < 0:
            return None
        return used / elapsed * 100.0

    def cpu_time_delta(self) -> Optional[float]:
        if self._start_cpu_usage is None or self._latest_cpu_usage is None:
            return None
        delta = self._latest_cpu_usage - self._start_cpu_usage
        return delta if delta >= 0 else None

    def average_cpu_percent(self) -> Optional[float]:
        cpu_time = self.cpu_time_delta()
        if cpu_time is None or self._start_cpu_time is None or self._latest_cpu_time is None:
            return None
        elapsed = self._latest_cpu_time - self._start_cpu_time
        return cpu_time / elapsed * 100.0 if elapsed > 0 else None

    def pids(self) -> List[int]:
        path = os.path.join(
            self.cpu_path,
            "cgroup.procs" if self.mode == "v2" else "tasks",
        )
        text = _read_text(path) or ""
        return [int(value) for value in text.split() if value.isdigit()]

    def limits(self) -> dict:
        cpu_limit = None
        pids_limit = None
        if self.mode == "v2":
            cpu_max = (_read_text(os.path.join(self.cpu_path, "cpu.max")) or "").split()
            if len(cpu_max) == 2 and cpu_max[0] != "max":
                quota = _finite_number(cpu_max[0])
                period = _finite_number(cpu_max[1])
                if quota is not None and period and period > 0:
                    cpu_limit = quota / period
            pids_raw = _read_text(os.path.join(self.cpu_path, "pids.max"))
            pids_limit = _finite_number(pids_raw) if pids_raw != "max" else None
        else:
            quota = _finite_number(_read_text(os.path.join(
                self.cpu_path, "cpu.cfs_quota_us")))
            period = _finite_number(_read_text(os.path.join(
                self.cpu_path, "cpu.cfs_period_us")))
            if quota is not None and quota > 0 and period and period > 0:
                cpu_limit = quota / period
        memory = self.memory()
        return {
            "cpu_limit_cores": round(cpu_limit, 4) if cpu_limit is not None else None,
            "memory_limit_mb": round(memory["limit_mb"], 3)
            if memory.get("limit_mb") is not None else None,
            "pids_limit": int(pids_limit) if pids_limit is not None else None,
        }


def runtime_execution_info() -> dict:
    cgroup = _CgroupReader.discover()
    limits = cgroup.limits() if cgroup else {
        "cpu_limit_cores": None,
        "memory_limit_mb": None,
        "pids_limit": None,
    }
    try:
        cpu_affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_affinity = os.cpu_count()
    try:
        shm = os.statvfs("/dev/shm")
        shm_size_mb = shm.f_frsize * shm.f_blocks / _MIB
    except OSError:
        shm_size_mb = None
    return {
        "architecture": os.uname().machine,
        "containerized": _is_containerized(),
        "cpu_affinity_count": cpu_affinity,
        "cpu_limit_cores": limits["cpu_limit_cores"],
        "memory_limit_mb": limits["memory_limit_mb"],
        "pids_limit": limits["pids_limit"],
        "shm_size_mb": round(shm_size_mb, 3) if shm_size_mb is not None else None,
        "gpu_mode": os.environ.get("PDC_GPU_MODE", "auto"),
        "image_variant": os.environ.get("PDC_IMAGE_VARIANT", "unknown"),
    }


class MetricsCollector:
    """Amostra recursos da carga, do container e do host em uma thread."""

    def __init__(self, interval_sec: float = 1.0):
        self._interval = max(0.05, float(interval_sec))
        self._data: List[dict] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._processes = {}
        self._skip_process_cpu_sample = True
        self._last_process_refresh = 0.0
        self._cgroup = _CgroupReader.discover()
        self._containerized = _is_containerized()
        self._nvml_available = False
        self._nvml_handles = []
        self._nvml_process_timestamps = {}
        self._rapl = self._discover_rapl()
        self._rapl_last_time = None
        self._rapl_observed_duration = 0.0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self._data.clear()
        self._start_time = time.perf_counter()
        self._processes.clear()
        self._last_process_refresh = 0.0
        self._prime_process_cpu()
        self._skip_process_cpu_sample = True
        if self._cgroup:
            self._cgroup.reset(self._start_time)
        self._prime_rapl(self._start_time)
        self._init_nvml()
        if _PSUTIL:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        baseline = self._collect_snapshot()
        with self._lock:
            self._data.append(baseline)

        self._thread = threading.Thread(
            target=self._collect_loop,
            name="MetricsCollector",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._interval * 3))

        if self._start_time is not None:
            final = self._collect_snapshot()
            with self._lock:
                last_timestamp = self._data[-1]["timestamp"] if self._data else -1.0
                if final["timestamp"] - last_timestamp > 0.001:
                    self._data.append(final)

        if self._nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml_available = False
        self._nvml_handles = []
        self._nvml_process_timestamps = {}

    def get_data(self) -> List[dict]:
        with self._lock:
            return list(self._data)

    def get_summary(self) -> dict:
        data = self.get_data()
        if not data:
            return {}

        def values(key):
            return [number for sample in data
                    if (number := _finite_number(sample.get(key))) is not None]

        def peak(key):
            items = values(key)
            return max(items) if items else None

        def rounded(value, digits=3):
            return round(value, digits) if value is not None else None

        duration = max(0.0, data[-1]["timestamp"] - data[0]["timestamp"])
        exact_container_cpu = (
            self._cgroup.average_cpu_percent()
            if self._cgroup and self._containerized else None
        )
        runtime_limits = self._cgroup.limits() if self._cgroup else {}
        sampled_container_cpu = _interval_weighted_average(data, "container_cpu_pct")
        process_cpu_average = _interval_weighted_average(data, "process_cpu_pct")
        canonical_cpu_average = exact_container_cpu
        canonical_cpu_scope = "container_cgroup"
        if canonical_cpu_average is None:
            canonical_cpu_average = sampled_container_cpu
        if canonical_cpu_average is None:
            canonical_cpu_average = process_cpu_average
            canonical_cpu_scope = "process_tree" if process_cpu_average is not None else "unavailable"
        canonical_ram_scope = next(
            (sample.get("ram_scope") for sample in data if sample.get("ram_scope")),
            "unavailable",
        )
        gpu_mem_scope = _gpu_metric_scope(data, "gpu_mem_mb", "gpu_mem_scope")
        gpu_util_scope = _gpu_metric_scope(data, "gpu_pct", "gpu_util_scope")

        gpu_energy, gpu_energy_duration = _integrate_metric(data, "gpu_power_w")
        rapl_energy_values = values("cpu_package_energy_j")
        cpu_energy = max(rapl_energy_values) if rapl_energy_values else None
        rapl_duration_values = values("cpu_package_energy_duration_sec")
        cpu_energy_duration = max(rapl_duration_values) if rapl_duration_values else 0.0
        energy_parts = [value for value in (gpu_energy, cpu_energy) if value is not None]
        total_energy = sum(energy_parts) if energy_parts else None
        average_power_parts = []
        if gpu_energy is not None and gpu_energy_duration > 0:
            average_power_parts.append(gpu_energy / gpu_energy_duration)
        if cpu_energy is not None and cpu_energy_duration > 0:
            average_power_parts.append(cpu_energy / cpu_energy_duration)
        avg_power = sum(average_power_parts) if average_power_parts else None

        first = data[0]
        peak_ram = peak("ram_mb")
        peak_gpu_mem = peak("gpu_mem_mb")
        baseline_ram = _finite_number(first.get("ram_mb"))
        baseline_gpu_mem = _finite_number(first.get("gpu_mem_mb"))
        power_sources = []
        if gpu_energy is not None:
            power_sources.append("nvml_gpu")
        if cpu_energy is not None:
            power_sources.append("rapl_cpu_package")
        observed_durations = []
        if gpu_energy is not None:
            observed_durations.append(gpu_energy_duration)
        if cpu_energy is not None:
            observed_durations.append(cpu_energy_duration)
        energy_observed_duration = min(observed_durations) if observed_durations else 0.0
        energy_coverage = (
            min(1.0, energy_observed_duration / duration)
            if duration > 0 and observed_durations else None
        )

        return {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "timeseries": data,
            "scopes": {
                "ram_mb": canonical_ram_scope,
                "cpu_pct": canonical_cpu_scope,
                "gpu_pct": gpu_util_scope,
                "gpu_mem_mb": gpu_mem_scope,
                "power_w": "device_components"
                if total_energy is not None or avg_power is not None else "unavailable",
            },
            "peak_ram_mb": rounded(peak_ram, 1),
            "peak_container_ram_mb": rounded(peak("container_ram_mb"), 1),
            "peak_container_memory_total_mb": rounded(peak("container_memory_total_mb"), 1),
            "peak_process_ram_mb": rounded(peak("process_ram_mb"), 1),
            "peak_host_ram_mb": rounded(peak("host_ram_used_mb"), 1),
            "baseline_ram_mb": rounded(baseline_ram, 1),
            "peak_ram_delta_mb": rounded(
                max(0.0, peak_ram - baseline_ram)
                if peak_ram is not None and baseline_ram is not None else None,
                1,
            ),
            "avg_cpu_pct": rounded(canonical_cpu_average, 1),
            "avg_container_cpu_pct": rounded(
                exact_container_cpu if exact_container_cpu is not None
                else sampled_container_cpu, 1),
            "avg_process_cpu_pct": rounded(process_cpu_average, 1),
            "avg_host_cpu_pct": rounded(
                _interval_weighted_average(data, "host_cpu_pct"), 1),
            "container_cpu_time_sec": rounded(
                self._cgroup.cpu_time_delta()
                if self._cgroup and self._containerized else None, 6),
            "container_cpu_limit_cores": runtime_limits.get("cpu_limit_cores"),
            "container_memory_limit_mb": runtime_limits.get("memory_limit_mb"),
            "container_pids_limit": runtime_limits.get("pids_limit"),
            "peak_gpu_util": rounded(peak("gpu_pct"), 1),
            "avg_gpu_pct": rounded(_time_weighted_average(data, "gpu_pct"), 1),
            "peak_process_gpu_util": rounded(peak("gpu_process_pct"), 1),
            "peak_device_gpu_util": rounded(peak("gpu_device_pct"), 1),
            "peak_vram_mb": rounded(peak_gpu_mem, 1),
            "peak_gpu_mem_mb": rounded(peak_gpu_mem, 1),
            "peak_process_vram_mb": rounded(peak("gpu_process_mem_mb"), 1),
            "peak_device_vram_mb": rounded(peak("gpu_device_mem_mb"), 1),
            "baseline_vram_mb": rounded(baseline_gpu_mem, 1),
            "peak_vram_delta_mb": rounded(
                max(0.0, peak_gpu_mem - baseline_gpu_mem)
                if peak_gpu_mem is not None and baseline_gpu_mem is not None else None,
                1,
            ),
            "peak_temp_c": rounded(peak("temp_c"), 1),
            "peak_gpu_temp_c": rounded(peak("gpu_temp_c"), 1),
            "peak_cpu_temp_c": rounded(peak("cpu_temp_c"), 1),
            "avg_power_w": rounded(avg_power, 3),
            "avg_gpu_power_w": rounded(_time_weighted_average(data, "gpu_power_w"), 3),
            "peak_power_w": rounded(peak("power_w"), 3),
            "total_energy_j": rounded(total_energy, 3),
            "gpu_energy_j": rounded(gpu_energy, 3),
            "cpu_package_energy_j": rounded(cpu_energy, 3),
            "gpu_energy_observed_duration_sec": rounded(gpu_energy_duration, 3),
            "cpu_energy_observed_duration_sec": rounded(cpu_energy_duration, 3),
            "energy_observed_duration_sec": rounded(energy_observed_duration, 3),
            "energy_coverage_ratio": rounded(energy_coverage, 4),
            "power_sources": power_sources,
            "samples_collected": len(data),
            "measurement_duration_sec": rounded(duration, 3),
            "duration_sec": rounded(duration, 3),
            "telemetry": {
                "cgroup": self._cgroup is not None,
                "process": _PSUTIL,
                "nvml": any(sample.get("gpus") for sample in data),
                "rapl": bool(self._rapl),
            },
        }

    def _collect_loop(self):
        next_sample = time.perf_counter() + self._interval
        while True:
            if self._stop_event.wait(max(0.0, next_sample - time.perf_counter())):
                break
            snapshot = self._collect_snapshot()
            with self._lock:
                self._data.append(snapshot)
            next_sample += self._interval
            now = time.perf_counter()
            if next_sample <= now:
                next_sample = now + self._interval

    def _prime_process_cpu(self):
        if not _PSUTIL:
            return
        self._process_snapshot(force_refresh=True)

    def _process_snapshot(self, force_refresh: bool = False) -> dict:
        if not _PSUTIL:
            return {}
        now = time.perf_counter()
        if force_refresh or not self._processes or now - self._last_process_refresh >= 5.0:
            try:
                root = psutil.Process(os.getpid())
                processes = [root] + root.children(recursive=True)
                self._last_process_refresh = now
            except Exception:
                processes = list(self._processes.values())
        else:
            processes = list(self._processes.values())

        active = {}
        rss = 0
        cpu = 0.0
        visible_pids = []
        for process in processes:
            pid = process.pid
            tracked = self._processes.get(pid, process)
            try:
                rss += tracked.memory_info().rss
                cpu += tracked.cpu_percent(interval=None)
                active[pid] = tracked
                visible_pids.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
        self._processes = active
        if self._skip_process_cpu_sample:
            cpu = None
            self._skip_process_cpu_sample = False
        return {
            "ram_mb": rss / _MIB,
            "cpu_pct": cpu,
            "visible_pids": visible_pids,
        }

    def _collect_snapshot(self) -> dict:
        now = time.perf_counter()
        process = self._process_snapshot()
        cgroup_memory = self._cgroup.memory() if self._cgroup else {}
        cgroup_cpu = self._cgroup.cpu_percent(now) if self._cgroup else None

        host_ram = None
        host_cpu = None
        if _PSUTIL:
            try:
                memory = psutil.virtual_memory()
                host_ram = (memory.total - memory.available) / _MIB
                host_cpu = psutil.cpu_percent(interval=None)
            except Exception:
                pass

        cgroup_ram = _finite_number(cgroup_memory.get("working_set_mb"))
        container_ram = cgroup_ram if self._containerized else None
        process_ram = _finite_number(process.get("ram_mb"))
        ram = container_ram if container_ram is not None else process_ram
        ram_scope = "container_cgroup_working_set" if container_ram is not None else "process_tree_rss"
        process_cpu = _finite_number(process.get("cpu_pct"))
        container_cpu = cgroup_cpu if self._containerized else None
        cpu = container_cpu if container_cpu is not None else process_cpu

        gpu = self._collect_gpu(process.get("visible_pids", []))
        cpu_temp = self._cpu_temperature()
        rapl_power, rapl_energy, rapl_duration = self._collect_rapl(now)
        gpu_power = _finite_number(gpu.get("gpu_power_w"))
        power_parts = [value for value in (gpu_power, rapl_power) if value is not None]
        power = sum(power_parts) if power_parts else None
        gpu_temp = _finite_number(gpu.get("gpu_temp_c"))
        temperatures = [value for value in (gpu_temp, cpu_temp) if value is not None]

        return {
            "timestamp": round(now - self._start_time, 4),
            "ram_mb": round(ram, 3) if ram is not None else None,
            "ram_scope": ram_scope if ram is not None else "unavailable",
            "container_ram_mb": round(container_ram, 3) if container_ram is not None else None,
            "cgroup_ram_mb": round(cgroup_ram, 3) if cgroup_ram is not None else None,
            "container_memory_total_mb": round(cgroup_memory["total_mb"], 3)
            if cgroup_memory.get("total_mb") is not None else None,
            "container_cache_mb": round(cgroup_memory["cache_mb"], 3)
            if cgroup_memory.get("cache_mb") is not None else None,
            "container_memory_limit_mb": round(cgroup_memory["limit_mb"], 3)
            if cgroup_memory.get("limit_mb") is not None else None,
            "process_ram_mb": round(process_ram, 3) if process_ram is not None else None,
            "host_ram_used_mb": round(host_ram, 3) if host_ram is not None else None,
            "cpu_pct": round(cpu, 3) if cpu is not None else None,
            "container_cpu_pct": round(container_cpu, 3) if container_cpu is not None else None,
            "cgroup_cpu_pct": round(cgroup_cpu, 3) if cgroup_cpu is not None else None,
            "process_cpu_pct": round(process_cpu, 3) if process_cpu is not None else None,
            "host_cpu_pct": round(host_cpu, 3) if host_cpu is not None else None,
            "gpu_pct": gpu.get("gpu_pct"),
            "gpu_avg_pct": gpu.get("gpu_avg_pct"),
            "gpu_util_scope": gpu.get("gpu_util_scope"),
            "gpu_process_pct": gpu.get("gpu_process_pct"),
            "gpu_device_pct": gpu.get("gpu_device_pct"),
            "gpu_mem_mb": gpu.get("gpu_mem_mb"),
            "gpu_mem_scope": gpu.get("gpu_mem_scope"),
            "gpu_process_mem_mb": gpu.get("gpu_process_mem_mb"),
            "gpu_device_mem_mb": gpu.get("gpu_device_mem_mb"),
            "gpu_temp_c": gpu_temp,
            "cpu_temp_c": cpu_temp,
            "temp_c": max(temperatures) if temperatures else None,
            "gpu_power_w": gpu_power,
            "cpu_package_power_w": round(rapl_power, 3) if rapl_power is not None else None,
            "cpu_package_energy_j": round(rapl_energy, 6) if rapl_energy is not None else None,
            "cpu_package_energy_duration_sec": round(rapl_duration, 6)
            if rapl_duration is not None else None,
            "power_w": round(power, 3) if power is not None else None,
            "gpus": gpu.get("gpus", []),
        }

    def _init_nvml(self):
        self._nvml_available = False
        self._nvml_handles = []
        self._nvml_process_timestamps = {}
        if not _PYNVML:
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)
            ]
            self._nvml_available = bool(self._nvml_handles)
            for index, handle in enumerate(self._nvml_handles):
                self._nvml_process_utilization(handle, index, set())
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handles = []

    @staticmethod
    def _host_pid(pid: int) -> int:
        text = _read_text(f"/proc/{pid}/status") or ""
        for line in text.splitlines():
            if line.startswith("NSpid:"):
                values = line.split()[1:]
                if values:
                    try:
                        return int(values[0])
                    except ValueError:
                        break
        return pid

    def _nvml_process_memory(self, handle, host_pids: set) -> Tuple[Optional[float], bool]:
        process_entries = {}
        supported = False
        function_names = (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        )
        for name in function_names:
            function = getattr(pynvml, name, None)
            if function is None:
                continue
            try:
                entries = function(handle)
                supported = True
            except Exception:
                continue
            for entry in entries:
                pid = getattr(entry, "pid", None)
                used = _finite_number(getattr(entry, "usedGpuMemory", None))
                if pid in host_pids and used is not None and 0 <= used < 2 ** 60:
                    process_entries[pid] = max(process_entries.get(pid, 0.0), used)
        if not supported:
            return None, False
        return sum(process_entries.values()) / _MIB, True

    def _nvml_process_utilization(
            self, handle, device_index: int, host_pids: set) -> Tuple[Optional[float], bool]:
        timestamp = self._nvml_process_timestamps.get(device_index, 0)
        samples = None
        for name in ("nvmlDeviceGetProcessesUtilizationInfo",
                     "nvmlDeviceGetProcessUtilization"):
            function = getattr(pynvml, name, None)
            if function is None:
                continue
            try:
                samples = function(handle, timestamp)
                break
            except Exception as exc:
                if getattr(exc, "value", None) == getattr(
                        pynvml, "NVML_ERROR_NOT_FOUND", object()):
                    return 0.0, True
        if samples is None:
            return None, False

        latest_by_pid = {}
        last_seen = timestamp
        for sample in samples:
            sample_timestamp = int(getattr(sample, "timeStamp", 0) or 0)
            last_seen = max(last_seen, sample_timestamp)
            pid = int(getattr(sample, "pid", 0) or 0)
            sm_util = _finite_number(getattr(sample, "smUtil", None))
            if pid not in host_pids or sm_util is None:
                continue
            previous = latest_by_pid.get(pid)
            if previous is None or sample_timestamp >= previous[0]:
                latest_by_pid[pid] = (sample_timestamp, sm_util)
        if last_seen > timestamp:
            self._nvml_process_timestamps[device_index] = last_seen
        return min(100.0, sum(value for _, value in latest_by_pid.values())), True

    def _collect_gpu(self, visible_pids: List[int]) -> dict:
        empty = {
            "gpu_pct": None,
            "gpu_avg_pct": None,
            "gpu_util_scope": "unavailable",
            "gpu_process_pct": None,
            "gpu_device_pct": None,
            "gpu_mem_mb": None,
            "gpu_mem_scope": "unavailable",
            "gpu_process_mem_mb": None,
            "gpu_device_mem_mb": None,
            "gpu_temp_c": None,
            "gpu_power_w": None,
            "gpus": [],
        }
        if not self._nvml_available:
            return empty

        if self._cgroup and self._containerized:
            host_pids = set(self._cgroup.pids())
        else:
            host_pids = {self._host_pid(pid) for pid in visible_pids}
        devices = []
        for index, handle in enumerate(self._nvml_handles):
            item = {
                "index": index,
                "util_pct": None,
                "process_util_pct": None,
                "process_util_supported": False,
                "memory_used_mb": None,
                "process_memory_mb": None,
                "process_memory_supported": False,
                "temperature_c": None,
                "power_w": None,
            }
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                item["name"] = name.decode() if isinstance(name, bytes) else str(name)
            except Exception:
                item["name"] = None
            try:
                item["util_pct"] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                pass
            process_util, process_util_supported = self._nvml_process_utilization(
                handle, index, host_pids)
            item["process_util_pct"] = process_util
            item["process_util_supported"] = process_util_supported
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                item["memory_used_mb"] = memory.used / _MIB
            except Exception:
                pass
            process_memory, supported = self._nvml_process_memory(handle, host_pids)
            item["process_memory_mb"] = process_memory
            item["process_memory_supported"] = supported
            try:
                item["temperature_c"] = float(pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            try:
                item["power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                pass
            devices.append(item)

        util_values = [item["util_pct"] for item in devices if item["util_pct"] is not None]
        process_util_supported = bool(devices) and all(
            item["process_util_supported"] for item in devices)
        process_util_values = [item["process_util_pct"] for item in devices
                               if item["process_util_pct"] is not None]
        canonical_util = max(process_util_values) if process_util_supported \
            and process_util_values else (max(util_values) if util_values else None)
        device_memory = [item["memory_used_mb"] for item in devices
                         if item["memory_used_mb"] is not None]
        process_memory_supported = bool(devices) and all(
            item["process_memory_supported"] for item in devices)
        process_memory = sum(item["process_memory_mb"] or 0.0 for item in devices)
        canonical_memory = process_memory if process_memory_supported else (
            sum(device_memory) if device_memory else None)
        temperatures = [item["temperature_c"] for item in devices
                        if item["temperature_c"] is not None]
        powers = [item["power_w"] for item in devices if item["power_w"] is not None]

        for item in devices:
            for key in ("util_pct", "process_util_pct", "memory_used_mb", "process_memory_mb",
                        "temperature_c", "power_w"):
                if item[key] is not None:
                    item[key] = round(item[key], 3)

        return {
            "gpu_pct": round(canonical_util, 3) if canonical_util is not None else None,
            "gpu_avg_pct": round(sum(util_values) / len(util_values), 3)
            if util_values else None,
            "gpu_util_scope": "workload_processes" if process_util_supported else
            ("visible_devices" if canonical_util is not None else "unavailable"),
            "gpu_process_pct": round(max(process_util_values), 3)
            if process_util_supported and process_util_values else None,
            "gpu_device_pct": round(max(util_values), 3) if util_values else None,
            "gpu_mem_mb": round(canonical_memory, 3) if canonical_memory is not None else None,
            "gpu_mem_scope": "workload_processes" if process_memory_supported else
            ("visible_devices" if canonical_memory is not None else "unavailable"),
            "gpu_process_mem_mb": round(process_memory, 3) if process_memory_supported else None,
            "gpu_device_mem_mb": round(sum(device_memory), 3) if device_memory else None,
            "gpu_temp_c": round(max(temperatures), 3) if temperatures else None,
            "gpu_power_w": round(sum(powers), 3) if powers else None,
            "gpus": devices,
        }

    @staticmethod
    def _cpu_temperature() -> Optional[float]:
        readings = []
        for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            zone_type = (_read_text(os.path.join(os.path.dirname(path), "type")) or "").lower()
            if not any(marker in zone_type for marker in (
                    "cpu", "soc", "package", "x86_pkg", "coretemp", "k10temp")):
                continue
            value = _finite_number(_read_text(path))
            if value is None:
                continue
            value = value / 1000.0 if abs(value) > 500 else value
            if -50 <= value <= 250:
                readings.append(value)
        if not readings and _PSUTIL:
            try:
                sensors = psutil.sensors_temperatures()
                for name in ("coretemp", "k10temp", "cpu_thermal", "soc_thermal", "zenpower"):
                    entries = sensors.get(name, [])
                    readings.extend(entry.current for entry in entries
                                    if entry.current is not None and -50 <= entry.current <= 250)
            except Exception:
                pass
        return round(max(readings), 3) if readings else None

    @staticmethod
    def _discover_rapl() -> List[dict]:
        domains = []
        for path in glob.glob("/sys/class/powercap/*/energy_uj"):
            base = os.path.dirname(path)
            name = (_read_text(os.path.join(base, "name")) or "").lower()
            if name and "package" not in name and "socket" not in name:
                continue
            if not os.access(path, os.R_OK):
                continue
            max_range = _finite_number(_read_text(os.path.join(base, "max_energy_range_uj")))
            domains.append({
                "path": path,
                "max_range": max_range,
                "last": None,
                "energy_j": 0.0,
            })
        return domains

    def _prime_rapl(self, now: float):
        self._rapl_last_time = now
        self._rapl_observed_duration = 0.0
        for domain in self._rapl:
            domain["last"] = _finite_number(_read_text(domain["path"]))
            domain["energy_j"] = 0.0

    def _collect_rapl(
            self, now: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not self._rapl:
            return None, None, None
        elapsed = now - self._rapl_last_time if self._rapl_last_time is not None else 0.0
        interval_energy_j = 0.0
        valid = False
        for domain in self._rapl:
            current = _finite_number(_read_text(domain["path"]))
            previous = domain["last"]
            domain["last"] = current
            if current is None or previous is None:
                continue
            delta = current - previous
            if delta < 0 and domain["max_range"] is not None:
                delta += domain["max_range"]
            if delta < 0:
                continue
            energy_j = delta / 1_000_000.0
            domain["energy_j"] += energy_j
            interval_energy_j += energy_j
            valid = True
        self._rapl_last_time = now
        if valid and elapsed > 0:
            self._rapl_observed_duration += elapsed
        total_energy = sum(domain["energy_j"] for domain in self._rapl)
        power = interval_energy_j / elapsed if valid and elapsed > 0 else None
        return (
            power,
            total_energy if valid else None,
            self._rapl_observed_duration if valid else None,
        )
