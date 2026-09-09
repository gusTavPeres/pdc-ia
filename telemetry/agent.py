"""Publica uma amostra comum de sensores do host para os benchmarks."""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time

from shared.device_telemetry import (
    SysfsTelemetryCollector,
    load_host_telemetry,
    load_provider_snapshots,
    merge_telemetry,
    parse_tegrastats,
    write_host_telemetry,
)


class TegrastatsReader:
    def __init__(self, command: str, interval_sec: float):
        self.command = command
        self.interval_ms = max(100, round(interval_sec * 1000))
        self.process = None
        self.thread = None
        self.latest_line = None
        self.latest_at = None
        self.lock = threading.Lock()

    def available(self) -> bool:
        return bool(self.command and os.path.isfile(self.command)
                    and os.access(self.command, os.X_OK))

    def start(self) -> None:
        if not self.available() or (self.process and self.process.poll() is None):
            return
        try:
            self.process = subprocess.Popen(
                [self.command, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self.process = None
            return
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            if not line.strip():
                continue
            with self.lock:
                self.latest_line = line.strip()
                self.latest_at = time.monotonic()

    def snapshot(self, max_age_sec: float) -> dict:
        if self.process is None:
            self.start()
        if self.process and self.process.poll() is not None:
            self.stop()
            self.start()
        with self.lock:
            line = self.latest_line
            age = time.monotonic() - self.latest_at if self.latest_at is not None else None
        return parse_tegrastats(line) if line and age is not None and age <= max_age_sec else {}

    def stop(self) -> None:
        process = self.process
        self.process = None
        if not process:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.thread = None


def _positive_float(value: str, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def run_agent() -> None:
    interval = max(0.2, _positive_float(
        os.environ.get("PDC_TELEMETRY_INTERVAL_SEC", "1"), 1.0))
    output = os.environ.get(
        "PDC_HOST_TELEMETRY_PATH", "/app/data/telemetry/host.json")
    provider_dir = os.environ.get(
        "PDC_TELEMETRY_PROVIDER_DIR", "/app/data/telemetry/providers")
    sys_root = os.environ.get("PDC_HOST_SYS_ROOT", "/host/sys")
    proc_root = os.environ.get("PDC_HOST_PROC_ROOT", "/host/proc")
    tegrastats_path = os.environ.get(
        "PDC_TELEMETRY_TEGRASTATS", "/host-tools/tegrastats")

    collector = SysfsTelemetryCollector(sys_root=sys_root, proc_root=proc_root)
    tegrastats = TegrastatsReader(tegrastats_path, interval)
    tegrastats.start()
    try:
        while True:
            started = time.monotonic()
            snapshot = collector.snapshot()
            snapshot = merge_telemetry(snapshot, tegrastats.snapshot(interval * 3))
            for provider in load_provider_snapshots(provider_dir, max_age_sec=interval * 5):
                snapshot = merge_telemetry(snapshot, provider)
            snapshot.update({
                "timestamp_epoch": time.time(),
                "sample_interval_sec": interval,
                "agent": {"name": "pdc-telemetry", "pid": os.getpid()},
            })
            write_host_telemetry(output, snapshot)
            time.sleep(max(0.0, interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        tegrastats.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente local de telemetria PDC")
    parser.add_argument("--check", metavar="PATH", help="Verifica se uma amostra está atual")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if load_host_telemetry(args.check, max_age_sec=10) else 1)
    run_agent()


if __name__ == "__main__":
    main()
