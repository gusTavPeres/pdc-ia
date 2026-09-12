"""Testes dos coletores de telemetria para Linux ARM e Jetson."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from shared.device_telemetry import (
    HOST_TELEMETRY_SCHEMA,
    PROVIDER_TELEMETRY_SCHEMA,
    SysfsTelemetryCollector,
    load_host_telemetry,
    load_provider_snapshots,
    merge_telemetry,
    parse_tegrastats,
    write_host_telemetry,
)
from shared.metrics_collector import MetricsCollector


class TegrastatsParserTests(unittest.TestCase):
    def test_parses_jetson_resources_and_board_input_power(self):
        line = (
            "RAM 1545/31919MB (lfb 7400x4MB) CPU "
            "[10%@1190,20%@1190,off,0%@1190] EMC_FREQ 12%@1600 "
            "GR3D_FREQ 81%@918 GPU@56C CPU@48.5C "
            "VDD_GPU_SOC 3065/3000 VIN_SYS_5V0 5120/5000"
        )

        result = parse_tegrastats(line)

        self.assertEqual(result["platform"]["family"], "jetson")
        self.assertEqual(result["memory"]["used_mb"], 1545)
        self.assertEqual(result["cpu"]["utilization_pct"], 10.0)
        self.assertEqual(result["gpu"]["utilization_pct"], 81)
        self.assertEqual(result["gpu"]["frequency_mhz"], 918)
        self.assertEqual(result["memory_controller"]["utilization_pct"], 12)
        self.assertEqual(result["gpu"]["temperature_c"], 56)
        self.assertEqual(result["power"]["board_w"], 5.12)
        self.assertEqual(result["power"]["board_source"], "tegrastats:VIN_SYS_5V0")

    def test_parses_power_values_with_explicit_units(self):
        result = parse_tegrastats(
            "RAM 1000/8000MB GR3D_FREQ 0% cpu@42C "
            "VDD_IN 6240mW/6100mW VDD_CPU 900mW/850mW"
        )

        self.assertEqual(result["power"]["board_w"], 6.24)
        self.assertEqual(len(result["power"]["rails"]), 2)
        self.assertEqual(result["gpu"]["utilization_pct"], 0)

    def test_parses_current_jetson_gpu_dla_and_pva_formats(self):
        result = parse_tegrastats(
            "RAM 4601/63895MB GR3D_FREQ 99%@[1098,1098,1098] "
            "NVDLA0 73%@1331 NVDLA1 0%@1331 PVA0_FREQ [10%,25%]@1011"
        )

        self.assertEqual(result["gpu"]["frequency_mhz"], 1098)
        devices = {item["id"]: item for item in result["accelerators"]}
        self.assertEqual(devices["jetson-nvdla0"]["utilization_pct"], 73)
        self.assertEqual(devices["jetson-nvdla0"]["kind"], "npu")
        self.assertEqual(devices["jetson-pva0"]["utilization_pct"], 25)
        self.assertEqual(devices["jetson-pva0"]["frequency_mhz"], 1011)


class SysfsTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sys = self.root / "sys"
        self.proc = self.root / "proc"
        self.sys.mkdir()
        self.proc.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write(path: Path, value: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def test_collects_arm_platform_devfreq_thermal_and_board_power(self):
        self._write(
            self.sys / "firmware/devicetree/base/model",
            "Raspberry Pi 5 Model B Rev 1.0\x00",
        )
        self._write(self.sys / "class/thermal/thermal_zone0/type", "cpu-thermal")
        self._write(self.sys / "class/thermal/thermal_zone0/temp", "51250")
        self._write(self.sys / "class/devfreq/17000000.gpu/name", "nvgpu")
        self._write(self.sys / "class/devfreq/17000000.gpu/load", "820")
        self._write(self.sys / "class/devfreq/17000000.gpu/cur_freq", "918000000")
        self._write(self.sys / "class/hwmon/hwmon0/name", "ina3221")
        self._write(self.sys / "class/hwmon/hwmon0/power1_label", "VDD_IN")
        self._write(self.sys / "class/hwmon/hwmon0/power1_input", "5120000")
        self._write(
            self.sys / "devices/system/cpu/cpufreq/policy0/scaling_cur_freq", "2400000")
        self._write(self.sys / "class/thermal/cooling_device0/type", "thermal-cpufreq-0")
        self._write(self.sys / "class/thermal/cooling_device0/cur_state", "2")
        self._write(self.sys / "class/thermal/cooling_device0/max_state", "4")

        with patch("shared.device_telemetry.platform.machine", return_value="aarch64"):
            snapshot = SysfsTelemetryCollector(
                str(self.sys), str(self.proc)).snapshot()

        self.assertEqual(snapshot["platform"]["family"], "raspberry_pi")
        self.assertEqual(snapshot["cpu"]["temperature_c"], 51.25)
        self.assertEqual(snapshot["cpu"]["frequency_mhz"], 2400)
        self.assertEqual(snapshot["gpu"]["utilization_pct"], 82)
        self.assertEqual(snapshot["gpu"]["frequency_mhz"], 918)
        self.assertEqual(snapshot["power"]["board_w"], 5.12)
        self.assertTrue(snapshot["throttling"]["active"])

    def test_uses_devfreq_busy_time_deltas_when_available(self):
        base = self.sys / "class/devfreq/rknpu"
        self._write(base / "name", "rknpu")
        self._write(base / "busy_time", "10")
        self._write(base / "total_time", "100")
        collector = SysfsTelemetryCollector(str(self.sys), str(self.proc))

        first = collector.snapshot()
        self._write(base / "busy_time", "30")
        self._write(base / "total_time", "140")
        second = collector.snapshot()

        self.assertIsNone(first["accelerators"][0]["utilization_pct"])
        self.assertEqual(second["accelerators"][0]["kind"], "npu")
        self.assertEqual(second["accelerators"][0]["utilization_pct"], 50)

    def test_aggregates_duplicate_cooling_devices(self):
        for index, state in enumerate((0, 2)):
            base = self.sys / f"class/thermal/cooling_device{index}"
            self._write(base / "type", "Processor")
            self._write(base / "cur_state", str(state))
            self._write(base / "max_state", "4")

        throttling = SysfsTelemetryCollector(
            str(self.sys), str(self.proc)).snapshot()["throttling"]

        self.assertTrue(throttling["active"])
        self.assertEqual(throttling["devices"], [{
            "name": "Processor",
            "state": 2,
            "max_state": 4,
            "limits_compute": True,
            "count": 2,
        }])


class TelemetryDocumentTests(unittest.TestCase):
    def test_atomic_document_loader_rejects_stale_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "host.json")
            timestamp = time.time()
            write_host_telemetry(path, {
                "timestamp_epoch": timestamp,
                "sources": ["linux_sysfs"],
            })

            current = load_host_telemetry(path, max_age_sec=5, now=timestamp + 1)
            stale = load_host_telemetry(path, max_age_sec=5, now=timestamp + 6)

        self.assertEqual(current["schema_version"], HOST_TELEMETRY_SCHEMA)
        self.assertIsNone(stale)

    def test_loads_only_current_external_provider_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            now = time.time()
            valid = {
                "schema_version": PROVIDER_TELEMETRY_SCHEMA,
                "provider": "test-npu",
                "timestamp_epoch": now,
                "accelerators": [{
                    "id": "npu0",
                    "kind": "npu",
                    "utilization_pct": 75,
                }],
            }
            stale = dict(valid, provider="old", timestamp_epoch=now - 30)
            Path(directory, "valid.json").write_text(json.dumps(valid), encoding="utf-8")
            Path(directory, "stale.json").write_text(json.dumps(stale), encoding="utf-8")

            documents = load_provider_snapshots(directory, max_age_sec=5)

        self.assertEqual(len(documents), 1)
        self.assertIn("external:test-npu", documents[0]["sources"])

    def test_ignores_provider_with_non_finite_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "schema_version": PROVIDER_TELEMETRY_SCHEMA,
                "provider": "broken",
                "timestamp_epoch": time.time(),
                "accelerators": [{"id": "npu0", "utilization_pct": float("nan")}],
            }
            Path(directory, "broken.json").write_text(
                json.dumps(payload), encoding="utf-8")

            documents = load_provider_snapshots(directory, max_age_sec=5)

        self.assertEqual(documents, [])

    def test_specific_provider_values_replace_same_device(self):
        merged = merge_telemetry(
            {
                "sources": ["linux_sysfs"],
                "accelerators": [{"id": "npu0", "kind": "npu", "utilization_pct": None}],
            },
            {
                "sources": ["external:test"],
                "accelerators": [{"id": "npu0", "kind": "npu", "utilization_pct": 66}],
            },
        )

        self.assertEqual(merged["sources"], ["linux_sysfs", "external:test"])
        self.assertEqual(merged["accelerators"], [{
            "id": "npu0", "kind": "npu", "utilization_pct": 66,
        }])

    def test_specific_provider_keeps_discovery_metadata(self):
        merged = merge_telemetry(
            {"accelerators": [{
                "id": "npu0", "kind": "npu", "name": "NPU integrada",
                "source": "linux_devfreq",
            }]},
            {"accelerators": [{
                "id": "npu0", "utilization_pct": 66,
                "source": "vendor_runtime",
            }]},
        )

        self.assertEqual(merged["accelerators"], [{
            "id": "npu0",
            "kind": "npu",
            "name": "NPU integrada",
            "utilization_pct": 66,
            "source": "vendor_runtime",
        }])

    def test_metrics_collector_reads_non_gpu_accelerator_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "host.json")
            write_host_telemetry(path, {
                "timestamp_epoch": time.time(),
                "sources": ["external:test-npu"],
                "platform": {"family": "linux_arm", "model": "Test SBC"},
                "power": {"board_w": 8.5, "board_source": "sensor:input"},
                "accelerators": [{
                    "id": "npu0",
                    "kind": "npu",
                    "name": "Test NPU",
                    "utilization_pct": 72,
                    "memory_used_mb": 128,
                    "power_w": 2.5,
                    "temperature_c": 55,
                }],
            })
            with patch.dict("os.environ", {"PDC_HOST_TELEMETRY_PATH": path}):
                snapshot = MetricsCollector()._collect_host_device_telemetry()

        self.assertEqual(snapshot["platform_family"], "linux_arm")
        self.assertEqual(snapshot["platform_power_w"], 8.5)
        self.assertEqual(snapshot["accelerator_pct"], 72)
        self.assertEqual(snapshot["accelerator_memory_mb"], 128)

    def test_board_power_prevents_component_double_counting(self):
        collector = MetricsCollector()
        collector._data = [
            {
                "timestamp": 0,
                "ram_scope": "process_tree_rss",
                "platform_power_w": 10,
                "platform_power_source": "sensor:input",
                "gpu_power_w": 3,
                "gpu_util_scope": "visible_devices",
                "gpu_mem_scope": "unavailable",
                "accelerator_util_scope": "unavailable",
                "accelerator_memory_scope": "unavailable",
                "gpus": [{"power_w": 3, "source": "test"}],
            },
            {
                "timestamp": 1,
                "ram_scope": "process_tree_rss",
                "platform_power_w": 10,
                "platform_power_source": "sensor:input",
                "gpu_power_w": 3,
                "gpu_util_scope": "visible_devices",
                "gpu_mem_scope": "unavailable",
                "accelerator_util_scope": "unavailable",
                "accelerator_memory_scope": "unavailable",
                "gpus": [{"power_w": 3, "source": "test"}],
            },
        ]

        summary = collector.get_summary()

        self.assertEqual(summary["total_energy_j"], 10)
        self.assertEqual(summary["avg_power_w"], 10)
        self.assertEqual(summary["scopes"]["power_w"], "host_board")
        self.assertEqual(summary["power_sources"], ["sensor:input"])



class ReadTextResilienteTest(unittest.TestCase):
    """read_text é o choke point de todo sysfs: nunca pode propagar exceção.

    Um thermal_zone do Jetson que retorna EAGAIN derrubava o agente inteiro
    (TypeError do codec escapava do except). Aqui garantimos que qualquer
    falha de leitura vira None (campo ausente), não crash.
    """

    def test_arquivo_inexistente_vira_none(self):
        from shared.device_telemetry import read_text
        self.assertIsNone(read_text("/sys/nao/existe/temp"))

    def test_leitura_que_levanta_erro_vira_none(self):
        import builtins

        import shared.device_telemetry as dt
        real_open = builtins.open

        class Explode:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): raise TypeError("can't concat NoneType to bytes")

        def fake_open(path, *a, **k):
            if "sensor_eagain" in str(path):
                return Explode()
            return real_open(path, *a, **k)

        with mock.patch.object(builtins, "open", fake_open):
            self.assertIsNone(dt.read_text("/sys/sensor_eagain/temp"))


if __name__ == "__main__":
    unittest.main()
