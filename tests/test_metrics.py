import json
import math
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from metrics_server import gpu_metrics
from shared.audio_metrics import word_error_rate
from shared.benchmark_suite import (
    run_full_benchmark,
    run_latency,
    run_resource_usage,
    run_throughput,
)
from shared.capabilities import (
    _base_block_device_name,
    _pci_accelerator_category,
    hardware_fingerprint,
    load_hardware_snapshot,
    merge_host_capabilities,
    save_hardware_snapshot,
)
from shared.config_validation import validate_benchmark_config
from shared.dataset_loader import load_dataset
from shared.methodology import NumericResultAccumulator, benchmark_metadata, summarize_errors
from shared.metrics_collector import (
    MetricsCollector,
    _CgroupReader,
    _gpu_metric_scope,
    _integrate_metric,
    _time_weighted_average,
)
from shared.model_loader import _safe_filename, _valid_cache
from shared.result_parser import (
    finite_number,
    is_benchmark_result,
    normalize_resource_usage,
    normalize_result,
)
from shared.result_writer import save_result
from shared.scorecard import compute_scorecard


class ResultParserTests(unittest.TestCase):
    def test_only_benchmark_documents_are_accepted_as_results(self):
        self.assertTrue(is_benchmark_result({
            "front": "vision",
            "benchmark_results": {"throughput": {"ops_per_sec": 1}},
        }))
        self.assertFalse(is_benchmark_result({
            "cpu": {"model": "Cortex-A76"},
            "storage": {"type": "NVMe SSD"},
        }))
        self.assertFalse(is_benchmark_result({
            "front": "vision",
            "benchmark_results": {},
        }))

    def test_finite_number_preserves_zero(self):
        self.assertEqual(finite_number(0), 0.0)
        self.assertEqual(finite_number("0"), 0.0)
        self.assertIsNone(finite_number(True))
        self.assertIsNone(finite_number(math.nan))
        self.assertIsNone(finite_number(math.inf))

    def test_current_resource_schema_preserves_scopes_and_zeroes(self):
        resource = normalize_resource_usage({
            "schema_version": "resources-2026.07-v2",
            "scopes": {
                "ram_mb": "container_cgroup_working_set",
                "gpu_mem_mb": "workload_processes",
            },
            "peak_ram_mb": 0,
            "peak_gpu_util": 0,
            "peak_vram_mb": 0,
            "avg_power_w": 0,
            "total_energy_j": 0,
        })
        self.assertEqual(resource["peak_ram_mb"], 0.0)
        self.assertEqual(resource["peak_gpu_util"], 0.0)
        self.assertEqual(resource["peak_vram_mb"], 0.0)
        self.assertEqual(resource["avg_power_w"], 0.0)
        self.assertEqual(resource["total_energy_j"], 0.0)
        self.assertEqual(resource["ram_scope"], "container_cgroup_working_set")
        self.assertEqual(resource["vram_scope"], "workload_processes")

    def test_missing_current_telemetry_has_no_implied_scope(self):
        resource = normalize_resource_usage({
            "schema_version": "resources-2026.07-v2",
            "scopes": {
                "ram_mb": "container_cgroup_working_set",
                "gpu_mem_mb": "workload_processes",
                "gpu_pct": "workload_processes",
                "cpu_pct": "container_cgroup",
                "power_w": "measured_components",
            },
            "energy_coverage_ratio": 2,
        })
        self.assertEqual(resource["ram_scope"], "unavailable")
        self.assertEqual(resource["vram_scope"], "unavailable")
        self.assertEqual(resource["gpu_scope"], "unavailable")
        self.assertEqual(resource["cpu_scope"], "unavailable")
        self.assertEqual(resource["power_scope"], "unavailable")
        self.assertEqual(resource["energy_coverage_ratio"], 1.0)

    def test_legacy_process_metrics_take_precedence_over_host_aliases(self):
        resource = normalize_resource_usage({
            "peak_ram_mb": 8192,
            "peak_process_ram_mb": 512,
            "peak_gpu_mem_mb": 1024,
            "avg_process_cpu_pct": 75,
            "gpu_energy_wh": 0.25,
        })
        self.assertEqual(resource["peak_ram_mb"], 512.0)
        self.assertEqual(resource["ram_scope"], "process_tree_rss")
        self.assertEqual(resource["peak_vram_mb"], 1024.0)
        self.assertEqual(resource["total_energy_j"], 900.0)

    def test_result_aliases_are_normalized(self):
        row = normalize_result({
            "front": "vision",
            "model": "model.pt",
            "metadata": {
                "dataset": "images-v1",
                "execution": {"architecture": "aarch64", "gpu_mode": "cpu"},
                "hardware_profile": {"id": "sbc-a", "description": "Bancada ARM"},
            },
            "system_info": {
                "cpu": {"model": "Cortex-A76", "arch": "aarch64"},
                "gpu": {"model": "Mali-G52"},
                "storage": {"type": "NVMe SSD", "device": "/dev/nvme0n1"},
                "provenance": {
                    "hardware_id": "hardware-1",
                    "inventory_source": "host_snapshot",
                },
            },
            "benchmark_results": {
                "throughput": {"ops_per_sec": 2, "ops_total": 7},
                "resource_usage": {"peak_ram_mb": 20},
                "memory_bandwidth": {
                    "sequential": {"bandwidth_gbs": 4.5},
                    "random": {"bandwidth_gbs": 1.5},
                },
                "sustained_load": {"throughput_per_minute": [2, 4]},
            },
        }, filename="/tmp/result.json")
        self.assertEqual(row["file"], "result.json")
        self.assertEqual(row["total_iters"], 7.0)
        self.assertEqual(row["mem_bw_gbs"], 4.5)
        self.assertEqual(row["sustained_avg_ops"], 3.0)
        self.assertEqual(row["dataset"], "images-v1")
        self.assertEqual(row["execution"]["gpu_mode"], "cpu")
        self.assertEqual(row["cpu_model"], "Cortex-A76")
        self.assertEqual(row["gpu_model"], "Mali-G52")
        self.assertEqual(row["architecture"], "aarch64")
        self.assertEqual(row["hardware_id"], "hardware-1")
        self.assertEqual(row["inventory_source"], "host_snapshot")
        self.assertEqual(row["hardware_profile_id"], "sbc-a")
        self.assertEqual(row["storage_type"], "NVMe SSD")
        self.assertEqual(row["storage_device"], "/dev/nvme0n1")

    def test_cpu_average_uses_the_duration_of_each_interval(self):
        resource = normalize_resource_usage({
            "timeseries": [
                {"timestamp": 0, "process_cpu_pct": None},
                {"timestamp": 1, "process_cpu_pct": 100},
                {"timestamp": 4, "process_cpu_pct": 0},
            ],
        })
        self.assertEqual(resource["avg_cpu_pct"], 25.0)
        self.assertEqual(resource["cpu_scope"], "process_tree")


class MetricMathTests(unittest.TestCase):
    def test_gpu_metrics_hides_command_details_on_driver_failure(self):
        failure = subprocess.CalledProcessError(
            255,
            ["nvidia-smi", "--query-gpu=..."],
            stderr="Failed to initialize NVML: Driver/library version mismatch\n",
        )
        with mock.patch("metrics_server.subprocess.check_output", side_effect=failure):
            result = gpu_metrics()

        self.assertEqual(result["error"], "Telemetria NVIDIA indisponível")
        self.assertEqual(
            result["detail"], "Failed to initialize NVML: Driver/library version mismatch")
        self.assertNotIn("--query-gpu", json.dumps(result))

    def test_gpu_metrics_reads_driver_failure_from_stdout(self):
        failure = subprocess.CalledProcessError(
            255,
            ["nvidia-smi", "--query-gpu=..."],
            output="Failed to initialize NVML: Driver/library version mismatch\n",
        )
        with mock.patch("metrics_server.subprocess.check_output", side_effect=failure):
            result = gpu_metrics()

        self.assertEqual(
            result["detail"], "Failed to initialize NVML: Driver/library version mismatch")

    def test_block_device_names_cover_sbc_and_pc_storage(self):
        self.assertEqual(_base_block_device_name("/dev/nvme0n1p5"), "nvme0n1")
        self.assertEqual(_base_block_device_name("/dev/mmcblk0p1"), "mmcblk0")
        self.assertEqual(_base_block_device_name("/dev/sda3"), "sda")
        self.assertEqual(_base_block_device_name("/dev/dm-0"), "dm-0")

    def test_npu_driver_is_detected_even_with_a_generic_pci_class(self):
        self.assertEqual(_pci_accelerator_category("0x088000", "apex"), "npu")
        self.assertEqual(_pci_accelerator_category("0x030000", "nvidia"), "gpu")
        self.assertIsNone(_pci_accelerator_category("0x020000", "igc"))

    def test_word_error_rate_uses_word_level_edit_distance(self):
        self.assertEqual(word_error_rate("um dois tres", "um dois tres"), 0.0)
        self.assertEqual(word_error_rate("um dois tres", "um quatro tres"), 0.3333)
        self.assertEqual(word_error_rate("um dois", "um novo dois"), 0.5)
        self.assertEqual(word_error_rate("", "texto"), 1.0)

    def test_time_weighted_average_and_energy_integration(self):
        samples = [
            {"timestamp": 0, "power_w": 10},
            {"timestamp": 1, "power_w": 20},
            {"timestamp": 3, "power_w": 20},
        ]
        self.assertAlmostEqual(_time_weighted_average(samples, "power_w"), 55 / 3)
        energy, duration = _integrate_metric(samples, "power_w")
        self.assertAlmostEqual(energy, 55)
        self.assertEqual(duration, 3)

    def test_metric_integration_does_not_bridge_missing_samples(self):
        samples = [
            {"timestamp": 0, "power_w": 10},
            {"timestamp": 1, "power_w": None},
            {"timestamp": 2, "power_w": 20},
            {"timestamp": 3, "power_w": 20},
        ]
        self.assertEqual(_time_weighted_average(samples, "power_w"), 20)
        energy, duration = _integrate_metric(samples, "power_w")
        self.assertEqual(energy, 20)
        self.assertEqual(duration, 1)

    def test_gpu_scope_uses_all_samples_with_measurements(self):
        samples = [
            {"gpu_pct": 20, "gpu_util_scope": "workload_processes"},
            {"gpu_pct": 80, "gpu_util_scope": "visible_devices"},
            {"gpu_pct": None, "gpu_util_scope": "unavailable"},
        ]
        self.assertEqual(
            _gpu_metric_scope(samples, "gpu_pct", "gpu_util_scope"),
            "visible_devices",
        )

    def test_cpu_scope_follows_process_fallback(self):
        collector = object.__new__(MetricsCollector)
        collector._data = [
            {"timestamp": 0, "ram_mb": 10, "ram_scope": "process_tree_rss"},
            {
                "timestamp": 1,
                "ram_mb": 11,
                "ram_scope": "process_tree_rss",
                "process_cpu_pct": 75,
            },
        ]
        collector._lock = threading.Lock()
        collector._containerized = True
        collector._cgroup = SimpleNamespace(
            average_cpu_percent=lambda: None,
            limits=lambda: {},
            cpu_time_delta=lambda: None,
        )
        collector._rapl = []

        summary = collector.get_summary()

        self.assertEqual(summary["avg_cpu_pct"], 75.0)
        self.assertEqual(summary["scopes"]["cpu_pct"], "process_tree")

    def test_cgroup_working_set_and_cpu_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "memory.current").write_text(str(100 * 1024 * 1024))
            (base / "memory.stat").write_text(f"inactive_file {20 * 1024 * 1024}\n")
            (base / "memory.max").write_text(str(256 * 1024 * 1024))
            (base / "cpu.stat").write_text("usage_usec 1000000\n")
            reader = _CgroupReader("v2", str(base), str(base))

            memory = reader.memory()
            self.assertEqual(memory["working_set_mb"], 80)
            self.assertEqual(memory["total_mb"], 100)
            self.assertEqual(memory["limit_mb"], 256)

            reader.reset(1.0)
            (base / "cpu.stat").write_text("usage_usec 2000000\n")
            self.assertIsNone(reader.cpu_percent(2.0))
            (base / "cpu.stat").write_text("usage_usec 3000000\n")
            self.assertEqual(reader.cpu_percent(3.0), 100.0)

    def test_container_root_cgroup_prefers_host_mounted_docker_scope(self):
        scope = "/sys/fs/cgroup/system.slice/docker-abc123.scope"

        def glob_paths(pattern):
            return [scope] if "docker-abc123" in pattern else []

        def is_file(path):
            return path in {
                "/sys/fs/cgroup/memory.current",
                f"{scope}/memory.current",
            }

        with mock.patch(
                "shared.metrics_collector._read_text", return_value="0::/\n"), \
                mock.patch(
                    "shared.metrics_collector._is_containerized", return_value=True), \
                mock.patch(
                    "shared.metrics_collector.socket.gethostname", return_value="abc123"), \
                mock.patch("shared.metrics_collector.glob.glob", side_effect=glob_paths), \
                mock.patch("shared.metrics_collector.os.path.isfile", side_effect=is_file):
            reader = _CgroupReader.discover()

        self.assertIsNotNone(reader)
        self.assertEqual(reader.memory_path, scope)

    def test_host_mounted_v1_cgroup_is_discovered_by_container_id(self):
        memory_path = "/sys/fs/cgroup/memory/docker/abc123"
        cpu_path = "/sys/fs/cgroup/cpu,cpuacct/docker/abc123"

        def glob_paths(pattern):
            if pattern == "/sys/fs/cgroup/memory/docker/abc123*":
                return [memory_path]
            if pattern == "/sys/fs/cgroup/cpu,cpuacct/docker/abc123*":
                return [cpu_path]
            return []

        def is_file(path):
            return path in {
                f"{memory_path}/memory.usage_in_bytes",
                f"{cpu_path}/cpuacct.usage",
            }

        with mock.patch(
                "shared.metrics_collector.socket.gethostname", return_value="abc123"), \
                mock.patch("shared.metrics_collector.glob.glob", side_effect=glob_paths), \
                mock.patch("shared.metrics_collector.os.path.isfile", side_effect=is_file):
            reader = _CgroupReader._discover_host_mounted_v1()

        self.assertIsNotNone(reader)
        self.assertEqual(reader.mode, "v1")
        self.assertEqual(reader.memory_path, memory_path)
        self.assertEqual(reader.cpu_path, cpu_path)

    def test_numeric_accumulator_ignores_non_finite_values(self):
        accumulator = NumericResultAccumulator()
        accumulator.update({"valid": 2, "nan": math.nan, "flag": True})
        accumulator.update({"valid": 4, "inf": math.inf})
        self.assertEqual(accumulator.summary(), {
            "valid": {
                "avg": 3.0,
                "sum": 6.0,
                "min": 2.0,
                "max": 4.0,
                "samples": 2,
            },
        })

    def test_error_summary_keeps_total_count(self):
        summary = summarize_errors(["sample"] * 20, total_count=125)
        self.assertEqual(summary["count"], 125)
        self.assertEqual(summary["sampled_count"], 20)

    def test_benchmark_stops_after_repeated_inference_failures(self):
        def fail():
            raise RuntimeError("modelo indisponível")

        started = time.perf_counter()
        throughput = run_throughput(fail, duration_sec=5, warmup=0)
        latency = run_latency(fail, samples=100)
        resources = run_resource_usage(fail, duration_sec=5, interval_sec=0.05)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 2)
        self.assertTrue(throughput["aborted_due_to_errors"])
        self.assertEqual(throughput["error_count"], 10)
        self.assertTrue(latency["aborted_due_to_errors"])
        self.assertEqual(latency["error_count"], 10)
        self.assertTrue(resources["aborted_due_to_errors"])
        self.assertEqual(resources["inference_error_count"], 10)

    def test_latency_error_rate_uses_attempted_samples_after_early_stop(self):
        attempts = 0

        def infer():
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                raise RuntimeError("falha")
            return {}

        result = run_latency(infer, samples=50)

        self.assertTrue(result["aborted_due_to_errors"])
        self.assertEqual(result["samples_collected"], 1)
        self.assertEqual(result["error_count"], 10)
        self.assertEqual(result["error_rate"], round(10 / 11, 4))
        self.assertEqual(result["error_rate"], result["quality"]["error_rate"])

    def test_process_gpu_utilization_is_filtered_by_container_pids(self):
        collector = object.__new__(MetricsCollector)
        collector._nvml_process_timestamps = {}
        samples = [
            SimpleNamespace(pid=10, timeStamp=100, smUtil=30),
            SimpleNamespace(pid=11, timeStamp=101, smUtil=25),
            SimpleNamespace(pid=20, timeStamp=102, smUtil=80),
        ]
        with mock.patch(
                "shared.metrics_collector.pynvml.nvmlDeviceGetProcessesUtilizationInfo",
                return_value=samples):
            utilization, supported = collector._nvml_process_utilization(
                object(), 0, {10, 11})

        self.assertTrue(supported)
        self.assertEqual(utilization, 55)
        self.assertEqual(collector._nvml_process_timestamps[0], 102)

    def test_comparability_key_contains_profile_and_methodology(self):
        metadata = benchmark_metadata(
            "vision", "model.pt", None, "dataset", {"device": "cpu"},
            profile="quick", run_id="run-1")
        key = metadata["comparability_key"]
        self.assertEqual(metadata["run_id"], "run-1")
        self.assertEqual(key["profile"], "quick")
        self.assertTrue(key["methodology_version"])


class HardwareSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.host = {
            "cpu": {
                "model": "Cortex-A76", "arch": "aarch64",
                "cores_physical": 4, "cores_logical": 4, "cache_kb": 2048,
            },
            "ram": {"total_gb": 8.0, "available_gb": 6.0, "type": "LPDDR4"},
            "gpu": {"available": False, "model": None, "vram_mb": None},
            "accelerators": {"available": True, "devices": [{
                "kind": "pci", "category": "npu", "address": "0000:01:00.0",
                "vendor_id": "0x1e60", "device_id": "0x2864", "driver": "hailo_pci",
            }]},
            "serial": {"available": False, "devices": []},
            "storage": {"type": "NVMe SSD", "device": "/dev/nvme0n1p2"},
            "temperature": {"cpu_avg_c": 42.0},
            "power": {"package_watts": None},
        }

    def test_snapshot_is_atomic_and_preserves_runtime_visibility(self):
        runtime = {
            **self.host,
            "storage": {"type": None, "device": "overlay"},
            "accelerators": {"available": False, "devices": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "hardware" / "host.json")
            saved = save_hardware_snapshot(self.host, path)
            loaded = load_hardware_snapshot(path)
            merged = merge_host_capabilities(runtime, loaded)

        self.assertEqual(loaded["hardware_id"], saved["hardware_id"])
        self.assertEqual(merged["storage"]["type"], "NVMe SSD")
        self.assertEqual(merged["runtime_visibility"]["storage"]["device"], "overlay")
        self.assertFalse(merged["runtime_visibility"]["accelerators"]["available"])
        self.assertEqual(merged["provenance"]["inventory_source"], "host_snapshot")

        merged_again = merge_host_capabilities(merged, loaded)
        self.assertEqual(merged_again["storage"]["type"], "NVMe SSD")
        self.assertEqual(
            merged_again["runtime_visibility"]["storage"]["device"], "overlay")

    def test_fingerprint_ignores_dynamic_sensor_values(self):
        changed = json.loads(json.dumps(self.host))
        changed["ram"]["available_gb"] = 1.0
        changed["temperature"]["cpu_avg_c"] = 90.0
        self.assertEqual(hardware_fingerprint(self.host), hardware_fingerprint(changed))

    def test_invalid_snapshot_falls_back_to_runtime_detection(self):
        merged = merge_host_capabilities(self.host, snapshot={})
        self.assertEqual(merged["storage"]["type"], "NVMe SSD")
        self.assertEqual(merged["provenance"]["inventory_source"], "runtime_detection")

    def test_malformed_snapshot_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.json"
            path.write_text("{invalid", encoding="utf-8")
            self.assertIsNone(load_hardware_snapshot(str(path)))

    def test_benchmark_records_hardware_profile(self):
        result = run_full_benchmark(
            lambda: {},
            {"throughput": {"enabled": False}},
            self.host,
            profile_name="disabled",
            front="vision",
            model="model.pt",
            dataset="images-v1",
            hardware_config={
                "profile": {"id": "arm-board", "description": "Bancada ARM"},
            },
        )
        self.assertEqual(result["metadata"]["hardware_profile"]["id"], "arm-board")
        self.assertTrue(result["system_info"]["provenance"]["hardware_id"])


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_invalid_duration_and_default_profile(self):
        with self.assertRaisesRegex(ValueError, "duration_sec"):
            validate_benchmark_config({
                "profiles": {
                    "quick": {
                        "throughput": {"duration_sec": 0, "warmup": 1},
                    },
                },
            })

        with self.assertRaisesRegex(ValueError, "default_profile"):
            validate_benchmark_config({
                "profiles": {
                    "quick": {
                        "throughput": {"duration_sec": 1, "warmup": 0},
                    },
                },
                "default_profile": "missing",
            })

    def test_rejects_non_finite_and_unbounded_values(self):
        with self.assertRaisesRegex(ValueError, "finito"):
            validate_benchmark_config({
                "profiles": {
                    "bad": {"throughput": {"duration_sec": math.inf, "warmup": 0}},
                },
            })
        with self.assertRaisesRegex(ValueError, "262144"):
            validate_benchmark_config({
                "profiles": {
                    "bad": {
                        "memory_bandwidth": {
                            "enabled": True,
                            "patterns": ["sequential"],
                            "block_sizes_kb": [300_000],
                        },
                    },
                },
            })


class ModelCacheTests(unittest.TestCase):
    def test_filename_is_confined_to_cache(self):
        self.assertEqual(_safe_filename("../../model.gguf"), "model.gguf")
        with self.assertRaises(ValueError):
            _safe_filename("..")

    def test_cache_rejects_empty_file_and_checks_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(b"")
            self.assertFalse(_valid_cache(str(path)))
            path.write_bytes(b"model")
            self.assertTrue(_valid_cache(str(path)))
            self.assertTrue(_valid_cache(
                str(path),
                "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
            ))
        self.assertFalse(_valid_cache(str(path), "0" * 64))


class DatasetLoaderTests(unittest.TestCase):
    def test_invalid_source_does_not_create_placeholder_results(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.assertRaisesRegex(ValueError, "origem de dataset inválida"):
            load_dataset({"source": "missing.json"}, cache_dir=directory)

    def test_generated_audio_is_reproducible(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy não instalado")
        config = {
            "source": "generate",
            "type": "asr",
            "samples": 2,
            "duration_sec": 0.01,
            "seed": 17,
        }
        with tempfile.TemporaryDirectory() as directory:
            first = load_dataset(config, cache_dir=directory)
            second = load_dataset(config, cache_dir=directory)
        self.assertTrue(np.array_equal(first[0], second[0]))

    def test_generated_dataset_has_a_memory_limit(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.assertRaisesRegex(ValueError, "excede o limite"):
            load_dataset({
                "source": "generate",
                "type": "vision",
                "samples": 10_000,
                "width": 8192,
                "height": 8192,
            }, cache_dir=directory)

    def test_remote_dataset_uses_atomic_downloader(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("shared.dataset_loader.get_or_download",
                           return_value=f"{directory}/data.json") as download:
            path = load_dataset({
                "source": "https://example.com/data.json",
                "sha256": "a" * 64,
            }, cache_dir=directory)
        self.assertEqual(path, f"{directory}/data.json")
        download.assert_called_once_with(
            {
                "url": "https://example.com/data.json",
                "filename": "data.json",
                "sha256": "a" * 64,
            },
            cache_dir=directory,
            artifact_label="dataset",
        )


class ResultWriterTests(unittest.TestCase):
    def test_result_write_is_unique_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = save_result(directory, "vision quick", {"value": 1})
            second = save_result(directory, "vision quick", {"value": 2})
            self.assertNotEqual(first, second)
            self.assertEqual(Path(first).read_text().strip(), '{\n  "value": 1\n}')
            self.assertFalse(list(Path(directory).glob(".result-*.tmp")))


class ScorecardTests(unittest.TestCase):
    def test_efficiency_uses_resource_phase_counts(self):
        result = {
            "front": "slm",
            "benchmark_results": {
                "throughput": {
                    "ops_per_sec": 100,
                    "ops_total": 1000,
                    "duration_sec": 10,
                    "result_metrics": {
                        "tokens_per_sec": {"avg": 20},
                        "total_tokens": {"avg": 50},
                    },
                },
                "resource_usage": {
                    "schema_version": "resources-2026.07-v2",
                    "measurement_duration_sec": 10,
                    "inference_success_count": 20,
                    "total_energy_j": 200,
                    "avg_power_w": 20,
                    "result_metrics": {"total_tokens": {"avg": 30}},
                    "scopes": {
                        "ram_mb": "container_cgroup_working_set",
                        "gpu_mem_mb": "workload_processes",
                        "gpu_pct": "visible_devices",
                        "power_w": "measured_components",
                    },
                },
            },
        }
        scorecard = compute_scorecard(result, {})
        efficiency = scorecard["efficiency"]
        self.assertEqual(efficiency["energy_per_op_j"], 10.0)
        self.assertEqual(efficiency["primary_unit_count"], 600.0)
        self.assertEqual(efficiency["energy_per_primary_unit_j"], 0.333333)
        self.assertEqual(efficiency["resource_primary_rate"], 60.0)
        self.assertEqual(efficiency["primary_metric_per_watt"], 3.0)
        self.assertEqual(
            scorecard["measurement_basis"]["resource_operation_count"], 20.0)

    def test_primary_rates_use_phase_totals(self):
        result = {
            "front": "slm",
            "benchmark_results": {
                "throughput": {
                    "duration_sec": 5,
                    "ops_per_sec": 2,
                    "result_metrics": {
                        "total_tokens": {"sum": 125, "avg": 12.5},
                        "tokens_per_sec": {"avg": 99},
                    },
                },
                "resource_usage": {
                    "schema_version": "resources-2026.07-v2",
                    "measurement_duration_sec": 10,
                    "inference_success_count": 4,
                    "avg_power_w": 5,
                    "total_energy_j": 50,
                    "result_metrics": {
                        "total_tokens": {"sum": 80, "avg": 20},
                    },
                },
            },
        }
        scorecard = compute_scorecard(result, {})
        self.assertEqual(scorecard["primary_metric"]["value"], 25.0)
        self.assertEqual(scorecard["efficiency"]["resource_primary_rate"], 8.0)
        self.assertEqual(scorecard["efficiency"]["primary_unit_count"], 80.0)


class DashboardBackendTests(unittest.TestCase):
    def test_result_timestamp_is_rendered_in_the_host_timezone(self):
        import metrics_server

        value = "2026-07-29T05:37:20+00:00"
        expected = metrics_server.datetime.fromisoformat(value).astimezone().strftime("%d/%m %H:%M")
        self.assertEqual(metrics_server._format_result_timestamp(value), expected)

    def test_model_activation_skips_reload_when_model_is_active(self):
        import metrics_server

        with mock.patch.object(metrics_server, "_expected_runtime_model", return_value="model.gguf"), \
                mock.patch.object(metrics_server, "_loaded_runtime_model", return_value="model.gguf"), \
                mock.patch.object(metrics_server, "_http_post") as post:
            result = metrics_server.activate_model_target("slm", "model")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["already_loaded"])
        post.assert_not_called()

    def test_results_recompute_scorecard_with_current_hardware_config(self):
        import metrics_server

        raw = {
            "front": "vision",
            "model": "model.pt",
            "scorecard": {
                "schema_version": "scorecard-2026.07-v2",
                "performance_score": 1,
            },
            "benchmark_results": {
                "throughput": {"ops_per_sec": 10, "ops_total": 100},
                "resource_usage": {
                    "schema_version": "resources-2026.07-v2",
                    "measurement_duration_sec": 10,
                    "inference_success_count": 100,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(raw))
            metrics_server._result_cache.clear()
            with mock.patch.object(metrics_server, "RESULTS_DIR", directory), \
                    mock.patch.object(metrics_server, "load_hardware_config", return_value={
                        "score_references": {"vision": {"ops_per_sec": 10}},
                    }):
                rows = metrics_server.all_results()
            metrics_server._result_cache.clear()

        self.assertEqual(rows[0]["performance_score"], 100.0)

    def test_model_runtime_limits_are_validated(self):
        import metrics_server

        with self.assertRaisesRegex(ValueError, "camadas de GPU"):
            metrics_server._build_model_entry("slm", {
                "url": "https://example.com/model.gguf",
                "gpu_layers": -1,
                "context_length": 2048,
            })

    def test_identical_legacy_results_are_listed_once(self):
        import metrics_server

        raw = {
            "front": "vision",
            "model": "model.pt",
            "benchmark_results": {"throughput": {"ops_per_sec": 1}},
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            payload = json.dumps(raw)
            first.write_text(payload)
            second.write_text(payload)
            metrics_server._result_cache.clear()
            with mock.patch.object(metrics_server, "RESULTS_DIR", directory), \
                    mock.patch.object(metrics_server, "load_hardware_config", return_value={}):
                rows = metrics_server.all_results()
            metrics_server._result_cache.clear()

        self.assertEqual(len(rows), 1)


class CliTests(unittest.TestCase):
    def test_hardware_json_output_is_machine_readable(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "pdc"), "detect", "--json"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertIn("cpu", payload)
        self.assertNotIn("\033", result.stdout)


if __name__ == "__main__":
    unittest.main()
