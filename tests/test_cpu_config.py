"""Testes da configuração de threads de CPU das frentes."""
import os
import unittest
from unittest import mock

from shared.cpu_config import apply_openmp_env, cpu_threads


class CpuThreadsTest(unittest.TestCase):
    def test_usa_todos_os_nucleos_por_padrao(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("os.cpu_count", return_value=6):
                self.assertEqual(cpu_threads(), 6)

    def test_pdc_num_threads_tem_prioridade(self):
        with mock.patch.dict(os.environ, {"PDC_NUM_THREADS": "3"}):
            self.assertEqual(cpu_threads(), 3)

    def test_valor_invalido_cai_no_padrao(self):
        with mock.patch.dict(os.environ, {"PDC_NUM_THREADS": "0"}):
            with mock.patch("os.cpu_count", return_value=4):
                self.assertEqual(cpu_threads(), 4)
        with mock.patch.dict(os.environ, {"PDC_NUM_THREADS": "abc"}):
            with mock.patch("os.cpu_count", return_value=4):
                self.assertEqual(cpu_threads(), 4)

    def test_apply_openmp_seta_todas_as_vars(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("os.cpu_count", return_value=8):
                n = apply_openmp_env()
        self.assertEqual(n, 8)

    def test_apply_openmp_respeita_valor_existente(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
            with mock.patch("os.cpu_count", return_value=8):
                apply_openmp_env()
                self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")  # setdefault nao sobrescreve


if __name__ == "__main__":
    unittest.main()


class ResultCpuNormalizationTest(unittest.TestCase):
    """CPU do resultado deve expor a versão normalizada 0-100, não só cgroup."""

    def test_expoe_host_cpu_e_cores(self):
        from shared.result_parser import normalize_result
        data = {
            "front": "vision", "profile": "quick",
            "benchmark_results": {
                "resource_usage": {
                    "schema_version": "resource-2026.08-v1",
                    "avg_cpu_pct": 590.0,          # cgroup: 5.9 nucleos
                    "avg_host_cpu_pct": 98.3,      # normalizado 0-100
                    "timeseries": [],
                },
            },
        }
        r = normalize_result(data)
        self.assertAlmostEqual(r["avg_host_cpu_pct"], 98.3)
        self.assertAlmostEqual(r["cpu_cores_used"], 5.9)  # 590/100
        self.assertLessEqual(r["avg_host_cpu_pct"], 100.0)  # nunca 600
