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
