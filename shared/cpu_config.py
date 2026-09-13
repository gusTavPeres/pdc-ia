"""Configuração de paralelismo de CPU para as frentes de inferência.

Por padrão os runtimes (torch, llama.cpp, ctranslate2) usam só parte dos
núcleos — num hexa-core, ~metade fica ociosa e o benchmark subestima o
hardware. Aqui centralizamos: usar TODOS os núcleos, ou o que PDC_NUM_THREADS
mandar. Também exporta as env de OpenMP/OpenBLAS antes dos imports pesados.
"""
import os


def cpu_threads() -> int:
    """Núcleos a usar: PDC_NUM_THREADS, senão todos os do host/container."""
    env = os.environ.get("PDC_NUM_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return os.cpu_count() or 4


def apply_openmp_env() -> int:
    """Fixa OMP/OpenBLAS nos núcleos. Chamar ANTES de importar torch/ct2."""
    n = cpu_threads()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))
    return n
