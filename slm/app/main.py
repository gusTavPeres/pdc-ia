"""API do serviço de modelos de linguagem compactos."""

import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, "/app")

import asyncio
from functools import partial, wraps

import yaml

from app.inference import (
    generate,
    get_benchmark_fn,
    get_device,
    get_loaded_config,
    get_model_name,
    load_model,
)
from shared.benchmark_suite import run_full_benchmark
from shared.capabilities import CapabilityDetector
from shared.cpu_config import apply_openmp_env
from shared.metrics_collector import runtime_execution_info
from shared.result_writer import save_result

apply_openmp_env()  # fixa OMP/OpenBLAS nos nucleos antes de carregar modelos


class PredictRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    max_tokens: int = Field(default=150, ge=1, le=4096)


class BenchmarkRequest(BaseModel):
    profile: str = Field(default="quick", min_length=1, max_length=64)


class SwitchModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=128)


_capabilities = {}
_models_config = {}
_benchmark_config = {}
_datasets_config = {}
_last_benchmark_result = None
_benchmark_lock = threading.Lock()


def _single_benchmark(function):
    @wraps(function)
    async def wrapped(*args, **kwargs):
        if not _benchmark_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Já existe um benchmark SLM em execução")
        try:
            return await function(*args, **kwargs)
        finally:
            _benchmark_lock.release()
    return wrapped


def _reload_runtime_config():
    global _models_config, _benchmark_config, _datasets_config

    with open("/app/config/models.yaml") as f:
        _models_config = yaml.safe_load(f) or {}
    with open("/app/config/benchmark.yaml") as f:
        _benchmark_config = yaml.safe_load(f) or {}
    with open("/app/config/datasets.yaml") as f:
        _datasets_config = yaml.safe_load(f) or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _capabilities, _models_config, _benchmark_config, _datasets_config

    print("[slm] Iniciando...")

    try:
        _capabilities = CapabilityDetector().detect()
    except Exception:
        _capabilities = {}

    try:
        _reload_runtime_config()
    except Exception as e:
        print(f"[slm] Erro ao ler config: {e}")

    try:
        slm_cfg = _models_config.get("slm", {})
        default = slm_cfg.get("default", "tinyllama-1.1b-q4")
        model_cfg = slm_cfg.get("models", {}).get(default, {})
        dataset_cfg = _datasets_config.get("slm", {}).get("datasets", {}).get("prompts_pt_20", {})
        load_model(model_cfg, dataset_config=dataset_cfg)
        print(f"[slm] Pronto: {get_model_name()} em {get_device().upper()}")
    except Exception as e:
        print(f"[slm] ERRO ao carregar modelo: {e}")

    yield


app = FastAPI(title="PDC SLM", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    model = get_model_name()
    is_ok = model != "não carregado"
    payload = {
        "status": "ok" if is_ok else "degraded",
        "front": "slm",
        "model": model,
        "device": get_device(),
        "gpu": get_device() != "cpu",
    }
    return payload if is_ok else JSONResponse(payload, status_code=503)


@app.get("/info")
async def info():
    try:
        _reload_runtime_config()
    except Exception:
        pass
    return {
        "model": get_loaded_config(),
        "available_models": list(_models_config.get("slm", {}).get("models", {}).keys()),
        "system": _capabilities,
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """Gera texto com o modelo ativo."""
    if _benchmark_lock.locked():
        raise HTTPException(status_code=409, detail="Inferência indisponível durante o benchmark")
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(generate, request.prompt, max_tokens=request.max_tokens),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/switch-model")
async def switch_model(request: SwitchModelRequest):
    """Baixa, quando necessário, e ativa o modelo solicitado."""
    if _benchmark_lock.locked():
        raise HTTPException(status_code=409, detail="Não é possível trocar o modelo durante o benchmark")
    try:
        _reload_runtime_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao recarregar config: {e}")

    slm_cfg = _models_config.get("slm", {})
    model_cfg = slm_cfg.get("models", {}).get(request.model)

    if not model_cfg:
        available = list(slm_cfg.get("models", {}).keys())
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{request.model}' não encontrado. Disponíveis: {available}"
        )

    try:
        dataset_cfg = _datasets_config.get("slm", {}).get("datasets", {}).get("prompts_pt_20", {})
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(load_model, model_cfg, dataset_config=dataset_cfg),
        )
        return {
            "status": "ok",
            "loaded": get_loaded_config(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark")
@_single_benchmark
async def benchmark(request: BenchmarkRequest):
    """Executa benchmark com o modelo atualmente carregado."""
    global _last_benchmark_result

    try:
        _reload_runtime_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao recarregar config: {e}")

    profiles = _benchmark_config.get("profiles", {})
    profile = profiles.get(request.profile)
    if not profile:
        raise HTTPException(status_code=400, detail=f"Perfil '{request.profile}' não encontrado")

    try:
        loop = asyncio.get_running_loop()
        fn = get_benchmark_fn()
        benchmark_model = get_model_name()
        dataset_name = _datasets_config.get("slm", {}).get("default", "prompts_pt_20")
        result = await loop.run_in_executor(
            None,
            partial(
                run_full_benchmark,
                fn,
                profile,
                _capabilities,
                profile_name=request.profile,
                front="slm",
                model=benchmark_model,
                dataset=dataset_name,
                execution={
                    **runtime_execution_info(),
                    "device": get_device(),
                    "backend": "llama-cpp-python",
                    "gpu_layers": get_loaded_config().get("gpu_layers"),
                },
            ),
        )
        result["profile"] = request.profile
        result["front"] = "slm"
        result["model"] = benchmark_model

        model_slug = benchmark_model.replace(".gguf", "").replace(" ", "_")[:30]
        save_result(
            "/app/results",
            f"slm_{model_slug}_{request.profile}",
            result,
        )

        _last_benchmark_result = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmark/last")
async def benchmark_last():
    if _last_benchmark_result is None:
        raise HTTPException(status_code=404, detail="Nenhum benchmark executado")
    return _last_benchmark_result
