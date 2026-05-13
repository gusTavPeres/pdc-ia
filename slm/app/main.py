"""
main.py — API FastAPI para o container SLM.

Endpoints:
    GET  /health              → status
    GET  /info                → modelo atual + hardware
    POST /predict             → geração de texto
    POST /switch-model        → troca o modelo sem reiniciar container
    POST /benchmark           → suite de benchmarks
    GET  /benchmark/last      → último resultado
"""

import os, sys, json, datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, "/app")

import yaml
from shared.capabilities import CapabilityDetector
import asyncio
from functools import partial
from shared.benchmark_suite import run_full_benchmark
from app.inference import (
    load_model, generate, get_benchmark_fn,
    get_device, get_model_name, get_loaded_config
)


class PredictRequest(BaseModel):
    prompt: str
    max_tokens: int = 150


class BenchmarkRequest(BaseModel):
    profile: str = "quick"


class SwitchModelRequest(BaseModel):
    model: str   # chave do models.yaml, ex: "qwen25-0.5b-q4"


_capabilities = {}
_models_config = {}
_benchmark_config = {}
_datasets_config = {}
_last_benchmark_result = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _capabilities, _models_config, _benchmark_config, _datasets_config

    print("[slm] Iniciando...")

    try:
        _capabilities = CapabilityDetector().detect()
    except Exception:
        _capabilities = {}

    try:
        with open("/app/config/models.yaml") as f:
            _models_config = yaml.safe_load(f)
        with open("/app/config/benchmark.yaml") as f:
            _benchmark_config = yaml.safe_load(f)
        with open("/app/config/datasets.yaml") as f:
            _datasets_config = yaml.safe_load(f)
    except Exception as e:
        print(f"[slm] Erro ao ler config: {e}")

    # Carrega modelo default
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
    return {
        "status": "ok" if model != "não carregado" else "degraded",
        "front": "slm",
        "model": model,
        "device": get_device(),
        "gpu": get_device() == "cuda",
    }


@app.get("/info")
async def info():
    return {
        "model": get_loaded_config(),
        "available_models": list(_models_config.get("slm", {}).get("models", {}).keys()),
        "system": _capabilities,
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """Gera texto. Retorna também qual modelo foi usado."""
    try:
        return generate(request.prompt, max_tokens=request.max_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/switch-model")
async def switch_model(request: SwitchModelRequest):
    """
    Troca o modelo carregado sem reiniciar o container.

    Baixa o modelo se necessário, descarrega o anterior e carrega o novo.

    Args:
        model: Nome do modelo (chave em config/models.yaml > slm > models)

    Retorna:
        Info do modelo carregado após a troca.
    """
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
        load_model(model_cfg, dataset_config=dataset_cfg)
        return {
            "status": "ok",
            "loaded": get_loaded_config(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    """Executa benchmark com o modelo atualmente carregado."""
    global _last_benchmark_result

    profiles = _benchmark_config.get("profiles", {})
    profile = profiles.get(request.profile, profiles.get("quick", {}))
    if not profile:
        raise HTTPException(status_code=400, detail=f"Perfil '{request.profile}' não encontrado")

    try:
        loop = asyncio.get_event_loop()
        fn = get_benchmark_fn()
        result = await loop.run_in_executor(None, partial(run_full_benchmark, fn, profile, _capabilities))
        result["profile"] = request.profile
        result["front"] = "slm"
        result["model"] = get_model_name()

        os.makedirs("/app/results", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = get_model_name().replace(".gguf", "").replace(" ", "_")[:30]
        with open(f"/app/results/slm_{model_slug}_{request.profile}_{ts}.json", "w") as f:
            json.dump(result, f, indent=2)

        _last_benchmark_result = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmark/last")
async def benchmark_last():
    if _last_benchmark_result is None:
        raise HTTPException(status_code=404, detail="Nenhum benchmark executado")
    return _last_benchmark_result
