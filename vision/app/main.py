"""API do serviço de visão computacional."""

import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, "/app")

import asyncio
from functools import partial, wraps

import yaml

from app.inference import get_benchmark_fn, get_device, get_model_name, load_model, predict
from shared.benchmark_suite import run_full_benchmark
from shared.capabilities import CapabilityDetector
from shared.cpu_config import apply_openmp_env
from shared.metrics_collector import runtime_execution_info
from shared.result_writer import save_result

apply_openmp_env()  # fixa OMP/OpenBLAS nos nucleos antes de carregar modelos


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
            raise HTTPException(status_code=409, detail="Já existe um benchmark de visão em execução")
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

    print("[vision] Iniciando container...")

    try:
        detector = CapabilityDetector()
        _capabilities = detector.detect()
        print(f"[vision] CPU: {_capabilities.get('cpu', {}).get('model', 'desconhecido')}")
        print(f"[vision] GPU: {_capabilities.get('gpu', {}).get('model', 'não detectada')}")
    except Exception as e:
        print(f"[vision] Aviso: detecção de capabilities falhou: {e}")
        _capabilities = {}

    try:
        _reload_runtime_config()
    except Exception as e:
        print(f"[vision] Aviso: erro ao ler config: {e}")
        _models_config = {}
        _benchmark_config = {}

    try:
        vision_config = _models_config.get("vision", {})
        default_model = vision_config.get("default", "yolov8n")
        model_cfg = vision_config.get("models", {}).get(default_model, {})

        if not model_cfg:
            model_cfg = {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
                "filename": "yolov8n.pt"
            }

        load_model(model_cfg)
        print(f"[vision] Modelo carregado: {get_model_name()} em {get_device().upper()}")
    except Exception as e:
        print(f"[vision] ERRO ao carregar modelo: {e}")
        # O endpoint de saúde sinaliza o serviço degradado sem encerrar a API.

    print("[vision] Container pronto!")
    yield
    print("[vision] Encerrando container...")


app = FastAPI(
    title="PDC Vision",
    description="Container de Visão Computacional — YOLOv8 Object Detection",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Retorna 200 quando o modelo está carregado."""
    model = get_model_name()
    is_ok = model != "não carregado"
    payload = {
        "status": "ok" if is_ok else "degraded",
        "front": "vision",
        "model": model,
        "device": get_device(),
        "gpu": get_device() == "cuda",
    }
    return payload if is_ok else JSONResponse(payload, status_code=503)


@app.get("/info")
async def info():
    """Retorna o modelo ativo e o hardware detectado."""
    try:
        _reload_runtime_config()
    except Exception:
        pass
    return {
        "model": {
            "name": get_model_name(),
            "device": get_device(),
            "gpu": get_device() == "cuda",
            "type": "object_detection",
        },
        "available_models": list(_models_config.get("vision", {}).get("models", {}).keys()),
        "system": _capabilities,
    }


@app.post("/switch-model")
async def switch_model(request: SwitchModelRequest):
    if _benchmark_lock.locked():
        raise HTTPException(status_code=409, detail="Não é possível trocar o modelo durante o benchmark")
    try:
        _reload_runtime_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao recarregar config: {e}")

    vision_cfg = _models_config.get("vision", {})
    model_cfg = vision_cfg.get("models", {}).get(request.model)

    if not model_cfg:
        available = list(vision_cfg.get("models", {}).keys())
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{request.model}' não encontrado. Disponíveis: {available}"
        )

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, load_model, model_cfg)
        return {
            "status": "ok",
            "loaded": {
                "name": get_model_name(),
                "device": get_device(),
                "gpu": get_device() == "cuda",
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Executa detecção em uma imagem enviada por multipart/form-data."""
    if _benchmark_lock.locked():
        raise HTTPException(status_code=409, detail="Inferência indisponível durante o benchmark")
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise ValueError("Imagem vazia")
        if len(image_bytes) > 20 * 1024 * 1024:
            raise ValueError("Imagem excede 20 MB")
        loop = asyncio.get_running_loop()
        detections = await loop.run_in_executor(None, predict, image_bytes)
        return {
            "detections": detections,
            "count": len(detections),
            "model": get_model_name(),
            "device": get_device(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark")
@_single_benchmark
async def benchmark_endpoint(request: BenchmarkRequest):
    """Executa o perfil solicitado e salva o resultado em /app/results."""
    global _last_benchmark_result

    try:
        _reload_runtime_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao recarregar config: {e}")

    profile_name = request.profile
    profiles = _benchmark_config.get("profiles", {})
    profile = profiles.get(profile_name)

    if not profile:
        raise HTTPException(status_code=400, detail=f"Perfil '{profile_name}' não encontrado")

    try:
        vision_datasets = _datasets_config.get("vision", {})
        dataset_name = vision_datasets.get("default", "synthetic_10")
        dataset_config = vision_datasets.get("datasets", {}).get(dataset_name, {})
        benchmark_fn = get_benchmark_fn(dataset_config)
        benchmark_model = get_model_name()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                run_full_benchmark,
                benchmark_fn,
                profile,
                _capabilities,
                profile_name=profile_name,
                front="vision",
                model=benchmark_model,
                dataset=dataset_name,
                execution={
                    **runtime_execution_info(),
                    "device": get_device(),
                    "backend": "ultralytics",
                },
            ),
        )
        result["profile"] = profile_name
        result["front"] = "vision"
        result["model"] = benchmark_model

        save_result("/app/results", f"vision_{profile_name}", result)

        _last_benchmark_result = result
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmark/last")
async def benchmark_last():
    """Retorna o último resultado de benchmark executado nesta sessão."""
    if _last_benchmark_result is None:
        raise HTTPException(status_code=404, detail="Nenhum benchmark executado ainda")
    return _last_benchmark_result
