"""
main.py — API FastAPI para o container de Visão Computacional.

Endpoints:
    GET  /health         → status do container e modelo
    GET  /info           → informações detalhadas do sistema
    POST /predict        → detecção de objetos em imagem
    POST /benchmark      → executa suite de benchmarks
    GET  /benchmark/last → retorna último resultado de benchmark
"""

import os
import sys
import json
import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Adiciona /app ao path para importar shared/
sys.path.insert(0, "/app")

import yaml
from shared.capabilities import CapabilityDetector
import asyncio
from functools import partial
from shared.benchmark_suite import run_full_benchmark
from app.inference import load_model, predict, get_benchmark_fn, get_device, get_model_name


# ─── Modelos de request/response ─────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    """Parâmetros para executar benchmark."""
    profile: str = "quick"


# ─── Variáveis globais ────────────────────────────────────────────────────────

_capabilities = {}
_models_config = {}
_benchmark_config = {}
_last_benchmark_result = None


# ─── Startup/Shutdown ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executado no startup e shutdown do servidor."""
    global _capabilities, _models_config, _benchmark_config

    print("[vision] Iniciando container...")

    # Detecta hardware
    try:
        detector = CapabilityDetector()
        _capabilities = detector.detect()
        print(f"[vision] CPU: {_capabilities.get('cpu', {}).get('model', 'desconhecido')}")
        print(f"[vision] GPU: {_capabilities.get('gpu', {}).get('model', 'não detectada')}")
    except Exception as e:
        print(f"[vision] Aviso: detecção de capabilities falhou: {e}")
        _capabilities = {}

    # Lê configurações
    try:
        with open("/app/config/models.yaml") as f:
            _models_config = yaml.safe_load(f)
        with open("/app/config/benchmark.yaml") as f:
            _benchmark_config = yaml.safe_load(f)
    except Exception as e:
        print(f"[vision] Aviso: erro ao ler config: {e}")
        _models_config = {}
        _benchmark_config = {}

    # Carrega modelo de visão
    try:
        vision_config = _models_config.get("vision", {})
        default_model = vision_config.get("default", "yolov8n")
        model_cfg = vision_config.get("models", {}).get(default_model, {})

        if not model_cfg:
            # Fallback: config inline
            model_cfg = {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
                "filename": "yolov8n.pt"
            }

        load_model(model_cfg)
        print(f"[vision] Modelo carregado: {get_model_name()} em {get_device().upper()}")
    except Exception as e:
        print(f"[vision] ERRO ao carregar modelo: {e}")
        # Não re-raise: permite o servidor subir mesmo sem modelo
        # /health retornará status degraded

    print("[vision] Container pronto!")
    yield
    print("[vision] Encerrando container...")


# ─── Aplicação FastAPI ─────────────────────────────────────────────────────────

app = FastAPI(
    title="PDC Vision",
    description="Container de Visão Computacional — YOLOv8 Object Detection",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Verifica se o container está saudável.

    Retorna status 200 se o modelo estiver carregado.
    """
    model = get_model_name()
    is_ok = model != "não carregado"
    return {
        "status": "ok" if is_ok else "degraded",
        "front": "vision",
        "model": model,
        "device": get_device(),
        "gpu": get_device() == "cuda",
    }


@app.get("/info")
async def info():
    """
    Retorna informações detalhadas sobre o modelo e o hardware.
    """
    return {
        "model": {
            "name": get_model_name(),
            "device": get_device(),
            "gpu": get_device() == "cuda",
            "type": "object_detection",
        },
        "system": _capabilities,
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Executa detecção de objetos em uma imagem.

    Aceita imagem via multipart/form-data.
    Retorna lista de objetos detectados com bounding boxes.

    Args:
        file: Arquivo de imagem (JPEG, PNG, etc.)

    Retorna:
        {"detections": [...], "count": int, "model": str, "device": str}
    """
    try:
        image_bytes = await file.read()
        detections = predict(image_bytes)
        return {
            "detections": detections,
            "count": len(detections),
            "model": get_model_name(),
            "device": get_device(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark")
async def benchmark_endpoint(request: BenchmarkRequest):
    """
    Executa suite de benchmarks de inferência.

    Usa perfil definido em config/benchmark.yaml.
    Salva resultado em /app/results/.

    Args:
        profile: Nome do perfil (quick, standard, full)

    Retorna:
        JSON completo com resultados de throughput, latência, recursos, etc.
    """
    global _last_benchmark_result

    profile_name = request.profile
    profiles = _benchmark_config.get("profiles", {})
    profile = profiles.get(profile_name, profiles.get("quick", {}))

    if not profile:
        raise HTTPException(status_code=400, detail=f"Perfil '{profile_name}' não encontrado")

    try:
        benchmark_fn = get_benchmark_fn()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, partial(run_full_benchmark, benchmark_fn, profile, _capabilities))
        result["profile"] = profile_name
        result["front"] = "vision"

        # Salva resultado em arquivo
        os.makedirs("/app/results", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/app/results/vision_{profile_name}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)

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
