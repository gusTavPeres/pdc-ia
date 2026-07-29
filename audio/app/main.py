"""API dos serviços de ASR e TTS."""

import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

sys.path.insert(0, "/app")

import asyncio
from functools import partial, wraps

import yaml

from app.inference_asr import (
    get_asr_device,
    get_asr_model_name,
    is_asr_loaded,
    load_asr,
    transcribe,
)
from app.inference_tts import (
    get_tts_model_name,
    is_tts_loaded,
    load_tts,
    synthesize,
    synthesize_with_metrics,
)
from shared.audio_metrics import word_error_rate
from shared.benchmark_suite import run_full_benchmark
from shared.capabilities import CapabilityDetector
from shared.methodology import result_quality, summarize_errors
from shared.metrics_collector import runtime_execution_info
from shared.result_writer import save_result
from shared.scorecard import compute_scorecard, load_hardware_config


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5_000)


class BenchmarkRequest(BaseModel):
    profile: str = Field(default="quick", min_length=1, max_length=64)
    mode: Literal["asr", "tts", "pipeline"] = "pipeline"


class SwitchModelRequest(BaseModel):
    target: Literal["asr", "tts"]
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
            raise HTTPException(status_code=409, detail="Já existe um benchmark de áudio em execução")
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

    print("[audio] Iniciando container...")

    try:
        _capabilities = CapabilityDetector().detect()
        print(f"[audio] CPU: {_capabilities.get('cpu', {}).get('model', 'desconhecido')}")
    except Exception as e:
        print(f"[audio] Aviso: capabilities falhou: {e}")
        _capabilities = {}

    try:
        _reload_runtime_config()
    except Exception as e:
        print(f"[audio] Aviso: erro ao ler config: {e}")

    try:
        asr_config = _models_config.get("audio", {}).get("asr", {})
        default_asr = asr_config.get("default", "whisper-small")
        asr_model_cfg = asr_config.get("models", {}).get(default_asr, {"model_id": "small"})
        load_asr(asr_model_cfg)
        print(f"[audio] ASR pronto em {get_asr_device().upper()}")
    except Exception as e:
        print(f"[audio] ERRO ao carregar ASR: {e}")

    try:
        tts_config = _models_config.get("audio", {}).get("tts", {})
        default_tts = tts_config.get("default", "piper-pt-br")
        tts_model_cfg = tts_config.get("models", {}).get(default_tts, {})
        if tts_model_cfg:
            load_tts(tts_model_cfg)
        print(f"[audio] TTS pronto: {is_tts_loaded()}")
    except Exception as e:
        print(f"[audio] ERRO ao carregar TTS: {e}")

    print("[audio] Container pronto!")
    yield
    print("[audio] Encerrando...")


app = FastAPI(
    title="PDC Audio",
    description="Container de Áudio — Whisper ASR + Piper TTS",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    asr_ok = is_asr_loaded()
    tts_ok = is_tts_loaded()
    is_ok = asr_ok and tts_ok
    payload = {
        "status": "ok" if is_ok else "degraded",
        "front": "audio",
        "asr": get_asr_model_name() if asr_ok else "não carregado",
        "tts": get_tts_model_name() if tts_ok else "não carregado",
        "device": get_asr_device(),
        "gpu": get_asr_device() == "cuda",
    }
    return payload if is_ok else JSONResponse(payload, status_code=503)


@app.get("/info")
async def info():
    try:
        _reload_runtime_config()
    except Exception:
        pass
    return {
        "model": {
            "asr_engine": "faster-whisper",
            "asr_model": get_asr_model_name(),
            "tts_engine": "piper-tts",
            "tts_model": get_tts_model_name(),
            "device": get_asr_device(),
        },
        "available_models": {
            "asr": list(_models_config.get("audio", {}).get("asr", {}).get("models", {}).keys()),
            "tts": list(_models_config.get("audio", {}).get("tts", {}).get("models", {}).keys()),
        },
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

    audio_cfg = _models_config.get("audio", {})
    if request.target == "asr":
        section = audio_cfg.get("asr", {})
        model_cfg = section.get("models", {}).get(request.model)
        if not model_cfg:
            available = list(section.get("models", {}).keys())
            raise HTTPException(
                status_code=404,
                detail=f"Modelo ASR '{request.model}' não encontrado. Disponíveis: {available}"
            )
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, load_asr, model_cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif request.target == "tts":
        section = audio_cfg.get("tts", {})
        model_cfg = section.get("models", {}).get(request.model)
        if not model_cfg:
            available = list(section.get("models", {}).keys())
            raise HTTPException(
                status_code=404,
                detail=f"Modelo TTS '{request.model}' não encontrado. Disponíveis: {available}"
            )
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, load_tts, model_cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="target deve ser 'asr' ou 'tts'")

    return {
        "status": "ok",
        "target": request.target,
        "loaded": {
            "asr": get_asr_model_name(),
            "tts": get_tts_model_name(),
            "device": get_asr_device(),
        },
    }


@app.post("/asr")
async def asr_endpoint(file: UploadFile = File(...)):
    if _benchmark_lock.locked():
        raise HTTPException(status_code=409, detail="Inferência indisponível durante o benchmark")
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise ValueError("Áudio vazio")
        if len(audio_bytes) > 100 * 1024 * 1024:
            raise ValueError("Áudio excede 100 MB")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, transcribe, audio_bytes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    if _benchmark_lock.locked():
        raise HTTPException(status_code=409, detail="Inferência indisponível durante o benchmark")
    try:
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, synthesize, request.text)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark")
@_single_benchmark
async def benchmark_endpoint(request: BenchmarkRequest):
    """Executa ASR, TTS ou o pipeline completo, conforme o modo solicitado."""
    global _last_benchmark_result

    mode = request.mode

    try:
        _reload_runtime_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao recarregar config: {e}")

    profiles = _benchmark_config.get("profiles", {})
    profile = profiles.get(request.profile)
    if not profile:
        raise HTTPException(status_code=400, detail=f"Perfil '{request.profile}' não encontrado")

    phrases = (
        _datasets_config
        .get("audio", {})
        .get("asr", {})
        .get("datasets", {})
        .get("self_generated", {})
        .get("phrases", ["Teste de reconhecimento de voz."])
    )

    pipeline_results = {}
    if mode in ("pipeline", "asr"):
        wer_results = []
        rtf_list = []
        latency_list = []
        pipeline_errors = []
        pipeline_error_count = 0
        for phrase in phrases:
            try:
                pipeline_started = time.perf_counter()
                wav_bytes = synthesize(phrase)
                t0 = time.perf_counter()
                result = transcribe(wav_bytes)
                if not result.get("text"):
                    raise RuntimeError("ASR retornou transcrição vazia")
                asr_elapsed = time.perf_counter() - t0
                pipeline_elapsed = time.perf_counter() - pipeline_started
                measured_elapsed = pipeline_elapsed if mode == "pipeline" else asr_elapsed
                latency_ms = measured_elapsed * 1000
                wer = word_error_rate(phrase, result.get("text", ""))
                wer_results.append(wer)
                audio_duration = result.get("duration_sec")
                if audio_duration and audio_duration > 0:
                    rtf_list.append(measured_elapsed / audio_duration)
                latency_list.append(latency_ms)
            except Exception as e:
                pipeline_error_count += 1
                if len(pipeline_errors) < 20:
                    pipeline_errors.append(f"{type(e).__name__}: {e}")
        pipeline_results = {
            "phrases_requested": len(phrases),
            "phrases_tested": len(wer_results),
            "avg_wer": round(sum(wer_results) / len(wer_results), 4) if wer_results else None,
            "avg_rtf": round(sum(rtf_list) / len(rtf_list), 4) if rtf_list else None,
            "avg_latency_ms": round(sum(latency_list) / len(latency_list), 2) if latency_list else None,
            "quality": result_quality(len(wer_results), pipeline_error_count),
            "errors": summarize_errors(pipeline_errors, total_count=pipeline_error_count),
        }

    if mode == "tts":
        test_phrases = [
            "Teste de síntese de voz.",
            "O sistema está funcionando corretamente.",
            "Benchmark de text to speech com piper.",
        ]
        _phrase_idx = [0]

        def bench_fn():
            p = test_phrases[_phrase_idx[0] % len(test_phrases)]
            _phrase_idx[0] += 1
            return synthesize_with_metrics(p)

    elif mode == "asr":
        _pregen_wav = synthesize("Teste de inferência de reconhecimento de voz com whisper.")

        def bench_fn():
            result = transcribe(_pregen_wav)
            if not result.get("text"):
                raise RuntimeError("ASR retornou transcrição vazia")
            return result

    else:  # pipeline
        def bench_fn():
            started = time.perf_counter()
            wav = synthesize("Teste de benchmark do pipeline completo.")
            result = transcribe(wav)
            if not result.get("text"):
                raise RuntimeError("ASR retornou transcrição vazia")
            pipeline_elapsed = time.perf_counter() - started
            audio_duration = result.get("duration_sec")
            output = dict(result)
            output["asr_processing_sec"] = result.get("processing_sec")
            output["processing_sec"] = round(pipeline_elapsed, 4)
            output["pipeline_rtf"] = round(pipeline_elapsed / audio_duration, 4) \
                if audio_duration and audio_duration > 0 else None
            return output

    loop = asyncio.get_running_loop()
    bench_result = await loop.run_in_executor(
        None,
        partial(
            run_full_benchmark,
            bench_fn,
            profile,
            _capabilities,
            profile_name=request.profile,
            front="audio",
            model={"asr": get_asr_model_name(), "tts": get_tts_model_name()},
            mode=mode,
            dataset=_datasets_config.get("audio", {}).get("asr", {}).get(
                "default", "self_generated"),
            execution={
                **runtime_execution_info(),
                "asr_device": get_asr_device(),
                "tts_device": "cpu",
                "backend": "faster-whisper+piper",
            },
        ),
    )

    result = {
        "schema_version": bench_result.get("schema_version"),
        "run_id": bench_result.get("run_id"),
        "timestamp": bench_result.get("timestamp"),
        "profile": request.profile,
        "mode": mode,
        "front": "audio",
        "model": {
            "asr": get_asr_model_name(),
            "tts": get_tts_model_name(),
        },
        "pipeline_results": pipeline_results,
        "benchmark_results": bench_result.get("benchmark_results", {}),
        "methodology": bench_result.get("methodology", {}),
        "metadata": bench_result.get("metadata", {}),
        "scorecard": bench_result.get("scorecard", {}),
        "system_info": bench_result.get("system_info", _capabilities),
    }
    result["scorecard"] = compute_scorecard(result, load_hardware_config())

    save_result("/app/results", f"audio_{mode}_{request.profile}", result)

    _last_benchmark_result = result
    return result


@app.get("/benchmark/last")
async def benchmark_last():
    if _last_benchmark_result is None:
        raise HTTPException(status_code=404, detail="Nenhum benchmark executado")
    return _last_benchmark_result
