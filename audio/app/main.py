"""
main.py — API FastAPI para o container de Áudio (ASR + TTS).

Endpoints:
    GET  /health         → status do container
    GET  /info           → informações detalhadas
    POST /asr            → transcrição de áudio para texto
    POST /tts            → síntese de texto para áudio WAV
    POST /benchmark      → benchmark (modo: asr, tts ou pipeline)
    GET  /benchmark/last → último resultado
"""

import io
import os
import sys
import json
import time
import datetime
import difflib
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, "/app")

import yaml
from shared.capabilities import CapabilityDetector
import asyncio
from functools import partial
from shared.benchmark_suite import run_full_benchmark
from app.inference_asr import (
    load_asr,
    transcribe,
    get_asr_device,
    get_asr_model_name,
    is_asr_loaded,
)
from app.inference_tts import (
    load_tts,
    synthesize,
    get_tts_model_name,
    is_tts_loaded,
)


# ─── Modelos de request ───

class TTSRequest(BaseModel):
    text: str


class BenchmarkRequest(BaseModel):
    profile: str = "quick"
    mode: str = "pipeline"  # "asr", "tts", or "pipeline"


class SwitchModelRequest(BaseModel):
    target: str  # "asr" or "tts"
    model: str


# ─── Variáveis globais ───

_capabilities = {}
_models_config = {}
_benchmark_config = {}
_datasets_config = {}
_last_benchmark_result = None


def _reload_runtime_config():
    global _models_config, _benchmark_config, _datasets_config

    with open("/app/config/models.yaml") as f:
        _models_config = yaml.safe_load(f) or {}
    with open("/app/config/benchmark.yaml") as f:
        _benchmark_config = yaml.safe_load(f) or {}
    with open("/app/config/datasets.yaml") as f:
        _datasets_config = yaml.safe_load(f) or {}


# ─── Startup ───

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

    # ASR (Whisper)
    try:
        asr_config = _models_config.get("audio", {}).get("asr", {})
        default_asr = asr_config.get("default", "whisper-small")
        asr_model_cfg = asr_config.get("models", {}).get(default_asr, {"model_id": "small"})
        load_asr(asr_model_cfg)
        print(f"[audio] ASR pronto em {get_asr_device().upper()}")
    except Exception as e:
        print(f"[audio] ERRO ao carregar ASR: {e}")

    # TTS (Piper)
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


# ─── Endpoints ───

@app.get("/health")
async def health():
    asr_ok = is_asr_loaded()
    return {
        "status": "ok" if asr_ok else "degraded",
        "front": "audio",
        "asr": get_asr_model_name() if asr_ok else "não carregado",
        "tts": get_tts_model_name() if is_tts_loaded() else "fallback",
        "device": get_asr_device(),
        "gpu": get_asr_device() == "cuda",
    }


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
            load_asr(model_cfg)
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
            load_tts(model_cfg)
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
    try:
        audio_bytes = await file.read()
        result = transcribe(audio_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        wav_bytes = synthesize(request.text)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0

    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    matches = sum(block.size for block in matcher.get_matching_blocks())

    errors = len(ref_words) - matches
    wer = errors / len(ref_words)
    return round(max(0.0, wer), 4)


@app.post("/benchmark")
async def benchmark_endpoint(request: BenchmarkRequest):
    """
    Benchmark de audio com 3 modos:
      - asr: mede so o Whisper (transcrição)
      - tts: mede so o Piper (sintese de voz)
      - pipeline: TTS+ASR junto com calculo de WER
    """
    global _last_benchmark_result

    mode = request.mode

    try:
        _reload_runtime_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao recarregar config: {e}")

    profiles = _benchmark_config.get("profiles", {})
    profile = profiles.get(request.profile, profiles.get("quick", {}))

    phrases = (
        _datasets_config
        .get("audio", {})
        .get("asr", {})
        .get("datasets", {})
        .get("self_generated", {})
        .get("phrases", ["Teste de reconhecimento de voz."])
    )

    # Pipeline WER test (asr e pipeline modes)
    pipeline_results = {}
    if mode in ("pipeline", "asr"):
        wer_results = []
        rtf_list = []
        latency_list = []
        for phrase in phrases:
            try:
                wav_bytes = synthesize(phrase)
                t0 = time.perf_counter()
                result = transcribe(wav_bytes)
                latency_ms = (time.perf_counter() - t0) * 1000
                wer = _calculate_wer(phrase, result.get("text", ""))
                wer_results.append(wer)
                if result.get("rtf"):
                    rtf_list.append(result["rtf"])
                latency_list.append(latency_ms)
            except Exception as e:
                print(f"[audio] Erro no pipeline para '{phrase}': {e}")
        pipeline_results = {
            "phrases_tested": len(wer_results),
            "avg_wer": round(sum(wer_results) / len(wer_results), 4) if wer_results else None,
            "avg_rtf": round(sum(rtf_list) / len(rtf_list), 4) if rtf_list else None,
            "avg_latency_ms": round(sum(latency_list) / len(latency_list), 2) if latency_list else None,
        }

    # Benchmark function depends on mode
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
            return synthesize(p)

    elif mode == "asr":
        _pregen_wav = synthesize("Teste de inferência de reconhecimento de voz com whisper.")

        def bench_fn():
            return transcribe(_pregen_wav)

    else:  # pipeline
        def bench_fn():
            wav = synthesize("Teste de benchmark do pipeline completo.")
            return transcribe(wav)

    try:
        loop = asyncio.get_event_loop()
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
                dataset=_datasets_config.get("audio", {}).get("asr", {}).get("default", "self_generated"),
            ),
        )
    except Exception as e:
        bench_result = {"error": str(e)}

    result = {
        "timestamp": datetime.datetime.now().isoformat(),
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
        "system_info": _capabilities,
    }

    os.makedirs("/app/results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"/app/results/audio_{mode}_{request.profile}_{timestamp}.json", "w") as f:
        json.dump(result, f, indent=2)

    _last_benchmark_result = result
    return result


@app.get("/benchmark/last")
async def benchmark_last():
    if _last_benchmark_result is None:
        raise HTTPException(status_code=404, detail="Nenhum benchmark executado")
    return _last_benchmark_result
