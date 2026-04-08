"""
main.py — API FastAPI para o container de Áudio (ASR + TTS).

Endpoints:
    GET  /health         → status do container
    GET  /info           → informações detalhadas
    POST /asr            → transcrição de áudio para texto
    POST /tts            → síntese de texto para áudio WAV
    POST /benchmark      → suite de benchmarks (TTS→ASR pipeline)
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
from app.inference_asr import load_asr, transcribe, get_asr_device, is_asr_loaded
from app.inference_tts import load_tts, synthesize, is_tts_loaded


# ─── Modelos de request ───────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str


class BenchmarkRequest(BaseModel):
    profile: str = "quick"


# ─── Variáveis globais ─────────────────────────────────────────────────────────

_capabilities = {}
_models_config = {}
_benchmark_config = {}
_datasets_config = {}
_last_benchmark_result = None


# ─── Startup ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _capabilities, _models_config, _benchmark_config, _datasets_config

    print("[audio] Iniciando container...")

    # Detecta hardware
    try:
        _capabilities = CapabilityDetector().detect()
        print(f"[audio] CPU: {_capabilities.get('cpu', {}).get('model', 'desconhecido')}")
    except Exception as e:
        print(f"[audio] Aviso: capabilities falhou: {e}")
        _capabilities = {}

    # Lê configs
    try:
        with open("/app/config/models.yaml") as f:
            _models_config = yaml.safe_load(f)
        with open("/app/config/benchmark.yaml") as f:
            _benchmark_config = yaml.safe_load(f)
        with open("/app/config/datasets.yaml") as f:
            _datasets_config = yaml.safe_load(f)
    except Exception as e:
        print(f"[audio] Aviso: erro ao ler config: {e}")

    # Carrega ASR (Whisper)
    try:
        asr_config = _models_config.get("audio", {}).get("asr", {})
        default_asr = asr_config.get("default", "whisper-small")
        asr_model_cfg = asr_config.get("models", {}).get(default_asr, {"model_id": "small"})
        load_asr(asr_model_cfg)
        print(f"[audio] ASR pronto em {get_asr_device().upper()}")
    except Exception as e:
        print(f"[audio] ERRO ao carregar ASR: {e}")

    # Carrega TTS (Piper)
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


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDC Audio",
    description="Container de Áudio — Whisper ASR + Piper TTS",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Verifica saúde do container de áudio."""
    asr_ok = is_asr_loaded()
    return {
        "status": "ok" if asr_ok else "degraded",
        "front": "audio",
        "asr": "whisper" if asr_ok else "não carregado",
        "tts": "piper" if is_tts_loaded() else "fallback",
        "device": get_asr_device(),
        "gpu": get_asr_device() == "cuda",
    }


@app.get("/info")
async def info():
    """Informações do modelo e hardware."""
    return {
        "model": {
            "asr": "faster-whisper",
            "tts": "piper-tts",
            "device": get_asr_device(),
        },
        "system": _capabilities,
    }


@app.post("/asr")
async def asr_endpoint(file: UploadFile = File(...)):
    """
    Transcreve áudio WAV para texto.

    Args:
        file: Arquivo WAV de áudio.

    Retorna:
        text, language, rtf, duration_sec, segments
    """
    try:
        audio_bytes = await file.read()
        result = transcribe(audio_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    """
    Sintetiza texto em áudio WAV.

    Args:
        text: Texto para converter em voz.

    Retorna:
        WAV binary (audio/wav)
    """
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
    """
    Calcula Word Error Rate (WER) entre referência e hipótese.

    WER = (Substituições + Deleções + Inserções) / Palavras_Referência

    Args:
        reference: Texto correto (referência).
        hypothesis: Texto transcrito pelo ASR.

    Retorna:
        WER como float (0.0 = perfeito, 1.0 = completamente errado)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0

    # Usa difflib para calcular operações de edição
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    matches = sum(block.size for block in matcher.get_matching_blocks())

    # Palavras corretas = matches; resto = erros
    errors = len(ref_words) - matches
    wer = errors / len(ref_words)
    return round(max(0.0, wer), 4)


@app.post("/benchmark")
async def benchmark_endpoint(request: BenchmarkRequest):
    """
    Executa benchmark do pipeline TTS→ASR.

    Gera áudio com TTS para cada frase, transcreve com ASR,
    calcula WER e RTF. Também mede throughput e latência.
    """
    global _last_benchmark_result

    # Pega frases do dataset
    phrases = (
        _datasets_config
        .get("audio", {})
        .get("asr", {})
        .get("datasets", {})
        .get("self_generated", {})
        .get("phrases", ["Teste de reconhecimento de voz."])
    )

    # Pipeline TTS→ASR para calcular WER
    wer_results = []
    rtf_list = []
    latency_list = []

    for phrase in phrases:
        try:
            # TTS
            wav_bytes = synthesize(phrase)

            # ASR
            t0 = time.perf_counter()
            result = transcribe(wav_bytes)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Calcula WER
            wer = _calculate_wer(phrase, result.get("text", ""))
            wer_results.append(wer)

            if result.get("rtf"):
                rtf_list.append(result["rtf"])
            latency_list.append(latency_ms)
        except Exception as e:
            print(f"[audio] Erro no pipeline para '{phrase}': {e}")

    # Benchmark de throughput do ASR (função de benchmark)
    profiles = _benchmark_config.get("profiles", {})
    profile = profiles.get(request.profile, profiles.get("quick", {}))

    # Pre-gera audio pra medir so o ASR no benchmark
    _pregen_wav = synthesize("Teste de inferência de reconhecimento de voz com whisper.")

    def asr_benchmark_fn():
        """Transcreve audio pre-gerado (mede so ASR, sem TTS)."""
        return transcribe(_pregen_wav)

    try:
        loop = asyncio.get_event_loop()
        bench_result = await loop.run_in_executor(None, partial(run_full_benchmark, asr_benchmark_fn, profile, _capabilities))
    except Exception as e:
        bench_result = {"error": str(e)}

    # Resultado final
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "profile": request.profile,
        "front": "audio",
        "pipeline_results": {
            "phrases_tested": len(wer_results),
            "avg_wer": round(sum(wer_results) / len(wer_results), 4) if wer_results else None,
            "avg_rtf": round(sum(rtf_list) / len(rtf_list), 4) if rtf_list else None,
            "avg_latency_ms": round(sum(latency_list) / len(latency_list), 2) if latency_list else None,
        },
        "benchmark_results": bench_result.get("benchmark_results", {}),
        "system_info": _capabilities,
    }

    # Salva
    os.makedirs("/app/results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"/app/results/audio_{request.profile}_{timestamp}.json", "w") as f:
        json.dump(result, f, indent=2)

    _last_benchmark_result = result
    return result


@app.get("/benchmark/last")
async def benchmark_last():
    if _last_benchmark_result is None:
        raise HTTPException(status_code=404, detail="Nenhum benchmark executado")
    return _last_benchmark_result
