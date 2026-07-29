"""Carga e inferência de modelos GGUF com llama.cpp."""

import gc
import os
import sys
import threading
import time
from functools import wraps
from typing import Optional

sys.path.insert(0, "/app")

from shared.model_loader import get_or_download

# Estado global do modelo carregado
_model = None
_model_name = "não carregado"
_model_path = None
_n_gpu_layers = 0
_chat_format = "tinyllama"   # tinyllama | chatml
_prompts_cache = []
_benchmark_max_tokens = 80
_model_lock = threading.RLock()


def _serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _model_lock:
            return function(*args, **kwargs)
    return wrapped


def _build_prompt(user_text: str, chat_format: str) -> str:
    """Aplica o template de chat exigido pelo modelo."""
    if chat_format == "chatml":
        # Formato ChatML: <|im_start|>role\ncontent<|im_end|>
        return (
            "<|im_start|>system\nVocê é um assistente útil que responde em português.<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif chat_format == "gemma3":
        # Formato Gemma 3: <start_of_turn>role\ncontent<end_of_turn>
        return (
            f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    else:
        # Formato TinyLlama
        return (
            "<|system|>\nVocê é um assistente útil.</s>\n"
            f"<|user|>\n{user_text}</s>\n"
            "<|assistant|>\n"
        )


def _stop_tokens(chat_format: str) -> list:
    """Retorna tokens de parada para o formato do modelo."""
    if chat_format == "chatml":
        return ["<|im_end|>", "<|im_start|>"]
    elif chat_format == "gemma3":
        return ["<end_of_turn>", "<eos>"]
    else:
        return ["</s>", "<|user|>", "<|system|>"]


@_serialized
def load_model(config: dict, dataset_config: Optional[dict] = None) -> None:
    """Baixa e carrega um GGUF, com fallback de GPU para CPU."""
    global _model, _model_name, _model_path, _n_gpu_layers, _chat_format
    global _prompts_cache, _benchmark_max_tokens

    model_path = get_or_download(config, cache_dir="/app/data/models")
    model_name = os.path.basename(model_path)
    chat_format = config.get("chat_format", "tinyllama")

    if _model is not None:
        print(f"[slm] Descarregando modelo anterior: {_model_name}")
        del _model
        _model = None
        gc.collect()

    _model_path = model_path
    _model_name = model_name
    _chat_format = chat_format
    print(f"[slm] Carregando: {_model_name} (formato: {_chat_format})")

    # Prompts do dataset
    if dataset_config and "prompts" in dataset_config:
        _prompts_cache = dataset_config["prompts"]
        _benchmark_max_tokens = int(dataset_config.get("max_tokens", 80))

    # Parâmetros
    context_length = config.get("context_length", 2048)
    gpu_layers_cfg = config.get("gpu_layers", 33)
    gpu_mode = os.environ.get("PDC_GPU_MODE", "auto")
    desired_gpu_layers = 0 if gpu_mode == "cpu" else gpu_layers_cfg

    from llama_cpp import Llama

    # A CPU mantém o serviço disponível quando o backend CUDA falha.
    if desired_gpu_layers > 0:
        try:
            _model = Llama(
                model_path=_model_path,
                n_ctx=context_length,
                n_gpu_layers=desired_gpu_layers,
                verbose=False,
            )
            _n_gpu_layers = desired_gpu_layers
            print(f"[slm] Carregado com {desired_gpu_layers} layers na GPU")
            return
        except Exception as e:
            print(f"[slm] GPU falhou: {e}. Usando CPU...")

    _model = Llama(
        model_path=_model_path,
        n_ctx=context_length,
        n_gpu_layers=0,
        verbose=False,
    )
    _n_gpu_layers = 0
    print("[slm] Carregado em CPU")


@_serialized
def generate(prompt: str, max_tokens: int = 150) -> dict:
    """Gera texto e registra tokens, duração e tempo até o primeiro token."""
    if _model is None:
        raise RuntimeError("Modelo não carregado")

    formatted = _build_prompt(prompt, _chat_format)
    stop = _stop_tokens(_chat_format)

    first_token_time = None
    t_start = time.perf_counter()
    response_text = ""

    for token in _model(formatted, max_tokens=max_tokens, stream=True, stop=stop, echo=False):
        text = token["choices"][0]["text"]
        if text and first_token_time is None:
            first_token_time = time.perf_counter()
        response_text += text

    t_end = time.perf_counter()
    total_time = t_end - t_start
    tokens_generated = len(_model.tokenize(
        response_text.encode("utf-8"), add_bos=False, special=False)) if response_text else 0
    prompt_tokens = len(_model.tokenize(
        formatted.encode("utf-8"), add_bos=True, special=True))

    return {
        "response": response_text.strip(),
        "tokens_per_sec": round(tokens_generated / total_time, 2) if total_time > 0 else 0,
        "time_to_first_token_ms": round((first_token_time - t_start) * 1000, 2) if first_token_time else None,
        "total_tokens": tokens_generated,
        "prompt_tokens": prompt_tokens,
        "generation_sec": round(total_time, 4),
        "model": _model_name,
    }


def get_benchmark_fn():
    """Retorna uma carga determinística com os prompts configurados."""
    default_prompts = [
        "Explique o que é inteligência artificial em 3 frases.",
        "O que é edge computing?",
        "Descreva como funciona um motor elétrico.",
    ]
    prompts = _prompts_cache if _prompts_cache else default_prompts
    prompt_index = [0]

    def benchmark_inference():
        prompt = prompts[prompt_index[0] % len(prompts)]
        prompt_index[0] += 1
        return generate(prompt, max_tokens=_benchmark_max_tokens)

    return benchmark_inference


def get_device() -> str:
    return "cuda" if _n_gpu_layers > 0 else "cpu"


def get_model_name() -> str:
    return _model_name if _model is not None else "não carregado"


def get_loaded_config() -> dict:
    """Retorna info do modelo atualmente carregado."""
    return {
        "name": get_model_name(),
        "device": get_device(),
        "gpu_layers": _n_gpu_layers,
        "chat_format": _chat_format,
    }
