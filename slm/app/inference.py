"""
inference.py — Inferência SLM via llama-cpp-python.

Suporta múltiplos modelos GGUF com troca em runtime via load_model().
Formatos de prompt suportados: tinyllama, chatml (Qwen2.5, Phi-3, etc.)
"""

import os
import sys
import time
import random
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


def _build_prompt(user_text: str, chat_format: str) -> str:
    """
    Formata o prompt no estilo do modelo.

    TinyLlama usa seu próprio template de chat.
    ChatML é usado por Qwen2.5, Phi-3, etc.

    Args:
        user_text: Pergunta/instrução do usuário.
        chat_format: "tinyllama" ou "chatml"

    Retorna:
        Prompt formatado pronto para enviar ao modelo.
    """
    if chat_format == "chatml":
        # Formato ChatML: <|im_start|>role\ncontent<|im_end|>
        return (
            "<|im_start|>system\nVocê é um assistente útil que responde em português.<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
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
    else:
        return ["</s>", "<|user|>", "<|system|>"]


def load_model(config: dict, dataset_config: Optional[dict] = None) -> None:
    """
    Carrega modelo GGUF. Pode ser chamado novamente para trocar de modelo.

    Baixa o arquivo se necessário, detecta GPU e carrega via llama-cpp-python.
    Se GPU falhar, usa CPU como fallback.

    Args:
        config: Dict do models.yaml (url, gpu_layers, context_length, chat_format, ...)
        dataset_config: Dict do datasets.yaml com prompts para benchmark.
    """
    global _model, _model_name, _model_path, _n_gpu_layers, _chat_format, _prompts_cache

    # Descarrega modelo anterior se houver
    if _model is not None:
        print(f"[slm] Descarregando modelo anterior: {_model_name}")
        del _model
        _model = None

    # Baixa se necessário
    _model_path = get_or_download(config, cache_dir="/app/data/models")
    _model_name = os.path.basename(_model_path)
    _chat_format = config.get("chat_format", "tinyllama")
    print(f"[slm] Carregando: {_model_name} (formato: {_chat_format})")

    # Prompts do dataset
    if dataset_config and "prompts" in dataset_config:
        _prompts_cache = dataset_config["prompts"]

    # Parâmetros
    context_length = config.get("context_length", 2048)
    gpu_layers_cfg = config.get("gpu_layers", 33)
    gpu_mode = os.environ.get("PDC_GPU_MODE", "auto")
    desired_gpu_layers = 0 if gpu_mode == "cpu" else gpu_layers_cfg

    from llama_cpp import Llama

    # Tenta GPU, fallback CPU
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
    print(f"[slm] Carregado em CPU")


def generate(prompt: str, max_tokens: int = 150) -> dict:
    """
    Gera texto a partir de um prompt.

    Usa o formato de prompt correto para o modelo carregado.

    Args:
        prompt: Texto de entrada.
        max_tokens: Máximo de tokens a gerar.

    Retorna:
        dict com response, tokens_per_sec, time_to_first_token_ms, total_tokens
    """
    if _model is None:
        raise RuntimeError("Modelo não carregado")

    formatted = _build_prompt(prompt, _chat_format)
    stop = _stop_tokens(_chat_format)

    first_token_time = None
    t_start = time.perf_counter()
    tokens_generated = 0
    response_text = ""

    for token in _model(formatted, max_tokens=max_tokens, stream=True, stop=stop, echo=False):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        response_text += token["choices"][0]["text"]
        tokens_generated += 1

    t_end = time.perf_counter()
    total_time = t_end - t_start

    return {
        "response": response_text.strip(),
        "tokens_per_sec": round(tokens_generated / total_time, 2) if total_time > 0 else 0,
        "time_to_first_token_ms": round((first_token_time - t_start) * 1000, 2) if first_token_time else None,
        "total_tokens": tokens_generated,
        "prompt_tokens": len(formatted) // 4,
        "model": _model_name,
    }


def get_benchmark_fn():
    """Retorna closure para benchmark: sorteia prompt e gera."""
    default_prompts = [
        "Explique o que é inteligência artificial em 3 frases.",
        "O que é edge computing?",
        "Descreva como funciona um motor elétrico.",
    ]
    prompts = _prompts_cache if _prompts_cache else default_prompts

    def benchmark_inference():
        return generate(random.choice(prompts), max_tokens=80)

    return benchmark_inference


def get_device() -> str:
    return "cuda" if _n_gpu_layers > 0 else "cpu"


def get_model_name() -> str:
    return _model_name


def get_loaded_config() -> dict:
    """Retorna info do modelo atualmente carregado."""
    return {
        "name": _model_name,
        "device": get_device(),
        "gpu_layers": _n_gpu_layers,
        "chat_format": _chat_format,
    }
