"""
dataset_loader.py — Carregamento de datasets para benchmark.

Suporta dados inline (config), gerados sinteticamente, e auto-gerados
(TTS→ASR no container de áudio).
"""

import os
import random
from typing import Any

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


def load_dataset(dataset_config: dict, cache_dir: str = "/app/data/datasets") -> Any:
    """
    Carrega dataset conforme configuração.

    Modos suportados:
    - "inline": retorna lista de itens diretamente do config
    - "generate": gera dados sintéticos (ex: imagens numpy para vision)
    - "self_generate": retorna config para o container gerar via pipeline interno
    - URL: baixa arquivo e retorna path local

    Args:
        dataset_config: Dict com "source" e outros parâmetros específicos.
            Exemplos:
                {"source": "inline", "prompts": ["frase1", "frase2"]}
                {"source": "generate", "samples": 10, "type": "object_detection"}
                {"source": "self_generate", "phrases": ["frase para TTS"]}
                {"source": "https://example.com/data.json"}
        cache_dir: Diretório para cache de downloads.

    Retorna:
        list | dict | str dependendo do modo:
            inline → list de itens
            generate → list de arrays numpy (imagens)
            self_generate → dict com config e phrases
            URL → str path do arquivo local
    """
    source = dataset_config.get("source", "")
    os.makedirs(cache_dir, exist_ok=True)

    # --- Modo inline: dados direto do YAML ---
    if source == "inline":
        # Verifica campos comuns para dados inline
        for field in ("prompts", "phrases", "texts", "items", "data"):
            if field in dataset_config:
                return dataset_config[field]
        # Retorna tudo exceto "source" e metadados
        return {k: v for k, v in dataset_config.items()
                if k not in ("source", "type", "metrics", "max_tokens")}

    # --- Modo generate: gera dados sintéticos ---
    elif source == "generate":
        dataset_type = dataset_config.get("type", "")
        n_samples = dataset_config.get("samples", 10)

        if dataset_type == "object_detection" or "vision" in dataset_type:
            # Gera imagens sintéticas 640x640 RGB como arrays numpy
            if _NUMPY:
                images = []
                for i in range(n_samples):
                    # Imagem aleatória com alguns "objetos" (retângulos coloridos)
                    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    images.append(img)
                return images
            else:
                # Sem numpy: retorna lista de dicts descrevendo as imagens
                return [{"width": 640, "height": 640, "mode": "synthetic", "index": i}
                        for i in range(n_samples)]

        elif dataset_type == "asr" or "audio" in dataset_type:
            # Gera áudio sintético como arrays numpy (silêncio + ruído)
            if _NUMPY:
                sample_rate = 16000
                duration_sec = 2.0
                samples_audio = []
                for _ in range(n_samples):
                    # Ruído branco simulando fala
                    audio = np.random.normal(0, 0.1, int(sample_rate * duration_sec))
                    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
                    samples_audio.append(audio)
                return samples_audio
            else:
                return [{"sample_rate": 16000, "duration": 2.0, "mode": "synthetic"}
                        for _ in range(n_samples)]

        else:
            # Tipo genérico: gera strings de texto
            return [f"Dado sintético {i+1} para benchmark." for i in range(n_samples)]

    # --- Modo self_generate: container gera seus próprios dados via pipeline ---
    elif source == "self_generate":
        # Retorna config para que o container (ex: audio) gere TTS→ASR
        return {
            "mode": "self_generate",
            "phrases": dataset_config.get("phrases", [
                "Teste de reconhecimento de voz.",
                "Inferência em borda com modelos pequenos.",
            ]),
            "sample_rate": dataset_config.get("sample_rate", 16000),
        }

    # --- URL: baixa e cacheia ---
    elif source.startswith("http://") or source.startswith("https://"):
        if not _REQUESTS:
            raise ImportError("requests não instalado para baixar dataset")

        # Nome do arquivo a partir da URL
        filename = source.split("/")[-1].split("?")[0] or "dataset.bin"
        local_path = os.path.join(cache_dir, filename)

        if os.path.exists(local_path):
            print(f"[dataset_loader] Cache hit: {filename}")
            return local_path

        print(f"[dataset_loader] Baixando dataset: {source}")
        response = requests.get(source, timeout=120)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(response.content)

        print(f"[dataset_loader] Dataset salvo: {local_path}")
        return local_path

    else:
        # Fallback: trata source como lista de strings ou arquivo local
        if os.path.exists(source):
            return source
        return [f"Prompt de benchmark {i+1}." for i in range(10)]
