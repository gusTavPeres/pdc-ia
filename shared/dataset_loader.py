"""Carregamento e geração de datasets de benchmark."""

import os
from typing import Any

from shared.model_loader import get_or_download

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


def load_dataset(dataset_config: dict, cache_dir: str = "/app/data/datasets") -> Any:
    """Carrega dados inline, sintéticos, locais ou remotos."""
    if not isinstance(dataset_config, dict):
        raise ValueError("a configuração do dataset deve ser um objeto")
    source = dataset_config.get("source", "")
    if not isinstance(source, str):
        raise ValueError("dataset.source deve ser uma string")
    source = source.strip()
    os.makedirs(cache_dir, exist_ok=True)

    if source == "inline":
        for field in ("prompts", "phrases", "texts", "items", "data"):
            if field in dataset_config:
                return dataset_config[field]
        raise ValueError("dataset inline não contém prompts, phrases, texts, items ou data")

    elif source == "generate":
        dataset_type = str(dataset_config.get("type", "")).strip().lower()
        raw_samples = dataset_config.get("samples", 10)
        try:
            n_samples = int(raw_samples)
        except (TypeError, ValueError):
            raise ValueError("dataset.samples deve ser inteiro") from None
        if isinstance(raw_samples, bool) or (
                isinstance(raw_samples, float) and not raw_samples.is_integer()):
            raise ValueError("dataset.samples deve ser inteiro")
        if not 1 <= n_samples <= 10_000:
            raise ValueError("dataset.samples deve estar entre 1 e 10000")

        if dataset_type == "object_detection" or "vision" in dataset_type:
            seed = int(dataset_config.get("seed", 42))
            width = int(dataset_config.get("width", 640))
            height = int(dataset_config.get("height", 640))
            if not 1 <= width <= 8192 or not 1 <= height <= 8192:
                raise ValueError("dimensões do dataset devem estar entre 1 e 8192")
            if n_samples * width * height * 3 > 512 * 1024 * 1024:
                raise ValueError("dataset sintético excede o limite de 512 MB")
            if _NUMPY:
                rng = np.random.default_rng(seed)
                return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                        for _ in range(n_samples)]
            else:
                return [{"width": width, "height": height, "mode": "synthetic", "index": i}
                        for i in range(n_samples)]

        elif dataset_type == "asr" or "audio" in dataset_type:
            sample_rate = int(dataset_config.get("sample_rate", 16000))
            duration_sec = float(dataset_config.get("duration_sec", 2.0))
            if not 1 <= sample_rate <= 384_000 or not 0 < duration_sec <= 3_600:
                raise ValueError("sample_rate ou duration_sec fora do limite")
            if n_samples * sample_rate * duration_sec * 4 > 512 * 1024 * 1024:
                raise ValueError("dataset sintético excede o limite de 512 MB")
            if _NUMPY:
                rng = np.random.default_rng(int(dataset_config.get("seed", 42)))
                samples_audio = []
                for _ in range(n_samples):
                    audio = rng.normal(0, 0.1, int(sample_rate * duration_sec))
                    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
                    samples_audio.append(audio)
                return samples_audio
            else:
                return [{"sample_rate": sample_rate, "duration": duration_sec,
                         "mode": "synthetic"}
                        for _ in range(n_samples)]

        else:
            return [f"Dado sintético {i+1} para benchmark." for i in range(n_samples)]

    elif source == "self_generate":
        return {
            "mode": "self_generate",
            "phrases": dataset_config.get("phrases", [
                "Teste de reconhecimento de voz.",
                "Inferência em borda com modelos pequenos.",
            ]),
            "sample_rate": dataset_config.get("sample_rate", 16000),
        }

    elif source.startswith("http://") or source.startswith("https://"):
        filename = source.split("/")[-1].split("?")[0] or "dataset.bin"
        return get_or_download(
            {"url": source, "filename": filename, "sha256": dataset_config.get("sha256")},
            cache_dir=cache_dir,
            artifact_label="dataset",
        )

    else:
        if source and os.path.exists(source):
            return source
        raise ValueError(f"origem de dataset inválida ou inexistente: {source or '<vazia>'}")
