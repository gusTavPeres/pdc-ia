"""Carga e inferência de modelos YOLO."""

import os
import sys
import threading
from typing import List, Optional

import numpy as np

sys.path.insert(0, "/app")

from shared.cpu_config import cpu_threads as _cpu_threads
from shared.dataset_loader import load_dataset
from shared.model_loader import get_or_download

_model = None
_device = "cpu"
_model_path = None
_model_lock = threading.RLock()


def load_model(config: dict) -> None:
    """Carrega o modelo em CUDA quando disponível, com fallback para CPU."""
    global _model, _device, _model_path

    with _model_lock:
        model_path = get_or_download(config, cache_dir="/app/data/models")
        print(f"[vision] Modelo em: {model_path}")

        device = "cpu"
        try:
            import torch
            torch.set_num_threads(_cpu_threads())  # usa todos os nucleos, nao ~metade
            if torch.cuda.is_available() and os.environ.get("PDC_GPU_MODE") != "cpu":
                device = "cuda"
                print(f"[vision] GPU detectada: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

        config_dir = os.environ.get("YOLO_CONFIG_DIR")
        if config_dir:
            try:
                os.makedirs(config_dir, exist_ok=True)
            except OSError:
                fallback_dir = "/tmp/pdc-ultralytics"
                os.makedirs(fallback_dir, exist_ok=True)
                os.environ["YOLO_CONFIG_DIR"] = fallback_dir

        from ultralytics import YOLO
        try:
            model = YOLO(model_path)
            model.to(device)
        except Exception as exc:
            if device != "cuda":
                raise
            print(f"[vision] Falha ao carregar em GPU: {exc}. Tentando CPU...")
            device = "cpu"
            model = YOLO(model_path)
            model.to(device)

        _model = model
        _model_path = model_path
        _device = device
        print(f"[vision] Modelo carregado em {device.upper()}")


def predict(image_bytes: bytes) -> List[dict]:
    """Retorna caixas, classes e confiança para uma imagem."""
    with _model_lock:
        if _model is None:
            raise RuntimeError("Modelo não carregado. Chame load_model() primeiro.")

        if isinstance(image_bytes, bytes):
            import cv2
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Não foi possível decodificar a imagem")
        else:
            img = image_bytes

        results = _model(img, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [round(c, 2) for c in coords],
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "class_id": int(box.cls[0]),
                })
        return detections


def get_benchmark_fn(dataset_config: Optional[dict] = None):
    """Monta uma inferência cíclica sobre o dataset configurado."""
    images = load_dataset(dataset_config or {
        "source": "generate",
        "type": "object_detection",
        "samples": 10,
        "seed": 42,
    })
    if not isinstance(images, list) or not images:
        raise ValueError("Dataset de visão sem amostras utilizáveis")
    image_index = [0]

    def benchmark_inference():
        image = images[image_index[0] % len(images)]
        image_index[0] += 1
        detections = predict(image)
        confidence = (
            sum(item["confidence"] for item in detections) / len(detections)
            if detections else 0.0
        )
        return {
            "detections_count": len(detections),
            "avg_confidence": round(confidence, 4),
        }

    return benchmark_inference


def get_device() -> str:
    """Retorna o dispositivo sendo usado ('cuda' ou 'cpu')."""
    return _device


def get_model_name() -> str:
    """Retorna o nome do modelo carregado."""
    if _model is not None and _model_path:
        return os.path.basename(_model_path)
    return "não carregado"
