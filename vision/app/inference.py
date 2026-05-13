"""
inference.py — Inferência YOLOv8 para detecção de objetos.

Carrega o modelo YOLOv8n e fornece função de predição.
Detecta automaticamente CUDA e usa GPU se disponível.
"""

import io
import os
import sys
import random
import numpy as np
from typing import List, Optional

# Adiciona /app ao path para importar shared/
sys.path.insert(0, "/app")

from shared.model_loader import get_or_download


# Variáveis globais do modelo (carregadas uma vez no startup)
_model = None
_device = "cpu"
_model_path = None


def load_model(config: dict) -> None:
    """
    Carrega o modelo YOLOv8 na memória.

    Detecta automaticamente se CUDA está disponível.
    Se GPU falhar, faz fallback para CPU.

    Args:
        config: Dict de configuração do modelo (do models.yaml).
    """
    global _model, _device, _model_path

    # Baixa modelo se não estiver em cache
    _model_path = get_or_download(config, cache_dir="/app/data/models")
    print(f"[vision] Modelo em: {_model_path}")

    # Detecta CUDA
    try:
        import torch
        if torch.cuda.is_available() and os.environ.get("PDC_GPU_MODE") != "cpu":
            _device = "cuda"
            print(f"[vision] GPU detectada: {torch.cuda.get_device_name(0)}")
        else:
            _device = "cpu"
            print("[vision] Usando CPU")
    except Exception:
        _device = "cpu"
        print("[vision] torch não disponível para detectar CUDA, usando CPU")

    # Carrega modelo via ultralytics
    try:
        from ultralytics import YOLO
        _model = YOLO(_model_path)
        _model.to(_device)
        print(f"[vision] Modelo YOLOv8 carregado em {_device.upper()}")
    except Exception as e:
        # Fallback para CPU se GPU falhar
        if _device == "cuda":
            print(f"[vision] Falha ao carregar em GPU: {e}. Tentando CPU...")
            _device = "cpu"
            from ultralytics import YOLO
            _model = YOLO(_model_path)
            _model.to("cpu")
            print("[vision] Modelo carregado em CPU (fallback)")
        else:
            raise


def predict(image_bytes: bytes) -> List[dict]:
    """
    Executa detecção de objetos em uma imagem.

    Args:
        image_bytes: Imagem em bytes (JPEG, PNG, etc.) ou array numpy.

    Retorna:
        Lista de detecções, cada uma com:
            bbox: [x1, y1, x2, y2] em pixels
            class_name: nome da classe detectada
            confidence: confiança (0.0 a 1.0)
            class_id: ID numérico da classe
    """
    global _model

    if _model is None:
        raise RuntimeError("Modelo não carregado. Chame load_model() primeiro.")

    # Converte bytes para array numpy se necessário
    if isinstance(image_bytes, bytes):
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Não foi possível decodificar a imagem")
    else:
        # Assume que já é um array numpy
        img = image_bytes

    # Executa inferência (verbose=False suprime prints do ultralytics)
    results = _model(img, verbose=False)

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            # box.xyxy: coordenadas [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()
            detections.append({
                "bbox": [round(c, 2) for c in coords],
                "class_name": result.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "class_id": int(box.cls[0]),
            })

    return detections


def get_benchmark_fn():
    """
    Retorna uma função sem argumentos para uso nos benchmarks.

    A função gera uma imagem sintética 640x640 RGB e executa predict().
    Isso simula uma câmera gerando frames continuamente.

    Retorna:
        callable: função () -> list[dict]
    """
    def benchmark_inference():
        """Gera imagem sintética e executa detecção."""
        # Imagem aleatória 640x640 com 3 canais (RGB)
        fake_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        return predict(fake_image)

    return benchmark_inference


def get_device() -> str:
    """Retorna o dispositivo sendo usado ('cuda' ou 'cpu')."""
    return _device


def get_model_name() -> str:
    """Retorna o nome do modelo carregado."""
    if _model_path:
        return os.path.basename(_model_path)
    return "não carregado"
