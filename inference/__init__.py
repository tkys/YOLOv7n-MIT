"""
YOLOv7n MIT Edition - 推論パッケージ
CPU最適化されたONNX Runtime推論システム
"""

__version__ = "1.0.0"
__author__ = "YOLOv7n MIT Edition Team"

# 主要クラスのインポート
from .single_image import SingleImageInference
from .batch_images import BatchImageInference
from .logger import (
    StructuredLogger,
    get_logger,
    inference_logger,
    training_logger,
    api_logger,
    benchmark_logger
)

__all__ = [
    "SingleImageInference",
    "BatchImageInference", 
    "StructuredLogger",
    "get_logger",
    "inference_logger",
    "training_logger",
    "api_logger",
    "benchmark_logger"
]