"""Perception stack for GPS-denied drone racing.

Provides gate detection, PnP localization, VIO state estimation,
and camera preprocessing for real-world deployment.
"""

from aigp.perception.gate_detector import GateDetection, GateDetector
from aigp.perception.image_preprocessor import ImagePreprocessor, PreprocessedFrames
from aigp.perception.pnp_localizer import PnPLocalizer, PnPResult
from aigp.perception.vio_ekf import StateEstimate, VIOStateEstimator

__all__ = [
    "GateDetection",
    "GateDetector",
    "ImagePreprocessor",
    "PnPLocalizer",
    "PnPResult",
    "PreprocessedFrames",
    "StateEstimate",
    "VIOStateEstimator",
]
