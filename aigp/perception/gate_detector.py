"""
YOLOv8 Gate Detector (V2)
=========================
Inference wrapper for a trained YOLOv8-nano gate detection model.

Improvements over V1:
  - Typed ``GateDetection`` dataclass replaces raw dict returns
  - Batch detection for multi-frame pipelines
  - GPU / CPU auto-selection with explicit override
  - Configurable confidence threshold (default 0.30)
  - Graceful fallback when model file is missing or ultralytics unavailable

Designed for real-time gate detection at 30 Hz on a 640x480 input frame
running on a ~100 TOPS onboard compute module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GateDetection:
    """Single gate detection result.

    Attributes:
        centroid_px: Gate centre ``(cx, cy)`` in pixel coordinates.
        confidence: Detection confidence score in ``[0, 1]``.
        bbox_xyxy: Bounding box ``(x1, y1, x2, y2)`` in pixels.
        area_fraction: Fraction of the image covered by the bounding box.
    """

    centroid_px: tuple[int, int]
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    area_fraction: float


class GateDetector:
    """YOLOv8-nano gate detection wrapper.

    Loads a trained ``.pt`` or ``.onnx`` model and exposes single-frame
    and batch inference. Falls back gracefully if the model file or the
    ``ultralytics`` package is unavailable.

    Args:
        model_path: Path to the YOLOv8 model weights.
        conf_threshold: Minimum confidence to accept a detection.
        device: Inference device -- ``"auto"``, ``"cuda"``, ``"cpu"``.
            ``"auto"`` selects CUDA when available.
        warmup: If ``True``, run a dummy forward pass during init to
            trigger JIT compilation / TensorRT build.
    """

    def __init__(
        self,
        model_path: str = "gate_detector.pt",
        conf_threshold: float = 0.30,
        device: str = "auto",
        warmup: bool = True,
    ) -> None:
        self._conf = conf_threshold
        self._model: object | None = None
        self._device = device

        path = Path(model_path)
        if not path.exists():
            logger.warning(
                "Model not found at %s -- detector disabled. "
                "detect() will return None until a valid model is loaded.",
                path,
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._model = YOLO(str(path))

            # Resolve device
            if device == "auto":
                try:
                    import torch
                    resolved = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    resolved = "cpu"
            else:
                resolved = device
            self._device = resolved

            # Warmup: first inference triggers CUDA graph / TRT compilation
            if warmup:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                self._model(dummy, verbose=False, device=resolved)

            logger.info(
                "GateDetector loaded %s on %s (conf >= %.2f)",
                path, resolved, conf_threshold,
            )
        except Exception:
            logger.exception("Failed to load YOLO model -- detector disabled")
            self._model = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether the underlying YOLO model is loaded and ready."""
        return self._model is not None

    @property
    def conf_threshold(self) -> float:
        """Current confidence threshold."""
        return self._conf

    @conf_threshold.setter
    def conf_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Confidence threshold must be in [0, 1], got {value}")
        self._conf = value

    # ------------------------------------------------------------------
    # Single-frame detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> GateDetection | None:
        """Run gate detection on a single BGR frame.

        Selects the highest-confidence detection above the threshold.

        Args:
            frame: BGR uint8 image, typically ``(480, 640, 3)``.

        Returns:
            A ``GateDetection`` for the best gate found, or ``None``
            if no detection exceeds the confidence threshold.
        """
        if self._model is None:
            return None

        try:
            results = self._model(
                frame, verbose=False, conf=self._conf, device=self._device,
            )
        except Exception:
            logger.debug("YOLO inference failed on frame", exc_info=True)
            return None

        return self._best_detection(results, frame.shape[:2])

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect_batch(
        self, frames: list[np.ndarray],
    ) -> list[GateDetection | None]:
        """Run gate detection on a batch of BGR frames.

        Args:
            frames: List of BGR uint8 images. All should share the same
                resolution for optimal throughput.

        Returns:
            List of ``GateDetection | None``, one per input frame.
        """
        if self._model is None or not frames:
            return [None] * len(frames)

        try:
            batch_results = self._model(
                frames, verbose=False, conf=self._conf, device=self._device,
            )
        except Exception:
            logger.debug("YOLO batch inference failed", exc_info=True)
            return [None] * len(frames)

        output: list[GateDetection | None] = []
        for idx, result in enumerate(batch_results):
            hw = frames[idx].shape[:2]
            det = self._best_detection([result], hw)
            output.append(det)
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_detection(
        self,
        results: list,  # ultralytics Results objects
        frame_hw: tuple[int, int],
    ) -> GateDetection | None:
        """Extract the highest-confidence detection from YOLO results.

        Args:
            results: List of ultralytics ``Results`` objects.
            frame_hw: ``(height, width)`` of the source frame.

        Returns:
            Best ``GateDetection`` or ``None``.
        """
        fh, fw = frame_hw
        frame_area = float(fw * fh) if fw * fh > 0 else 1.0

        best: GateDetection | None = None
        best_conf = 0.0

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                conf = float(box.conf[0])
                if conf < self._conf or conf <= best_conf:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area_frac = float((x2 - x1) * (y2 - y1)) / frame_area

                best_conf = conf
                best = GateDetection(
                    centroid_px=(cx, cy),
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    area_fraction=area_frac,
                )

        return best
