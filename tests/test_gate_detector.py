"""Unit tests for YOLOv8 gate detector."""

import numpy as np
import pytest

from aigp.perception.gate_detector import GateDetector, GateDetection


class TestGateDetector:
    def test_unavailable_when_model_missing(self):
        detector = GateDetector(model_path="/nonexistent/model.pt", warmup=False)
        assert not detector.available

    def test_detect_returns_none_when_unavailable(self):
        detector = GateDetector(model_path="/nonexistent/model.pt", warmup=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result is None

    def test_batch_returns_nones_when_unavailable(self):
        detector = GateDetector(model_path="/nonexistent/model.pt", warmup=False)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
        results = detector.detect_batch(frames)
        assert len(results) == 3
        assert all(r is None for r in results)

    def test_conf_threshold_property(self):
        detector = GateDetector(model_path="/nonexistent/model.pt", warmup=False)
        assert detector.conf_threshold == 0.30
        detector.conf_threshold = 0.5
        assert detector.conf_threshold == 0.5

    def test_invalid_conf_threshold_raises(self):
        detector = GateDetector(model_path="/nonexistent/model.pt", warmup=False)
        with pytest.raises(ValueError):
            detector.conf_threshold = 1.5


class TestGateDetection:
    def test_frozen_dataclass(self):
        det = GateDetection(
            centroid_px=(320, 240),
            confidence=0.95,
            bbox_xyxy=(100, 80, 540, 400),
            area_fraction=0.47,
        )
        assert det.centroid_px == (320, 240)
        assert det.confidence == 0.95
        with pytest.raises(AttributeError):
            det.confidence = 0.5  # type: ignore[misc]
