"""
Camera Image Preprocessing Pipeline
====================================
Handles conversion from 12MP raw camera frames to the two downstream
resolution targets used by the perception stack:

  1. **Policy input** -- 80x80 RGB for CNN policy inference
  2. **Detection input** -- 640x480 BGR for YOLOv8 gate detection

Applies optional camera undistortion using a calibration matrix before
downsampling. Thread-safe: all state is either immutable after __init__
or computed locally per call.

Designed for real-time operation on a ~100 TOPS compute module with a
12MP FPV camera (4000x3000 BGR uint8 input).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessedFrames:
    """Container for the two output resolution paths.

    Attributes:
        policy: 80x80 RGB uint8 array for CNN policy inference.
        detection: 640x480 BGR uint8 array for YOLOv8 gate detection.
    """

    policy: np.ndarray      # (80, 80, 3) RGB uint8
    detection: np.ndarray   # (480, 640, 3) BGR uint8


class ImagePreprocessor:
    """Thread-safe camera preprocessing pipeline.

    Converts 12MP raw frames to policy and detection resolution targets
    with optional lens undistortion.

    Args:
        camera_matrix: 3x3 camera intrinsic matrix. ``None`` disables
            undistortion (identity pass-through).
        dist_coeffs: Distortion coefficients for ``cv2.undistort``.
            ``None`` disables undistortion.
        raw_size: Expected raw frame dimensions as ``(width, height)``.
        policy_size: Output size for policy inference ``(width, height)``.
        detector_size: Output size for YOLOv8 detection ``(width, height)``.
        interpolation: OpenCV interpolation flag for downsampling.
            ``cv2.INTER_AREA`` is best for large downsamples.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        raw_size: tuple[int, int] = (4000, 3000),
        policy_size: tuple[int, int] = (80, 80),
        detector_size: tuple[int, int] = (640, 480),
        interpolation: int = cv2.INTER_AREA,
    ) -> None:
        self._policy_size = policy_size
        self._detector_size = detector_size
        self._raw_w, self._raw_h = raw_size
        self._interp = interpolation

        # Pre-compute undistortion maps if calibration is provided.
        # These are immutable numpy arrays -- safe to share across threads.
        self._undistort_map1: np.ndarray | None = None
        self._undistort_map2: np.ndarray | None = None

        if camera_matrix is not None and dist_coeffs is not None:
            K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
            D = np.asarray(dist_coeffs, dtype=np.float64).ravel()
            # Compute optimal new camera matrix to avoid cropping
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                K, D, (self._raw_w, self._raw_h), alpha=0.0,
            )
            self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(
                K, D, None, new_K, (self._raw_w, self._raw_h), cv2.CV_16SC2,
            )

        # Lock only protects lazy-initialized state (currently none),
        # but kept for future extension (e.g., runtime calibration swap).
        self._lock = threading.Lock()

    def process(self, frame: np.ndarray) -> PreprocessedFrames:
        """Run the full preprocessing pipeline on a single raw frame.

        Args:
            frame: Raw camera image, expected shape ``(H, W, 3)`` BGR uint8.
                Nominal input is 4000x3000 but any resolution is accepted.

        Returns:
            A ``PreprocessedFrames`` containing both output paths.

        Raises:
            ValueError: If the input frame does not have 3 channels.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel BGR image, got shape {frame.shape}"
            )

        # Step 1: Undistort (if calibration available)
        undistorted = self._undistort(frame)

        # Step 2a: Detection path -- 640x480 BGR
        detection = cv2.resize(
            undistorted, self._detector_size, interpolation=self._interp,
        )

        # Step 2b: Policy path -- 80x80 RGB
        # Downsample to detection size first (already computed), then
        # further resize to policy size. Two-stage resize is faster than
        # a single 50x downsample and preserves more spatial information.
        policy_bgr = cv2.resize(
            detection, self._policy_size, interpolation=self._interp,
        )
        policy_rgb = cv2.cvtColor(policy_bgr, cv2.COLOR_BGR2RGB)

        return PreprocessedFrames(policy=policy_rgb, detection=detection)

    def process_detection_only(self, frame: np.ndarray) -> np.ndarray:
        """Produce only the 640x480 detection frame (skip policy resize).

        Args:
            frame: Raw camera image ``(H, W, 3)`` BGR uint8.

        Returns:
            Resized frame ``(480, 640, 3)`` BGR uint8.
        """
        undistorted = self._undistort(frame)
        return cv2.resize(
            undistorted, self._detector_size, interpolation=self._interp,
        )

    def process_policy_only(self, frame: np.ndarray) -> np.ndarray:
        """Produce only the 80x80 policy frame (skip detection resize).

        Args:
            frame: Raw camera image ``(H, W, 3)`` BGR uint8.

        Returns:
            Resized frame ``(80, 80, 3)`` RGB uint8.
        """
        undistorted = self._undistort(frame)
        # Two-stage: first to detector resolution, then to policy
        intermediate = cv2.resize(
            undistorted, self._detector_size, interpolation=self._interp,
        )
        policy_bgr = cv2.resize(
            intermediate, self._policy_size, interpolation=self._interp,
        )
        return cv2.cvtColor(policy_bgr, cv2.COLOR_BGR2RGB)

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        """Apply lens undistortion if calibration maps are available.

        Uses pre-computed remap tables for maximum throughput.
        """
        if self._undistort_map1 is not None and self._undistort_map2 is not None:
            return cv2.remap(
                frame,
                self._undistort_map1,
                self._undistort_map2,
                cv2.INTER_LINEAR,
            )
        return frame

    @property
    def policy_size(self) -> tuple[int, int]:
        """Policy output size as ``(width, height)``."""
        return self._policy_size

    @property
    def detector_size(self) -> tuple[int, int]:
        """Detection output size as ``(width, height)``."""
        return self._detector_size

    def update_calibration(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        """Hot-swap camera calibration at runtime (thread-safe).

        Args:
            camera_matrix: New 3x3 intrinsic matrix.
            dist_coeffs: New distortion coefficients.
        """
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        D = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, D, (self._raw_w, self._raw_h), alpha=0.0,
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (self._raw_w, self._raw_h), cv2.CV_16SC2,
        )
        with self._lock:
            self._undistort_map1 = map1
            self._undistort_map2 = map2
