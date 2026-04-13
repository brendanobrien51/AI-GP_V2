"""PnP-based drone localization from gate bounding box detection.

Uses the Perspective-n-Point algorithm to recover the drone's 3D position
from a detected gate bounding box and known gate world coordinates.

Based on the Swift paper's approach: measurement covariance scales with
distance squared — closer gates yield more pixels and better accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PnPResult:
    """Result of a PnP localization attempt.

    Attributes:
        success: Whether the PnP solve converged.
        position: Drone position [x, y, z] in world/NED frame.
        covariance: 3x3 position covariance matrix.
        distance_m: Estimated distance to the gate.
    """

    success: bool
    position: np.ndarray      # (3,) float64
    covariance: np.ndarray    # (3, 3) float64
    distance_m: float


class PnPLocalizer:
    """Localize the drone using Perspective-n-Point on detected gates.

    Given a bounding box in image coordinates and the known 3D gate
    position, solves for the camera (drone) pose. Uses IPPE for planar
    targets with an ITERATIVE fallback.

    Args:
        gate_half_size_m: Half-width of the gate opening (metres).
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        fov_h_deg: Horizontal field of view (degrees).
        base_cov_scale: Base covariance multiplier (scales with distance^2).
    """

    def __init__(
        self,
        gate_half_size_m: float = 0.75,
        img_w: int = 640,
        img_h: int = 480,
        fov_h_deg: float = 120.0,
        base_cov_scale: float = 0.01,
    ) -> None:
        self._half = gate_half_size_m
        self._img_w = img_w
        self._img_h = img_h
        self._base_cov = base_cov_scale

        # Compute camera intrinsics from FOV
        fx = img_w / (2.0 * np.tan(np.radians(fov_h_deg / 2.0)))
        fy = fx  # square pixels
        cx = img_w / 2.0
        cy = img_h / 2.0

        self._camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)

        self._dist_coeffs = np.zeros(5, dtype=np.float64)

        # 3D model points of the gate corners in gate-local frame
        # Gate is a square opening in the YZ plane, centred at origin
        h = self._half
        self._gate_3d = np.array([
            [-h, -h, 0],  # top-left
            [ h, -h, 0],  # top-right
            [ h,  h, 0],  # bottom-right
            [-h,  h, 0],  # bottom-left
        ], dtype=np.float64)

    @property
    def camera_matrix(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        return self._camera_matrix.copy()

    def set_camera_calibration(
        self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        """Override default camera intrinsics with calibration data.

        Args:
            camera_matrix: 3x3 intrinsic matrix.
            dist_coeffs: Distortion coefficients (4, 5, 8, or 14 elements).
        """
        self._camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self._dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    def localize(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        gate_world_pos: np.ndarray,
        drone_quat_wxyz: np.ndarray | None = None,
    ) -> PnPResult:
        """Estimate drone position from a gate bounding box.

        Derives 4 image-space corner points from the bounding box, then
        solves PnP against the known gate 3D model.

        Args:
            bbox_xyxy: Gate bounding box (x1, y1, x2, y2) in pixels.
            gate_world_pos: Known gate position [x, y, z] in world frame.
            drone_quat_wxyz: Optional drone orientation for refinement.

        Returns:
            PnPResult with position, covariance, and distance.
        """
        x1, y1, x2, y2 = bbox_xyxy

        # Map bbox corners to image points (ordered to match gate_3d)
        image_points = np.array([
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2],  # bottom-left
        ], dtype=np.float64)

        # Try IPPE (best for planar targets), fall back to ITERATIVE
        success = False
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)

        for method in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
            ok, rv, tv = cv2.solvePnP(
                self._gate_3d, image_points,
                self._camera_matrix, self._dist_coeffs,
                flags=method,
            )
            if ok:
                rvec = rv.flatten()
                tvec = tv.flatten()
                success = True
                break

        if not success:
            return PnPResult(
                success=False,
                position=np.zeros(3),
                covariance=np.eye(3) * 1e6,
                distance_m=float("inf"),
            )

        # Convert camera-frame position to world frame
        # tvec is the gate position in camera frame
        # We want the camera (drone) position in world frame
        R, _ = cv2.Rodrigues(rvec)
        cam_pos_gate_frame = -R.T @ tvec  # camera pos in gate's local frame
        drone_pos = gate_world_pos + cam_pos_gate_frame  # world frame

        distance = float(np.linalg.norm(tvec))

        # Distance-scaled covariance (Swift paper):
        # Closer gate = more pixels = lower uncertainty
        cov_scale = self._base_cov * (distance**2)
        covariance = np.eye(3) * cov_scale

        return PnPResult(
            success=True,
            position=drone_pos,
            covariance=covariance,
            distance_m=distance,
        )
