"""Unit tests for VIO + EKF state estimator."""

import numpy as np
import pytest

from aigp.perception.vio_ekf import AeroResidualModel, StateEstimate, VIOStateEstimator


class TestAeroResidualModel:
    def test_zero_coeff_returns_zero(self):
        model = AeroResidualModel(drag_coeff=0.0)
        drag = model.predict(np.array([10.0, 5.0, 0.0]))
        np.testing.assert_array_equal(drag, np.zeros(3))

    def test_drag_proportional_to_speed_squared(self):
        model = AeroResidualModel(drag_coeff=0.01)
        v1 = np.array([10.0, 0.0, 0.0])
        v2 = np.array([20.0, 0.0, 0.0])
        d1 = np.linalg.norm(model.predict(v1))
        d2 = np.linalg.norm(model.predict(v2))
        assert d2 == pytest.approx(d1 * 4.0, rel=0.01)

    def test_drag_in_velocity_direction(self):
        model = AeroResidualModel(drag_coeff=0.01)
        vel = np.array([3.0, 4.0, 0.0])
        drag = model.predict(vel)
        direction = vel / np.linalg.norm(vel)
        drag_dir = drag / np.linalg.norm(drag)
        np.testing.assert_array_almost_equal(direction, drag_dir)


class TestVIOStateEstimator:
    def test_initialization(self):
        est = VIOStateEstimator()
        assert not est.initialized
        est.initialize(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        assert est.initialized
        state = est.get_state()
        np.testing.assert_array_almost_equal(state.position, [1.0, 2.0, 3.0])

    def test_predict_produces_drift(self):
        est = VIOStateEstimator(accel_noise=0.1, gyro_noise=0.05)
        est.initialize(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        # Run many predict steps with zero IMU
        for _ in range(100):
            est.predict(
                accel_raw=np.array([0.0, 0.0, -9.81]),  # gravity in body frame
                gyro_raw=np.zeros(3),
                dt=0.004,  # 250 Hz
            )
        state = est.get_state()
        # Position should drift due to noise
        assert np.linalg.norm(state.position) > 0.0

    def test_pnp_update_reduces_covariance(self):
        est = VIOStateEstimator()
        est.initialize(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        # Predict to increase uncertainty
        for _ in range(50):
            est.predict(np.array([0.0, 0.0, -9.81]), np.zeros(3), 0.004)

        pre_cov_trace = np.trace(est.get_state().position_covariance)

        # PnP update should reduce covariance
        est.update_pnp(
            position=np.array([0.1, 0.05, 0.02]),
            covariance=np.eye(3) * 0.01,
        )

        post_cov_trace = np.trace(est.get_state().position_covariance)
        assert post_cov_trace < pre_cov_trace

    def test_state_estimate_speed(self):
        state = StateEstimate(
            position=np.zeros(3),
            velocity=np.array([3.0, 4.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            position_covariance=np.eye(3),
        )
        assert state.speed == pytest.approx(5.0)
