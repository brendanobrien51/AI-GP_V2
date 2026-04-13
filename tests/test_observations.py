"""Unit tests for observation utilities."""

import torch
import pytest

from aigp.utils.math_utils import (
    quat_rotate,
    quat_rotate_inverse,
    quat_to_gravity_body,
    world_to_body,
)


class TestQuatRotate:
    def test_identity_rotation(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity wxyz
        v = torch.tensor([[1.0, 2.0, 3.0]])
        result = quat_rotate(q, v)
        torch.testing.assert_close(result, v, atol=1e-6, rtol=1e-6)

    def test_inverse_roundtrip(self):
        q = torch.tensor([[0.707, 0.707, 0.0, 0.0]])  # 90 deg roll
        v = torch.tensor([[1.0, 0.0, 0.0]])
        rotated = quat_rotate(q, v)
        back = quat_rotate_inverse(q, rotated)
        torch.testing.assert_close(back, v, atol=1e-5, rtol=1e-5)


class TestGravityBody:
    def test_identity_returns_down(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        g = quat_to_gravity_body(q)
        # Identity rotation: gravity in body = gravity in world = [0, 0, -1]
        expected = torch.tensor([[0.0, 0.0, -1.0]])
        torch.testing.assert_close(g, expected, atol=1e-6, rtol=1e-6)


class TestWorldToBody:
    def test_zero_delta_at_same_pos(self):
        pos = torch.tensor([[5.0, 5.0, 5.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        result = world_to_body(pos, pos, quat)
        torch.testing.assert_close(result, torch.zeros(1, 3), atol=1e-6, rtol=1e-6)

    def test_forward_gate(self):
        # Gate is 10m ahead in world X
        gate = torch.tensor([[10.0, 0.0, 0.0]])
        drone_pos = torch.tensor([[0.0, 0.0, 0.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # facing +X
        result = world_to_body(gate, drone_pos, quat)
        assert result[0, 0].item() == pytest.approx(10.0, abs=1e-5)
