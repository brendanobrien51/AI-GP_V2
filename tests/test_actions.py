"""Unit tests for CTBR action processing."""

import torch
import pytest

from aigp.envs.actions import scale_ctbr_actions, clamp_actions


class TestScaleActions:
    def test_min_thrust_at_neg_one(self):
        raw = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
        scaled = scale_ctbr_actions(raw)
        assert scaled[0, 0].item() == pytest.approx(0.0)

    def test_max_thrust_at_pos_one(self):
        raw = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        scaled = scale_ctbr_actions(raw)
        assert scaled[0, 0].item() == pytest.approx(19.62)

    def test_hover_thrust_at_zero(self):
        raw = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        scaled = scale_ctbr_actions(raw)
        assert scaled[0, 0].item() == pytest.approx(9.81)

    def test_body_rates_at_extremes(self):
        raw = torch.tensor([[0.0, 1.0, -1.0, 0.5]])
        scaled = scale_ctbr_actions(raw)
        assert scaled[0, 1].item() == pytest.approx(6.0)
        assert scaled[0, 2].item() == pytest.approx(-6.0)
        assert scaled[0, 3].item() == pytest.approx(3.0)

    def test_batch_shape(self):
        raw = torch.randn(16, 4)
        scaled = scale_ctbr_actions(raw)
        assert scaled.shape == (16, 4)


class TestClampActions:
    def test_clamps_to_range(self):
        raw = torch.tensor([[2.0, -3.0, 0.5, 1.5]])
        clamped = clamp_actions(raw)
        assert clamped[0, 0].item() == 1.0
        assert clamped[0, 1].item() == -1.0
        assert clamped[0, 2].item() == 0.5
        assert clamped[0, 3].item() == 1.0
