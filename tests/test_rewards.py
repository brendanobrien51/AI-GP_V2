"""Unit tests for reward functions."""

import torch
import pytest

from aigp.envs.rewards import (
    progress_reward,
    gate_pass_bonus,
    smoothness_penalty,
    collision_penalty,
    compute_total_reward,
)


@pytest.fixture
def device():
    return torch.device("cpu")


class TestProgressReward:
    def test_positive_when_closer(self, device):
        prev = torch.tensor([10.0, 8.0], device=device)
        curr = torch.tensor([8.0, 6.0], device=device)
        r = progress_reward(curr, prev)
        assert (r > 0).all()

    def test_negative_when_farther(self, device):
        prev = torch.tensor([5.0], device=device)
        curr = torch.tensor([8.0], device=device)
        r = progress_reward(curr, prev)
        assert (r < 0).all()

    def test_zero_when_same(self, device):
        d = torch.tensor([5.0], device=device)
        r = progress_reward(d, d)
        assert r.item() == pytest.approx(0.0)


class TestGatePassBonus:
    def test_bonus_when_passed(self, device):
        passed = torch.tensor([True, False, True], device=device)
        r = gate_pass_bonus(passed, bonus=10.0)
        assert r[0].item() == 10.0
        assert r[1].item() == 0.0
        assert r[2].item() == 10.0


class TestSmoothnessReward:
    def test_zero_when_same_action(self, device):
        action = torch.ones(4, 4, device=device)
        r = smoothness_penalty(action, action)
        assert (r == 0).all()

    def test_negative_when_different(self, device):
        a1 = torch.zeros(2, 4, device=device)
        a2 = torch.ones(2, 4, device=device)
        r = smoothness_penalty(a1, a2, scale=-0.01)
        assert (r < 0).all()


class TestCollisionPenalty:
    def test_penalty_on_collision(self, device):
        collided = torch.tensor([True, False], device=device)
        r = collision_penalty(collided, penalty=-5.0)
        assert r[0].item() == -5.0
        assert r[1].item() == 0.0
