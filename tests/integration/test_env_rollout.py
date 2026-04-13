"""Integration test: full environment rollout with random policy.

Requires Isaac Sim to be installed. Run with:
    pytest tests/integration/test_env_rollout.py -m integration
"""

import os
import pytest

# Workarounds for Isaac Sim 4.5 pip install on Windows:
# 1. Accept EULA non-interactively
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
# 2. Pre-import osqp to avoid DLL conflict with Omniverse kernel
try:
    import osqp  # noqa: F401
except ImportError:
    pass

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def env():
    """Create a small racing environment for testing."""
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True)

        from aigp.envs.racing_env import RacingEnv
        from aigp.envs.racing_env_cfg import RacingEnvCfg

        cfg = RacingEnvCfg()
        cfg.scene.num_envs = 4
        cfg.num_gates = 3
        racing_env = RacingEnv(cfg=cfg)
        yield racing_env
        launcher.app.close()
    except ImportError:
        pytest.skip("Isaac Sim not installed")


class TestEnvRollout:
    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset()
        assert "policy" in obs
        assert obs["policy"].shape == (4, 13)

    def test_step_returns_correct_shapes(self, env):
        import torch
        obs, info = env.reset()
        action = torch.zeros(4, 4, device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs["policy"].shape == (4, 13)
        assert reward.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)

    def test_100_step_rollout_no_crash(self, env):
        import torch
        obs, info = env.reset()
        for _ in range(100):
            action = torch.randn(4, 4, device=env.device) * 0.1
            obs, reward, terminated, truncated, info = env.step(action)
            assert not torch.isnan(reward).any()
            assert not torch.isnan(obs["policy"]).any()
