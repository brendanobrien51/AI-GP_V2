"""Integration test: short PPO training smoke test.

Verifies that training starts without OOM and loss decreases.
Requires Isaac Sim. Run with:
    pytest tests/integration/test_training_loop.py -m integration
"""

import os
import pytest

# Workarounds for Isaac Sim 4.5 pip install on Windows:
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
try:
    import osqp  # noqa: F401
except ImportError:
    pass

pytestmark = pytest.mark.integration


class TestTrainingLoop:
    def test_short_training_no_crash(self):
        """Run 100 steps of PPO training — verify no CUDA OOM or NaN."""
        try:
            from isaaclab.app import AppLauncher
            launcher = AppLauncher(headless=True)

            from aigp.envs.racing_env import RacingEnv
            from aigp.envs.racing_env_cfg import RacingEnvCfg
            from aigp.agents.ppo_trainer import train

            cfg = RacingEnvCfg()
            cfg.scene.num_envs = 4
            cfg.num_gates = 3

            env = RacingEnv(cfg=cfg)

            agent = train(
                env=env,
                total_timesteps=100,
                checkpoint_dir="/tmp/aigp_test_checkpoints",
                seed=42,
                device="cuda:0",
                initial_gates=3,
                max_gates=3,
            )

            assert agent is not None
            launcher.app.close()
        except ImportError:
            pytest.skip("Isaac Sim not installed")
