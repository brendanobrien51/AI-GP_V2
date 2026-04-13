"""Training curriculum manager for progressive difficulty scaling.

Starts with few gates and adds more as the agent demonstrates mastery.
Tracks gate-pass success rate over a rolling window to decide promotions.
"""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)


class CurriculumManager:
    """Manages gate count curriculum during RL training.

    Promotion rule: when gate-pass rate exceeds ``promotion_threshold``
    over the last ``window_size`` episodes, add one gate.

    Args:
        initial_gates: Starting number of gates.
        max_gates: Maximum number of gates.
        promotion_threshold: Success rate (0-1) required to add a gate.
        window_size: Number of episodes to evaluate over.
    """

    def __init__(
        self,
        initial_gates: int = 3,
        max_gates: int = 8,
        promotion_threshold: float = 0.80,
        window_size: int = 50,
    ) -> None:
        self.current_gates = initial_gates
        self.max_gates = max_gates
        self.threshold = promotion_threshold
        self._window: deque[float] = deque(maxlen=window_size)

    @property
    def success_rate(self) -> float:
        """Current rolling success rate."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def at_max(self) -> bool:
        """Whether we've reached maximum gate count."""
        return self.current_gates >= self.max_gates

    def record_episode(self, gates_passed: int, total_gates: int) -> None:
        """Record an episode's gate-pass ratio.

        Args:
            gates_passed: Number of gates successfully passed.
            total_gates: Total gates in the course.
        """
        rate = gates_passed / max(total_gates, 1)
        self._window.append(rate)

    def should_promote(self) -> bool:
        """Check if the agent should be promoted to more gates.

        Returns:
            True if promotion criteria met and not at max.
        """
        if self.at_max:
            return False
        if len(self._window) < self._window.maxlen:
            return False
        return self.success_rate >= self.threshold

    def promote(self) -> int:
        """Add one gate and reset the evaluation window.

        Returns:
            New gate count.
        """
        if self.at_max:
            return self.current_gates

        self.current_gates += 1
        self._window.clear()
        logger.info(
            "Curriculum promoted to %d gates (max %d)",
            self.current_gates, self.max_gates,
        )
        return self.current_gates
