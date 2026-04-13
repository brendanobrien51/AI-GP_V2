"""Registry that maps track-type names to their generator functions.

Usage::

    from aigp.track.track_registry import get_track, list_track_types

    layout = get_track("zigzag", num_gates=5, seed=42)
    print(list_track_types())   # ['zigzag', 'split_s', 'circular']
"""

from __future__ import annotations

from typing import Any, Callable

from aigp.track.track_generator import (
    generate_circular,
    generate_split_s,
    generate_zigzag,
)
from aigp.track.track_types import TrackLayout

# Type alias for a track generator callable.
TrackGeneratorFn = Callable[..., TrackLayout]

# Internal registry -- maps canonical name to generator function.
_REGISTRY: dict[str, TrackGeneratorFn] = {
    "zigzag": generate_zigzag,
    "split_s": generate_split_s,
    "circular": generate_circular,
}


def register_track_type(name: str, generator: TrackGeneratorFn) -> None:
    """Register a custom track generator under *name*.

    Args:
        name: Canonical track-type name (lowercase, no spaces).
        generator: A callable with the signature
            ``(num_gates, *, seed=None, device="cpu", **kwargs) -> TrackLayout``.

    Raises:
        ValueError: If *name* is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Track type '{name}' is already registered")
    _REGISTRY[name] = generator


def get_track(
    track_type: str,
    num_gates: int,
    seed: int | None = None,
    *,
    device: str = "cpu",
    **kwargs: Any,
) -> TrackLayout:
    """Build a track layout by name.

    Args:
        track_type: One of the registered track type names (see
            :func:`list_track_types`).
        num_gates: Number of gates to generate (>= 2).
        seed: Random seed for reproducibility.
        device: Torch device string forwarded to the generator.
        **kwargs: Extra keyword arguments forwarded to the generator
            (e.g. ``corridor_width``, ``radius``).

    Returns:
        The generated :class:`TrackLayout`.

    Raises:
        KeyError: If *track_type* is not registered.
    """
    if track_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown track type '{track_type}'. Available: {available}"
        )
    generator = _REGISTRY[track_type]
    return generator(num_gates, seed=seed, device=device, **kwargs)


def list_track_types() -> list[str]:
    """Return sorted list of registered track type names.

    Returns:
        Sorted list of track type name strings.
    """
    return sorted(_REGISTRY)
