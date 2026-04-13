"""GPU VRAM monitoring for 12GB RTX 4080 budget management."""

from __future__ import annotations

import torch


def get_vram_usage() -> dict[str, float]:
    """Get current GPU VRAM usage in MB.

    Returns:
        Dictionary with allocated, reserved, and free VRAM in MB.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}

    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    total = torch.cuda.get_device_properties(0).total_mem / (1024**2)
    free = total - reserved

    return {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "free_mb": round(free, 1),
        "total_mb": round(total, 1),
    }


def log_vram_usage(label: str = "") -> None:
    """Print current VRAM usage with an optional label."""
    usage = get_vram_usage()
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}VRAM: {usage['allocated_mb']:.0f}MB allocated, "
        f"{usage['reserved_mb']:.0f}MB reserved, "
        f"{usage['free_mb']:.0f}MB free / {usage['total_mb']:.0f}MB total"
    )


def check_vram_budget(budget_mb: float = 10000.0) -> bool:
    """Check if current VRAM usage is within budget.

    Args:
        budget_mb: Maximum allowed VRAM in MB. Default 10GB (leaving 2GB headroom on 12GB).

    Returns:
        True if within budget.
    """
    usage = get_vram_usage()
    return usage["reserved_mb"] <= budget_mb
