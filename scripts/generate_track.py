"""CLI for generating and visualizing racing track layouts.

Usage:
    python scripts/generate_track.py --type zigzag --gates 5
    python scripts/generate_track.py --type circular --gates 8 --seed 42
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate racing track layouts")
    parser.add_argument("--type", default="zigzag", choices=["zigzag", "split_s", "circular"])
    parser.add_argument("--gates", type=int, default=5, help="Number of gates")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--jitter", type=float, default=0.0, help="Position jitter (metres)")
    parser.add_argument("--output", default=None, help="Save to JSON file")
    args = parser.parse_args()

    from aigp.track.track_registry import get_track

    track = get_track(args.type, num_gates=args.gates, seed=args.seed)

    print(f"Track: {track.name} ({track.num_gates} gates)")
    print("-" * 50)
    for i, gate in enumerate(track.gates):
        print(f"  Gate {i}: pos=[{gate.x:.2f}, {gate.y:.2f}, {gate.z:.2f}] rot={gate.rotation_deg:.1f}°")

    if args.output:
        data = {
            "name": track.name,
            "num_gates": track.num_gates,
            "gates": [
                {
                    "position": gate.position,
                    "rotation_deg": gate.rotation_deg,
                }
                for gate in track.gates
            ],
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
