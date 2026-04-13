"""Unit tests for procedural track generation."""

import math

import pytest

from aigp.track.track_generator import (
    MIN_GATE_SEPARATION_M,
    generate_circular,
    generate_split_s,
    generate_zigzag,
)


class TestZigzag:
    def test_generates_correct_gate_count(self):
        track = generate_zigzag(5, seed=42)
        assert track.num_gates == 5

    def test_minimum_gates(self):
        track = generate_zigzag(2, seed=42)
        assert track.num_gates == 2

    def test_rejects_single_gate(self):
        with pytest.raises(ValueError, match="num_gates must be >= 2"):
            generate_zigzag(1)

    def test_gate_separation(self):
        track = generate_zigzag(6, seed=42)
        for i in range(track.num_gates - 1):
            dist = track.gates[i].distance_to(track.gates[i + 1])
            assert dist >= MIN_GATE_SEPARATION_M, f"Gates {i}-{i+1} too close: {dist:.2f}m"

    def test_gates_face_next_gate(self):
        track = generate_zigzag(5, seed=42)
        for i in range(track.num_gates - 1):
            g0, g1 = track.gates[i], track.gates[i + 1]
            expected_angle = math.degrees(math.atan2(g1.y - g0.y, g1.x - g0.x))
            assert abs(g0.rotation_deg - expected_angle) < 1.0

    def test_seed_reproducibility(self):
        t1 = generate_zigzag(5, seed=123)
        t2 = generate_zigzag(5, seed=123)
        for g1, g2 in zip(t1.gates, t2.gates):
            assert g1.position == g2.position


class TestSplitS:
    def test_generates_correct_gate_count(self):
        track = generate_split_s(5, seed=42)
        assert track.num_gates == 5

    def test_descending_altitude(self):
        track = generate_split_s(5, seed=42)
        # First gate should be higher than last (descending spiral)
        assert track.gates[0].z > track.gates[-1].z

    def test_gate_separation(self):
        track = generate_split_s(6, seed=42)
        for i in range(track.num_gates - 1):
            dist = track.gates[i].distance_to(track.gates[i + 1])
            assert dist >= MIN_GATE_SEPARATION_M


class TestCircular:
    def test_generates_correct_gate_count(self):
        track = generate_circular(6, seed=42)
        assert track.num_gates == 6

    def test_constant_altitude(self):
        track = generate_circular(6, altitude=3.0, seed=42)
        for gate in track.gates:
            assert abs(gate.z - 3.0) < 0.01

    def test_eccentricity(self):
        circle = generate_circular(8, eccentricity=0.0, seed=42)
        ellipse = generate_circular(8, eccentricity=0.4, seed=42)
        # Ellipse should have different X/Y spread
        circle_xs = [g.x for g in circle.gates]
        ellipse_xs = [g.x for g in ellipse.gates]
        # Both should have same X range (semi-major) but different Y range
        assert max(circle_xs) == pytest.approx(max(ellipse_xs), abs=0.1)

    def test_max_gates(self):
        track = generate_circular(8, seed=42)
        assert track.num_gates == 8
