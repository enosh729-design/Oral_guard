"""
Unit tests for src/detector/fdi_mapper.py

FDI Convention recap (for a standard OPG/panoramic X-ray):
  - Upper jaw  → y < img_height/2
  - Lower jaw  → y >= img_height/2
  - Patient RIGHT (image LEFT)  → x < img_width/2  → Quadrants 1 (upper) & 4 (lower)
  - Patient LEFT  (image RIGHT) → x >= img_width/2 → Quadrants 2 (upper) & 3 (lower)

FDI numbers:
  Q1 (Upper Right): 11–18
  Q2 (Upper Left):  21–28
  Q3 (Lower Left):  31–38
  Q4 (Lower Right): 41–48
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path so imports work in CI
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.detector.fdi_mapper import (
    map_to_fdi,
    map_batch,
    fdi_label,
    _get_quadrant,
    _get_tooth_position,
    FDI_NAMES,
)

# ---------------------------------------------------------------------------
# Standard test image dimensions
# ---------------------------------------------------------------------------
W, H = 1000, 600


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def quadrant_of(fdi: int) -> int:
    return fdi // 10


# ---------------------------------------------------------------------------
# Quadrant detection tests
# ---------------------------------------------------------------------------

class TestGetQuadrant:
    def test_upper_right(self):
        # Upper half, image left → Quadrant 1
        assert _get_quadrant(200, 100, W, H) == 1

    def test_upper_left(self):
        # Upper half, image right → Quadrant 2
        assert _get_quadrant(800, 100, W, H) == 2

    def test_lower_left(self):
        # Lower half, image right → Quadrant 3
        assert _get_quadrant(800, 400, W, H) == 3

    def test_lower_right(self):
        # Lower half, image left → Quadrant 4
        assert _get_quadrant(200, 400, W, H) == 4

    def test_midline_x_upper(self):
        # Exactly on midline x — left half includes midline for Q1
        q = _get_quadrant(500, 100, W, H)
        assert q in (1, 2)

    def test_midline_y_left(self):
        # Exactly on midline y — should map to lower jaw
        q = _get_quadrant(200, 300, W, H)
        assert q in (1, 4)

    def test_normalised_coords(self):
        # With img_width=1, img_height=1 → normalised input
        assert _get_quadrant(0.2, 0.3, 1.0, 1.0) == 1
        assert _get_quadrant(0.8, 0.3, 1.0, 1.0) == 2
        assert _get_quadrant(0.8, 0.7, 1.0, 1.0) == 3
        assert _get_quadrant(0.2, 0.7, 1.0, 1.0) == 4


# ---------------------------------------------------------------------------
# Tooth position tests
# ---------------------------------------------------------------------------

class TestGetToothPosition:
    def test_position_clamped_to_1_8(self):
        # Position must always be in [1, 8]
        for x in [0, 50, 250, 499, 501, 750, 950, 1000]:
            for q in [1, 2, 3, 4]:
                pos = _get_tooth_position(x, 100, q, W, H)
                assert 1 <= pos <= 8, f"Position {pos} out of [1,8] for x={x}"

    def test_near_midline_is_position_1(self):
        # Tooth very close to the midline (x ≈ 500) should be position 1
        pos = _get_tooth_position(501, 100, 2, W, H)
        assert pos == 1

    def test_far_from_midline_is_position_8(self):
        # Tooth at the far edge (x=0 or x=1000) should be position 8
        pos_right = _get_tooth_position(5, 100, 1, W, H)
        assert pos_right == 8

    def test_increasing_distance_from_midline(self):
        # Positions should be monotonically non-decreasing as we move from midline
        positions = [
            _get_tooth_position(500 - 50 * i, 100, 1, W, H)
            for i in range(1, 9)
        ]
        # Each position should be >= previous (or same)
        for a, b in zip(positions, positions[1:]):
            assert b >= a, f"Non-monotonic positions: {positions}"


# ---------------------------------------------------------------------------
# map_to_fdi integration tests
# ---------------------------------------------------------------------------

class TestMapToFdi:
    def test_upper_right_quadrant(self):
        fdi = map_to_fdi(200, 100, 30, 40, W, H)
        assert quadrant_of(fdi) == 1, f"Expected Q1, got Q{quadrant_of(fdi)} (FDI {fdi})"

    def test_upper_left_quadrant(self):
        fdi = map_to_fdi(800, 100, 30, 40, W, H)
        assert quadrant_of(fdi) == 2, f"Expected Q2, got Q{quadrant_of(fdi)} (FDI {fdi})"

    def test_lower_left_quadrant(self):
        fdi = map_to_fdi(800, 450, 30, 40, W, H)
        assert quadrant_of(fdi) == 3, f"Expected Q3, got Q{quadrant_of(fdi)} (FDI {fdi})"

    def test_lower_right_quadrant(self):
        fdi = map_to_fdi(200, 450, 30, 40, W, H)
        assert quadrant_of(fdi) == 4, f"Expected Q4, got Q{quadrant_of(fdi)} (FDI {fdi})"

    def test_fdi_range_q1(self):
        fdi = map_to_fdi(200, 100, 30, 40, W, H)
        assert 11 <= fdi <= 18, f"Q1 FDI {fdi} out of [11,18]"

    def test_fdi_range_q2(self):
        fdi = map_to_fdi(800, 100, 30, 40, W, H)
        assert 21 <= fdi <= 28, f"Q2 FDI {fdi} out of [21,28]"

    def test_fdi_range_q3(self):
        fdi = map_to_fdi(800, 450, 30, 40, W, H)
        assert 31 <= fdi <= 38, f"Q3 FDI {fdi} out of [31,38]"

    def test_fdi_range_q4(self):
        fdi = map_to_fdi(200, 450, 30, 40, W, H)
        assert 41 <= fdi <= 48, f"Q4 FDI {fdi} out of [41,48]"

    def test_invalid_image_dimensions_raises(self):
        with pytest.raises(ValueError):
            map_to_fdi(100, 100, 30, 40, 0, 600)

    def test_invalid_negative_dimensions_raises(self):
        with pytest.raises(ValueError):
            map_to_fdi(100, 100, 30, 40, -10, 600)

    def test_incisor_near_midline(self):
        # x=490 (very close to midline 500) → should be position 1 → FDI 11
        fdi = map_to_fdi(490, 100, 20, 30, W, H)
        assert quadrant_of(fdi) == 1
        # position should be 1 (central incisor) — verify FDI ends in 1
        assert fdi % 10 == 1, f"Expected position 1, FDI={fdi}"

    def test_molar_far_from_midline(self):
        # x=50 (far left in image = patient's right = Q1 molar)
        fdi = map_to_fdi(50, 100, 30, 40, W, H)
        assert quadrant_of(fdi) == 1
        assert fdi % 10 >= 6, f"Expected molar (position 6-8), FDI={fdi}"

    def test_output_is_integer(self):
        fdi = map_to_fdi(300, 200, 40, 50, W, H)
        assert isinstance(fdi, int)

    def test_symmetric_upper_jaw(self):
        # Mirror images about midline → Q1 and Q2, different positions but same abs position
        fdi_left  = map_to_fdi(W // 2 - 100, 150, 30, 40, W, H)   # Q1
        fdi_right = map_to_fdi(W // 2 + 100, 150, 30, 40, W, H)   # Q2
        assert quadrant_of(fdi_left)  == 1
        assert quadrant_of(fdi_right) == 2
        # Same tooth position number
        assert fdi_left % 10 == fdi_right % 10, (
            f"Mirror teeth should have same position: {fdi_left} vs {fdi_right}"
        )


# ---------------------------------------------------------------------------
# Batch mapping tests
# ---------------------------------------------------------------------------

class TestMapBatch:
    def test_batch_adds_fdi_key(self):
        dets = [
            {"x_center": 200, "y_center": 100, "width": 30, "height": 40},
            {"x_center": 800, "y_center": 450, "width": 30, "height": 40},
        ]
        result = map_batch(dets, W, H)
        for det in result:
            assert "fdi" in det
            assert "quadrant" in det
            assert "position" in det

    def test_batch_fdi_ranges(self):
        dets = [
            {"x_center": 200, "y_center": 100, "width": 30, "height": 40},  # Q1
            {"x_center": 800, "y_center": 100, "width": 30, "height": 40},  # Q2
            {"x_center": 800, "y_center": 450, "width": 30, "height": 40},  # Q3
            {"x_center": 200, "y_center": 450, "width": 30, "height": 40},  # Q4
        ]
        result = map_batch(dets, W, H)
        assert 11 <= result[0]["fdi"] <= 18
        assert 21 <= result[1]["fdi"] <= 28
        assert 31 <= result[2]["fdi"] <= 38
        assert 41 <= result[3]["fdi"] <= 48

    def test_empty_batch(self):
        assert map_batch([], W, H) == []


# ---------------------------------------------------------------------------
# FDI name lookup tests
# ---------------------------------------------------------------------------

class TestFdiLabel:
    def test_known_fdi(self):
        assert fdi_label(11) == "UR Central Incisor"
        assert fdi_label(36) == "LL 1st Molar"
        assert fdi_label(48) == "LR 3rd Molar (Wisdom)"
        assert fdi_label(21) == "UL Central Incisor"

    def test_all_32_teeth_in_dict(self):
        expected = set()
        for q in [1, 2, 3, 4]:
            for p in range(1, 9):
                expected.add(q * 10 + p)
        assert expected == set(FDI_NAMES.keys()), (
            f"Missing FDI numbers: {expected - set(FDI_NAMES.keys())}"
        )

    def test_unknown_fdi_returns_fallback(self):
        label = fdi_label(99)
        assert "99" in label or "Unknown" in label

    def test_label_is_string(self):
        for fdi_num in FDI_NAMES:
            assert isinstance(fdi_label(fdi_num), str)
