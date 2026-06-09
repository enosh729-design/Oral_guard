"""
OralGuard — FDI Tooth Numbering Mapper
Maps bounding box coordinates on a panoramic X-ray to FDI tooth numbers.

FDI World Dental Federation Notation:
  Quadrant 1 (Upper Right): 11–18
  Quadrant 2 (Upper Left):  21–28
  Quadrant 3 (Lower Left):  31–38
  Quadrant 4 (Lower Right): 41–48

On a standard OPG/panoramic X-ray:
  - Upper jaw = top half of image (y < midline)
  - Lower jaw = bottom half (y >= midline)
  - Patient's RIGHT = image LEFT  → Quadrants 1 (upper) and 4 (lower)
  - Patient's LEFT  = image RIGHT → Quadrants 2 (upper) and 3 (lower)

Tooth position (1–8) is determined by x-distance from the midline,
with 1 being the central incisor (closest to midline) and 8 being
the third molar (furthest from midline).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_quadrant(
    x_center: float,
    y_center: float,
    img_width: float,
    img_height: float,
) -> int:
    """
    Return the FDI quadrant (1–4) based on normalised center coordinates.

    Args:
        x_center:   Horizontal center of bounding box (pixels or 0–1 normalised).
        y_center:   Vertical center of bounding box   (pixels or 0–1 normalised).
        img_width:  Image width  (use 1.0 if coords are already normalised).
        img_height: Image height (use 1.0 if coords are already normalised).

    Returns:
        Quadrant number 1–4.
    """
    x_norm = x_center / img_width
    y_norm = y_center / img_height

    upper_jaw = y_norm < 0.5   # top half of OPG = upper jaw
    # OPG is a mirror view: patient's right teeth appear on image LEFT
    patient_right = x_norm < 0.5

    if upper_jaw and patient_right:
        return 1   # Upper Right
    elif upper_jaw and not patient_right:
        return 2   # Upper Left
    elif not upper_jaw and not patient_right:
        return 3   # Lower Left
    else:
        return 4   # Lower Right


def _get_tooth_position(
    x_center: float,
    y_center: float,
    quadrant: int,
    img_width: float,
    img_height: float,
) -> int:
    """
    Return the tooth position (1–8) within its quadrant.

    Position 1 = central incisor (nearest the midline)
    Position 8 = third molar     (farthest from midline)

    The OPG midline is at x = img_width / 2.
    Distance from midline is used to assign position buckets.

    Args:
        x_center:   Horizontal center of bounding box (pixels).
        y_center:   Vertical center of bounding box   (pixels, unused here).
        quadrant:   FDI quadrant (1–4) as returned by _get_quadrant.
        img_width:  Image width in pixels.
        img_height: Image height in pixels (unused here).

    Returns:
        Tooth position integer 1–8.
    """
    midline_x = img_width / 2.0
    distance = abs(x_center - midline_x)
    # Normalize distance to [0, 1] relative to half the image width
    half_width = img_width / 2.0
    norm_dist = min(distance / half_width, 1.0)

    # Divide the half-width into 8 equal buckets (tooth positions 1–8)
    position = int(norm_dist * 8) + 1
    position = max(1, min(position, 8))   # clamp to [1, 8]
    return position


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_to_fdi(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: float,
    img_height: float,
) -> int:
    """
    Map a YOLO bounding box to an FDI tooth number.

    Accepts pixel-space coordinates (not normalised).
    If your YOLO output is normalised (0–1 range), pass
    img_width=1.0 and img_height=1.0.

    Args:
        x_center:   Horizontal center of bounding box in pixels.
        y_center:   Vertical center of bounding box in pixels.
        width:      Width of bounding box in pixels  (unused in mapping,
                    kept for API completeness).
        height:     Height of bounding box in pixels (unused in mapping).
        img_width:  Full image width in pixels.
        img_height: Full image height in pixels.

    Returns:
        FDI tooth number (e.g. 11, 36, 47).

    Example:
        >>> fdi = map_to_fdi(120, 80, 30, 40, 500, 300)
        >>> assert 11 <= fdi <= 18 or 21 <= fdi <= 28
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(
            f"img_width and img_height must be positive, got "
            f"({img_width}, {img_height})"
        )

    quadrant = _get_quadrant(x_center, y_center, img_width, img_height)
    position = _get_tooth_position(x_center, y_center, quadrant, img_width, img_height)
    fdi_number = quadrant * 10 + position
    return fdi_number


def map_batch(
    detections: list[dict],
    img_width: float,
    img_height: float,
) -> list[dict]:
    """
    Map a list of YOLO detection dicts to FDI numbers.

    Each detection dict must have keys: x_center, y_center, width, height.
    The function adds the key 'fdi' to each dict and returns the list.

    Args:
        detections: List of dicts with bounding box fields.
        img_width:  Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Same list with 'fdi' and 'quadrant' fields added to each dict.
    """
    for det in detections:
        quad = _get_quadrant(
            det["x_center"], det["y_center"], img_width, img_height
        )
        pos = _get_tooth_position(
            det["x_center"], det["y_center"], quad, img_width, img_height
        )
        det["quadrant"] = quad
        det["position"] = pos
        det["fdi"] = quad * 10 + pos
    return detections


# ---------------------------------------------------------------------------
# FDI label utility
# ---------------------------------------------------------------------------

FDI_NAMES: dict[int, str] = {
    # Quadrant 1 — Upper Right
    11: "UR Central Incisor",   12: "UR Lateral Incisor",
    13: "UR Canine",            14: "UR 1st Premolar",
    15: "UR 2nd Premolar",      16: "UR 1st Molar",
    17: "UR 2nd Molar",         18: "UR 3rd Molar (Wisdom)",
    # Quadrant 2 — Upper Left
    21: "UL Central Incisor",   22: "UL Lateral Incisor",
    23: "UL Canine",            24: "UL 1st Premolar",
    25: "UL 2nd Premolar",      26: "UL 1st Molar",
    27: "UL 2nd Molar",         28: "UL 3rd Molar (Wisdom)",
    # Quadrant 3 — Lower Left
    31: "LL Central Incisor",   32: "LL Lateral Incisor",
    33: "LL Canine",            34: "LL 1st Premolar",
    35: "LL 2nd Premolar",      36: "LL 1st Molar",
    37: "LL 2nd Molar",         38: "LL 3rd Molar (Wisdom)",
    # Quadrant 4 — Lower Right
    41: "LR Central Incisor",   42: "LR Lateral Incisor",
    43: "LR Canine",            44: "LR 1st Premolar",
    45: "LR 2nd Premolar",      46: "LR 1st Molar",
    47: "LR 2nd Molar",         48: "LR 3rd Molar (Wisdom)",
}


def fdi_label(fdi_number: int) -> str:
    """Human-readable label for an FDI tooth number."""
    return FDI_NAMES.get(fdi_number, f"Unknown tooth {fdi_number}")


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W, H = 1000, 600   # sample panoramic X-ray dimensions
    samples = [
        # (x_center, y_center, description, expected_quadrant)
        (200,  120, "Upper Right molar",    1),
        (700,  120, "Upper Left molar",     2),
        (700,  400, "Lower Left molar",     3),
        (200,  400, "Lower Right molar",    4),
        (480,  100, "Upper Right incisor",  1),
        (520,  100, "Upper Left incisor",   2),
        (520,  450, "Lower Left incisor",   3),
        (480,  450, "Lower Right incisor",  4),
    ]
    print(f"{'Description':<30} {'FDI':>5} {'Label'}")
    print("-" * 70)
    for xc, yc, desc, expected_q in samples:
        fdi = map_to_fdi(xc, yc, 30, 40, W, H)
        label = fdi_label(fdi)
        quad = fdi // 10
        status = "✅" if quad == expected_q else "❌"
        print(f"{desc:<30} {fdi:>5}  {label}  {status}")
