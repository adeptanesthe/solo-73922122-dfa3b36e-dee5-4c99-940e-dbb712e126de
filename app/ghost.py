"""Afterimage (ghosting) verification from screen photographs.

After a guide-screen image is replaced, faint traces of the previous image may
remain.  An inspector uploads the PNG that was actually delivered in the
previous round ("old") plus one or two grayscale screen photographs, all
aligned and the same size as the currently published page ("new").

Physical model
--------------
For a pixel whose new grayscale is ``x`` and old grayscale is ``o`` the screen
really shows the blend ``x + r * (o - x)`` where ``r`` in ``[0, 1]`` is the
shared residual (afterimage) ratio.  Each photograph adds its own exposure
gain ``a`` in ``[0.5, 2]`` and offset ``b``::

    photo_gray ~= a * (x + r * (o - x)) + b

Every calibration point and every pixel inside the region of interest must be
reproduced within a grayscale tolerance of 2.

For a fixed ``r`` each observation is one affine half-plane pair in ``(a, b)``,
so feasibility is a two-variable linear-feasibility problem solved exactly
(up to floating point) by minimising the convex gap between the strongest
lower and upper bounds on ``b``.  Feasible ``r`` values form a closed
interval: with ``s = a*r`` every constraint is linear in ``(a, b, s)`` and the
projective map to ``r = s/a`` preserves convexity, so the projection onto the
``r`` axis is an interval.  Per-photo intervals therefore intersect in an
interval; an empty intersection means the photographs have no common
explanation and the evidence conflicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .imaging import InputError

# --------------------------------------------------------------------------
# contract constants
# --------------------------------------------------------------------------
MIN_CALIBRATION_POINTS = 2
MAX_CALIBRATION_POINTS = 8
MIN_CALIBRATION_GRAYS = 2
MAX_GRAY_LEVELS_FOR_POINTS = 4
MAX_ROI_SIDE = 16

GAIN_MIN = 0.5
GAIN_MAX = 2.0
RATIO_MIN = 0.0
RATIO_MAX = 1.0
GRAY_TOLERANCE = 2.0
REFLASH_THRESHOLD = 0.20

# Numerical tolerances.  The feasibility gap is measured in grayscale units;
# anything below this is treated as touching the feasible region.
GAP_EPS = 1e-7
R_EPS = 1e-8
# Golden-section search over an interval of length 1: 40 iterations shrink it
# below 1e-8, tighter than the grayscale tolerance even for narrow feasible
# regions.  Bisection over [0, 1]: 34 iterations locate endpoints within ~1e-10.
GOLDEN_ITERATIONS = 40
BISECT_ITERATIONS = 34


class EvidenceConflictError(RuntimeError):
    """No residual ratio explains every photograph simultaneously."""

    def __init__(self, message: str, photo_intervals: list[dict | None]) -> None:
        super().__init__(message)
        self.photo_intervals = photo_intervals


@dataclass(frozen=True)
class CalibrationPoint:
    x: int
    y: int
    gray: int  # new gray == old gray at this pixel


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int
    changed_pixels: int  # pixels where old gray != new gray
    gray_levels: int  # distinct new grayscale values inside the region


@dataclass(frozen=True)
class PhotoFit:
    index: int
    gain_a: float
    offset_b: float
    r_min: float
    r_max: float
    max_abs_residual: float  # worst |predicted - photo| over all constraints


@dataclass(frozen=True)
class GhostVerdict:
    decision: str  # "reflash" | "no_reflash" | "reshoot"
    r_min: float
    r_max: float
    calibration_points: list[CalibrationPoint]
    roi: Region
    photos: list[PhotoFit]


# --------------------------------------------------------------------------
# calibration points and region of interest
# --------------------------------------------------------------------------
def select_calibration_points(
    new_gray: bytes, old_gray: bytes, width: int, height: int
) -> list[CalibrationPoint]:
    """Pick 2-8 pixels where old and new grayscales agree.

    The points must span at least two grayscale values so a photograph's gain
    and offset can be identified independently of the residual ratio.  The
    selection is deterministic: gray levels are spread across the available
    value range and, when a level has enough pixels, two spatially separated
    representatives are kept.
    """
    groups: dict[int, list[int]] = {}
    for index, (new_value, old_value) in enumerate(zip(new_gray, old_gray)):
        if new_value == old_value:
            groups.setdefault(new_value, []).append(index)
    if len(groups) < MIN_CALIBRATION_GRAYS:
        raise InputError(
            "need at least two grayscale values at pixels where old and new "
            "images agree, found "
            f"{len(groups)}; the page cannot be calibrated for an afterimage check"
        )

    levels = sorted(groups)
    level_count = min(MAX_GRAY_LEVELS_FOR_POINTS, len(levels))
    chosen_levels = [
        levels[round(position * (len(levels) - 1) / (level_count - 1))]
        for position in range(level_count)
    ]

    points: list[CalibrationPoint] = []
    for level in chosen_levels:
        indices = groups[level]
        picks = {indices[len(indices) // 2]}
        if len(indices) >= 4:
            picks.add(indices[len(indices) // 4])
        for index in sorted(picks):
            points.append(
                CalibrationPoint(
                    x=index % width, y=index // width, gray=level
                )
            )
            if len(points) >= MAX_CALIBRATION_POINTS:
                return sorted(points, key=lambda point: (point.y, point.x))
    if len(points) < MIN_CALIBRATION_POINTS:  # pragma: no cover - defensive
        raise InputError("not enough calibration pixels")
    return sorted(points, key=lambda point: (point.y, point.x))


def _integral_image(flags: list[int], width: int, height: int) -> list[list[int]]:
    integral = [[0] * (width + 1) for _ in range(height + 1)]
    for y in range(height):
        running = 0
        base = y * width
        for x in range(width):
            running += flags[base + x]
            integral[y + 1][x + 1] = integral[y][x + 1] + running
    return integral


def _rect_sum(
    integral: list[list[int]], x: int, y: int, width: int, height: int
) -> int:
    return (
        integral[y + height][x + width]
        - integral[y][x + width]
        - integral[y + height][x]
        + integral[y][x]
    )


def select_roi(
    new_gray: bytes, old_gray: bytes, width: int, height: int
) -> Region:
    """Choose a <= 16x16 window that contains grayscale variation.

    A useful window holds at least two distinct new grayscale values and at
    least one pixel whose old and new grayscales differ; the latter is what
    makes the residual ratio observable.  Windows are ranked by the number of
    changed pixels, then by grayscale diversity, with row-major order as the
    deterministic tie-breaker.
    """
    side = min(MAX_ROI_SIDE, width, height)
    changed_flags = [
        1 if new_value != old_value else 0
        for new_value, old_value in zip(new_gray, old_gray)
    ]
    integral = _integral_image(changed_flags, width, height)

    best: Region | None = None
    best_score: tuple[int, int] | None = None
    for y in range(0, height - side + 1):
        for x in range(0, width - side + 1):
            changed = _rect_sum(integral, x, y, side, side)
            if changed < 1:
                continue
            values = set()
            base = y * width
            for row in range(side):
                offset = base + row * width + x
                values.update(new_gray[offset : offset + side])
            if len(values) < MIN_CALIBRATION_GRAYS:
                continue
            score = (changed, len(values))
            if best_score is None or score > best_score:
                best_score = score
                best = Region(
                    x=x,
                    y=y,
                    width=side,
                    height=side,
                    changed_pixels=changed,
                    gray_levels=len(values),
                )
    if best is None:
        raise InputError(
            "no region of up to 16x16 pixels contains both grayscale variation "
            "and a pixel that changed between the old and new images; the page "
            "offers nothing measurable for an afterimage check"
        )
    return best


# --------------------------------------------------------------------------
# per-photo linear feasibility
# --------------------------------------------------------------------------
def _observations(
    photo_gray: bytes,
    new_gray: bytes,
    old_gray: bytes,
    width: int,
    points: list[CalibrationPoint],
    roi: Region,
) -> list[tuple[int, int, int]]:
    """Return ``(new_gray, old_minus_new, photo_gray)`` for every constraint."""
    observations: list[tuple[int, int, int]] = []
    for point in points:
        index = point.y * width + point.x
        new_value = new_gray[index]
        observations.append(
            (new_value, old_gray[index] - new_value, photo_gray[index])
        )
    for y in range(roi.y, roi.y + roi.height):
        base = y * width
        for x in range(roi.x, roi.x + roi.width):
            index = base + x
            new_value = new_gray[index]
            observations.append(
                (new_value, old_gray[index] - new_value, photo_gray[index])
            )
    return observations


def _bounds_at(
    observations: list[tuple[int, int, int]], ratio: float, gain: float
) -> tuple[float, float]:
    """Strongest lower/upper bounds on ``b`` for a fixed ``(r, a)``."""
    lower = -math.inf
    upper = math.inf
    for new_value, delta, photo_value in observations:
        blend = new_value + ratio * delta
        shifted = photo_value - gain * blend
        lower = max(lower, shifted - GRAY_TOLERANCE)
        upper = min(upper, shifted + GRAY_TOLERANCE)
    return lower, upper


def _min_gap(
    observations: list[tuple[int, int, int]], ratio: float
) -> tuple[float, float]:
    """Minimise the convex gap ``lower(a) - upper(a)`` over ``a``.

    ``lower`` is the maximum of affine functions (convex), ``upper`` the
    minimum (concave), so their difference is convex and golden-section search
    converges to its global minimum.
    """

    def gap(gain: float) -> float:
        lower, upper = _bounds_at(observations, ratio, gain)
        return lower - upper

    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = GAIN_MIN, GAIN_MAX
    probe_left = right - golden * (right - left)
    probe_right = left + golden * (right - left)
    gap_left = gap(probe_left)
    gap_right = gap(probe_right)
    for _ in range(GOLDEN_ITERATIONS):
        if gap_left < gap_right:
            right = probe_right
            probe_right = probe_left
            gap_right = gap_left
            probe_left = right - golden * (right - left)
            gap_left = gap(probe_left)
        else:
            left = probe_left
            probe_left = probe_right
            gap_left = gap_right
            probe_right = left + golden * (right - left)
            gap_right = gap(probe_right)
    gain = (probe_left + probe_right) / 2.0
    return gain, gap(gain)


def is_feasible(
    observations: list[tuple[int, int, int]], ratio: float
) -> tuple[bool, float]:
    gain, gap = _min_gap(observations, ratio)
    return gap <= GAP_EPS, gain


def _min_gap_over_ratio(
    observations: list[tuple[int, int, int]],
) -> tuple[float, float]:
    """Minimise the feasibility gap over ``r`` in ``[0, 1]``.

    Feasibility in ``r`` is a convex projection, hence the minimum gap is a
    unimodal (quasi-convex) function of ``r`` and golden-section search finds
    its global minimum without relying on a hand-picked grid, so arbitrarily
    narrow feasible intervals are still detected.
    """
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = RATIO_MIN, RATIO_MAX
    probe_left = right - golden * (right - left)
    probe_right = left + golden * (right - left)
    gap_left = _min_gap(observations, probe_left)[1]
    gap_right = _min_gap(observations, probe_right)[1]
    for _ in range(GOLDEN_ITERATIONS):
        if gap_left < gap_right:
            right = probe_right
            probe_right = probe_left
            gap_right = gap_left
            probe_left = right - golden * (right - left)
            gap_left = _min_gap(observations, probe_left)[1]
        else:
            left = probe_left
            probe_left = probe_right
            gap_left = gap_right
            probe_right = left + golden * (right - left)
            gap_right = _min_gap(observations, probe_right)[1]
    ratio = (probe_left + probe_right) / 2.0
    gain, gap = _min_gap(observations, ratio)
    return ratio, gap


def photo_ratio_interval(
    observations: list[tuple[int, int, int]],
) -> tuple[float, float, float] | None:
    """Return ``(r_min, r_max, r_seed)`` for one photo.

    ``r_seed`` is the ratio with the smallest feasibility gap; it is the
    safest point at which to report the representative ``(a, b)`` fit.
    """
    seed, gap = _min_gap_over_ratio(observations)
    if gap > GAP_EPS:
        return None

    low, high = RATIO_MIN, seed
    for _ in range(BISECT_ITERATIONS):
        middle = (low + high) / 2.0
        if is_feasible(observations, middle)[0]:
            high = middle
        else:
            low = middle
    ratio_low = high

    low, high = seed, RATIO_MAX
    for _ in range(BISECT_ITERATIONS):
        middle = (low + high) / 2.0
        if is_feasible(observations, middle)[0]:
            low = middle
        else:
            high = middle
    ratio_high = low
    return ratio_low, ratio_high, seed


def fit_at_ratio(
    observations: list[tuple[int, int, int]], ratio: float
) -> tuple[float, float, float]:
    """Pick a representative feasible ``(a, b)`` and report the worst residual."""
    gain, gap = _min_gap(observations, ratio)
    if gap > GAP_EPS:
        raise EvidenceConflictError(
            "internal: ratio became infeasible while fitting exposure", []
        )
    lower, upper = _bounds_at(observations, ratio, gain)
    offset = (lower + upper) / 2.0
    worst = 0.0
    for new_value, delta, photo_value in observations:
        predicted = gain * (new_value + ratio * delta) + offset
        worst = max(worst, abs(predicted - photo_value))
    return gain, offset, worst


# --------------------------------------------------------------------------
# top-level verification
# --------------------------------------------------------------------------
def _decision(ratio_low: float, ratio_high: float) -> str:
    # Every feasible ratio reaches the threshold: recommend re-flashing.
    if ratio_low + R_EPS >= REFLASH_THRESHOLD:
        return "reflash"
    # Every feasible ratio is strictly below the threshold.
    if ratio_high < REFLASH_THRESHOLD - R_EPS:
        return "no_reflash"
    return "reshoot"


def verify_afterimage(
    width: int,
    height: int,
    new_gray: bytes,
    old_gray: bytes,
    photo_grays: list[bytes],
) -> GhostVerdict:
    points = select_calibration_points(new_gray, old_gray, width, height)
    roi = select_roi(new_gray, old_gray, width, height)

    intervals: list[tuple[float, float, float] | None] = []
    observation_sets: list[list[tuple[int, int, int]]] = []
    for photo_gray in photo_grays:
        observations = _observations(
            photo_gray, new_gray, old_gray, width, points, roi
        )
        observation_sets.append(observations)
        intervals.append(photo_ratio_interval(observations))

    serialised_intervals = [
        None
        if interval is None
        else {"r_min": interval[0], "r_max": interval[1]}
        for interval in intervals
    ]
    if any(interval is None for interval in intervals):
        raise EvidenceConflictError(
            "at least one photograph cannot be explained by any residual ratio "
            "with a gain in [0.5, 2] and the grayscale tolerance of 2",
            serialised_intervals,
        )

    ratio_low = max(interval[0] for interval in intervals)  # type: ignore[index]
    ratio_high = min(interval[1] for interval in intervals)  # type: ignore[index]
    if ratio_low > ratio_high + R_EPS:
        raise EvidenceConflictError(
            "the photographs admit residual-ratio intervals with an empty "
            "intersection; they have no common physical explanation",
            serialised_intervals,
        )

    fits: list[PhotoFit] = []
    for index, observations in enumerate(observation_sets, start=1):
        interval = intervals[index - 1]
        assert interval is not None
        # Report the exposure at this photo's own best-explained ratio, kept
        # inside the joint interval shared by every photograph.
        fit_ratio = min(max(interval[2], ratio_low), ratio_high)
        gain, offset, worst = fit_at_ratio(observations, fit_ratio)
        fits.append(
            PhotoFit(
                index=index,
                gain_a=gain,
                offset_b=offset,
                r_min=interval[0],
                r_max=interval[1],
                max_abs_residual=worst,
            )
        )

    return GhostVerdict(
        decision=_decision(ratio_low, ratio_high),
        r_min=ratio_low,
        r_max=ratio_high,
        calibration_points=points,
        roi=roi,
        photos=fits,
    )
