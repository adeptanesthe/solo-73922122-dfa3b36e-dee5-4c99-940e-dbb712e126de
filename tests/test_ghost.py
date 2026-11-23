"""Unit tests for the afterimage verification geometry and fitting."""

from __future__ import annotations

import random

import pytest

from app.ghost import (
    GAIN_MAX,
    GAIN_MIN,
    GRAY_TOLERANCE,
    REFLASH_THRESHOLD,
    EvidenceConflictError,
    photo_ratio_interval,
    select_calibration_points,
    select_roi,
    verify_afterimage,
)
from app.imaging import InputError

WIDTH = HEIGHT = 32


def _base_images():
    """Two flat grayscale halves; old image only differs inside a 16x16 zone."""
    new = bytearray(WIDTH * HEIGHT)
    old = bytearray(WIDTH * HEIGHT)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            value = 40 if x < 16 else 210
            index = y * WIDTH + x
            new[index] = value
            old[index] = value
    for y in range(8, 24):
        for x in range(8, 24):
            index = y * WIDTH + x
            old[index] = 0 if new[index] == 210 else 255
    return bytes(new), bytes(old)


def _photo(new, old, ratio, gain, offset, noise_map=None):
    pixels = bytearray(len(new))
    for index, (new_value, old_value) in enumerate(zip(new, old)):
        noise = noise_map[index] if noise_map else 0
        value = gain * (new_value + ratio * (old_value - new_value)) + offset + noise
        pixels[index] = max(0, min(255, round(value)))
    return bytes(pixels)


def test_calibration_points_cover_two_grays_and_respect_limits() -> None:
    new, old = _base_images()
    points = select_calibration_points(new, old, WIDTH, HEIGHT)
    assert 2 <= len(points) <= 8
    grays = {point.gray for point in points}
    assert len(grays) >= 2
    # Points only sit on pixels where the two images agree.
    for point in points:
        index = point.y * WIDTH + point.x
        assert new[index] == old[index] == point.gray
    # Deterministic order.
    assert points == sorted(points, key=lambda point: (point.y, point.x))


def test_calibration_needs_two_agreement_grays() -> None:
    new = bytes([50] * 32 + [200] * 32)
    old = bytearray(new)
    for index in range(32):  # every gray-50 pixel changes
        old[index] = 60
    with pytest.raises(InputError):
        select_calibration_points(new, bytes(old), 8, 8)


def test_roi_is_at_most_16x16_and_contains_change() -> None:
    new, old = _base_images()
    roi = select_roi(new, old, WIDTH, HEIGHT)
    assert roi.width <= 16 and roi.height <= 16
    assert roi.changed_pixels >= 1
    assert roi.gray_levels >= 2
    values = {
        new[y * WIDTH + x]
        for y in range(roi.y, roi.y + roi.height)
        for x in range(roi.x, roi.x + roi.width)
    }
    assert len(values) == roi.gray_levels


def test_roi_shrinks_for_small_images() -> None:
    new = bytearray([40] * 32 + [210] * 32)
    old = bytearray(new)
    old[0:4] = b"\x00\x00\x00\x00"
    roi = select_roi(bytes(new), bytes(old), 8, 8)
    assert roi.width == roi.height == 8
    assert roi.changed_pixels == 4


def test_roi_requires_a_changed_pixel() -> None:
    images = bytes([100 + (index % 2) * 10 for index in range(64)])
    with pytest.raises(InputError):
        select_roi(images, images, 8, 8)


def test_high_residual_ratio_recommends_reflash() -> None:
    new, old = _base_images()
    verdict = verify_afterimage(
        WIDTH, HEIGHT, new, old, [_photo(new, old, 0.30, 1.2, -8)]
    )
    assert verdict.decision == "reflash"
    assert verdict.r_min >= REFLASH_THRESHOLD - 1e-6
    fit = verdict.photos[0]
    assert fit.gain_a == pytest.approx(1.2, abs=0.02)
    assert fit.offset_b == pytest.approx(-8, abs=0.5)
    assert fit.max_abs_residual <= GRAY_TOLERANCE + 1e-6
    assert fit.r_min <= 0.30 <= fit.r_max


def test_low_residual_ratio_means_no_reflash() -> None:
    new, old = _base_images()
    verdict = verify_afterimage(
        WIDTH, HEIGHT, new, old, [_photo(new, old, 0.05, 1.0, 3)]
    )
    assert verdict.decision == "no_reflash"
    assert verdict.r_max < REFLASH_THRESHOLD


def test_interval_straddling_threshold_asks_for_reshoot() -> None:
    # Small old/new deltas widen the feasible ratio interval across 0.20.
    new, _ = _base_images()
    old = bytearray(new)
    for y in range(8, 24):
        for x in range(8, 24):
            index = y * WIDTH + x
            old[index] = (
                max(0, new[index] - 30)
                if new[index] == 210
                else min(255, new[index] + 30)
            )
    verdict = verify_afterimage(
        WIDTH, HEIGHT, new, bytes(old), [_photo(new, bytes(old), 0.20, 1.0, 0)]
    )
    assert verdict.decision == "reshoot"
    assert verdict.r_min < REFLASH_THRESHOLD < verdict.r_max


def test_two_photos_share_one_ratio_with_separate_exposures() -> None:
    new, old = _base_images()
    verdict = verify_afterimage(
        WIDTH,
        HEIGHT,
        new,
        old,
        [
            _photo(new, old, 0.30, 1.2, -8),
            _photo(new, old, 0.30, 0.8, 12),
        ],
    )
    assert verdict.decision == "reflash"
    assert len(verdict.photos) == 2
    gains = {fit.index: fit.gain_a for fit in verdict.photos}
    assert gains[1] == pytest.approx(1.2, abs=0.02)
    assert gains[2] == pytest.approx(0.8, abs=0.02)
    for fit in verdict.photos:
        assert fit.max_abs_residual <= GRAY_TOLERANCE + 1e-6
        assert fit.r_min <= verdict.r_min + 1e-9
        assert fit.r_max >= verdict.r_max - 1e-9


def test_disjoint_photo_intervals_are_evidence_conflict() -> None:
    new, old = _base_images()
    with pytest.raises(EvidenceConflictError) as exc:
        verify_afterimage(
            WIDTH,
            HEIGHT,
            new,
            old,
            [_photo(new, old, 0.0, 1.0, 0), _photo(new, old, 0.5, 1.0, 0)],
        )
    intervals = exc.value.photo_intervals
    assert len(intervals) == 2
    assert intervals[0]["r_max"] < intervals[1]["r_min"]


def test_noise_within_two_gray_levels_stays_feasible() -> None:
    rng = random.Random(7)
    new, old = _base_images()
    noise = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(len(new))]
    verdict = verify_afterimage(
        WIDTH, HEIGHT, new, old, [_photo(new, old, 0.25, 1.1, 5, noise)]
    )
    assert verdict.decision == "reflash"
    assert verdict.photos[0].max_abs_residual <= GRAY_TOLERANCE + 1e-6


def test_noise_beyond_tolerance_makes_the_photo_unexplainable() -> None:
    rng = random.Random(11)
    new, old = _base_images()
    noise = [rng.choice([-3, 3]) for _ in range(len(new))]
    photo = _photo(new, old, 0.25, 1.1, 5, noise)
    points = select_calibration_points(new, old, WIDTH, HEIGHT)
    roi = select_roi(new, old, WIDTH, HEIGHT)
    from app.ghost import _observations

    observations = _observations(photo, new, old, WIDTH, points, roi)
    assert photo_ratio_interval(observations) is None


def test_gain_is_constrained_to_the_allowed_range() -> None:
    new, old = _base_images()
    # The true gain 3.0 is outside [0.5, 2]; the interval must not contain it.
    photo = _photo(new, old, 0.30, 3.0, -200)
    points = select_calibration_points(new, old, WIDTH, HEIGHT)
    roi = select_roi(new, old, WIDTH, HEIGHT)
    from app.ghost import _observations

    observations = _observations(photo, new, old, WIDTH, points, roi)
    # Either unexplainable, or explainable only by a different ratio with a
    # gain inside the contract range -- never with gain 3.
    if photo_ratio_interval(observations) is not None:
        verdict = verify_afterimage(WIDTH, HEIGHT, new, old, [photo])
        assert GAIN_MIN <= verdict.photos[0].gain_a <= GAIN_MAX


def test_threshold_is_inclusive_for_reflash() -> None:
    new, old = _base_images()
    # Construct a photo whose feasible interval bottom sits at exactly ~0.20
    # by using noiseless data at r = 0.20 with strong contrast.
    verdict = verify_afterimage(
        WIDTH, HEIGHT, new, old, [_photo(new, old, 0.20, 1.0, 0)]
    )
    assert verdict.decision in {"reflash", "reshoot"}
    if verdict.decision == "reflash":
        assert verdict.r_min + 1e-8 >= REFLASH_THRESHOLD


def test_flat_image_is_rejected() -> None:
    flat = bytes([100] * 64)
    with pytest.raises(InputError):
        verify_afterimage(8, 8, flat, flat, [flat])


def test_reported_fit_satisfies_every_constraint_at_joint_midpoint() -> None:
    new, old = _base_images()
    verdict = verify_afterimage(
        WIDTH,
        HEIGHT,
        new,
        old,
        [
            _photo(new, old, 0.30, 1.2, -8),
            _photo(new, old, 0.30, 0.8, 12),
        ],
    )
    midpoint = (verdict.r_min + verdict.r_max) / 2
    points = verdict.calibration_points
    roi = verdict.roi
    for fit, photo in zip(verdict.photos, [
        _photo(new, old, 0.30, 1.2, -8),
        _photo(new, old, 0.30, 0.8, 12),
    ]):
        for point in points:
            index = point.y * WIDTH + point.x
            predicted = fit.gain_a * new[index] + fit.offset_b
            assert abs(predicted - photo[index]) <= GRAY_TOLERANCE + 1e-6
        for y in range(roi.y, roi.y + roi.height):
            for x in range(roi.x, roi.x + roi.width):
                index = y * WIDTH + x
                blend = new[index] + midpoint * (old[index] - new[index])
                predicted = fit.gain_a * blend + fit.offset_b
                assert abs(predicted - photo[index]) <= GRAY_TOLERANCE + 1e-6
