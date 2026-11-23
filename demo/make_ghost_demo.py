"""Build inputs for the afterimage (ghosting) verification walkthrough.

The screen page is 32x32 with two flat grayscale halves.  The PNG delivered
in the previous round agreed with the current image everywhere except a
central 16x16 zone, so the afterimage check has both stable calibration
pixels and a measurable region of interest.

Photographs are synthesised from the physical model::

    photo = a * (new + r * (old - new)) + b

Four photographs are emitted into ``demo/out/ghost`` (created on demand,
never committed):

* ``shot_ambiguous.png`` - a single weak-contrast exposure whose feasible
  residual-ratio interval straddles the 20% threshold, so one photo alone
  comes back ``reshoot``;
* ``shot_clear_a.png`` / ``shot_clear_b.png`` - two low-residual exposures of
  the same screen, each with its own gain/offset, whose common interval with
  the ambiguous photo stays below 20%, so adding the second photo changes the
  verdict to ``no_reflash``;
* ``shot_high_ghost.png`` - one strongly ghosted exposure whose whole
  feasible interval is above 20%, returning ``reflash`` on its own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

WIDTH = HEIGHT = 32
# The service renders the two palette colours as pure black/white, so the
# "new" grayscale seen by the check is 0 on the left half and 255 on the right.
DARK = 0
BRIGHT = 255


def _save_gray(path: Path, pixels: list[int]) -> None:
    image = Image.new("L", (WIDTH, HEIGHT))
    image.putdata(pixels)
    image.save(path)


def _save_rgb(path: Path, gray: list[int]) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT))
    image.putdata([(value, value, value) for value in gray])
    image.save(path)


def build(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    # Weak old/new contrast inside the region of interest: small grayscale
    # deltas make a single photograph's feasible residual-ratio interval wide
    # enough to straddle the 20% threshold, while a second well-exposed photo
    # pulls the common interval back below it.
    weak_delta = 60
    new_gray: list[int] = []
    old_gray: list[int] = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            value = DARK if x < WIDTH // 2 else BRIGHT
            new_gray.append(value)
            if WIDTH // 4 <= x < 3 * WIDTH // 4 and HEIGHT // 4 <= y < 3 * HEIGHT // 4:
                old_gray.append(
                    min(255, value + weak_delta)
                    if value == DARK
                    else max(0, value - weak_delta)
                )
            else:
                old_gray.append(value)

    def photograph(ratio: float, gain: float, offset: float) -> list[int]:
        pixels = []
        for new_value, old_value in zip(new_gray, old_gray):
            blended = new_value + ratio * (old_value - new_value)
            value = round(gain * blended + offset)
            pixels.append(max(0, min(255, value)))
        return pixels

    paths = {
        "current_rgb": directory / "current_rgb.png",
        "current_labels": directory / "current_labels.png",
        "old": directory / "previous_delivery.png",
        "ambiguous": directory / "shot_ambiguous.png",
        "clear_a": directory / "shot_clear_a.png",
        "clear_b": directory / "shot_clear_b.png",
        "high_ghost": directory / "shot_high_ghost.png",
    }
    # Label map: one label per half, matching the two-colour demo palette.
    labels = [1 if value == DARK else 2 for value in new_gray]
    _save_rgb(paths["current_rgb"], new_gray)
    _save_gray(paths["current_labels"], labels)
    _save_gray(paths["old"], old_gray)
    # r = 0.15 on a weak signal: rounding plus the +/-2 grayscale tolerance
    # leaves a feasible interval that straddles 0.20 on a single photograph.
    _save_gray(paths["ambiguous"], photograph(0.15, 1.0, 0))
    # Two clearly low-residual exposures (r = 0.08), different gain/offset,
    # chosen so no pixel saturates against 0/255.
    _save_gray(paths["clear_a"], photograph(0.08, 1.0, 0))
    _save_gray(paths["clear_b"], photograph(0.08, 0.9, 6))
    # One strongly ghosted exposure (r = 0.35) whose whole feasible interval
    # sits above the 20% re-flash threshold.
    _save_gray(paths["high_ghost"], photograph(0.35, 1.0, 0))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, nargs="?", default=Path("demo/out/ghost"))
    args = parser.parse_args()
    paths = build(args.out)
    for name, path in paths.items():
        print(f"{name:15s} -> {path}")


if __name__ == "__main__":
    main()
