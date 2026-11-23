"""Build a small demo set of labelled pages for the API walkthrough."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

# Palette used by the walkthrough.  Hand-picked e-paper colours; the pair
# distances are what makes the cross-page colour change happen.
PALETTE = ["#101010", "#f2eee2", "#a12f26", "#2f5d8a"]

# Label ids (grayscale values in the maps).
SKY = 32
PAPER = 240
TITLE = 64
RULE = 96
SEAL = 160


def build_page_one() -> tuple[Image.Image, Image.Image]:
    """Cover page: paper, title bar, rule, seal on paper."""
    width, height = 96, 96
    labels = Image.new("L", (width, height), PAPER)
    original = Image.new("RGB", (width, height), (242, 238, 226))

    # Title block.
    for y in range(8, 24):
        for x in range(8, 88):
            labels.putpixel((x, y), TITLE)
            original.putpixel((x, y), (96, 88, 76))

    # Horizontal rule.
    for x in range(8, 88):
        for y in (30, 31):
            labels.putpixel((x, y), RULE)
            original.putpixel((x, y), (150, 140, 120))

    # Seal in the lower right corner, touching the paper only.
    for y in range(64, 84):
        for x in range(64, 84):
            labels.putpixel((x, y), SEAL)
            original.putpixel((x, y), (168, 58, 44))
    return original, labels


def build_page_two() -> tuple[Image.Image, Image.Image]:
    """Map page: blue sky on paper, title, grey roads, red seal."""

    width, height = 96, 96
    labels = Image.new("L", (width, height), PAPER)
    original = Image.new("RGB", (width, height), (242, 238, 226))

    # Sky region in the upper third: it touches both the title bar and the
    # paper, and on this page it touches the roads as well.
    for y in range(6, 34):
        for x in range(6, 90):
            labels.putpixel((x, y), SKY)
            original.putpixel((x, y), (47, 93, 138))

    # Title text drawn over the sky.
    for y in range(10, 18):
        for x in range(14, 80):
            labels.putpixel((x, y), TITLE)
            original.putpixel((x, y), (92, 84, 72))

    # Roads below the sky.
    for x in range(6, 90):
        for y in (44, 45, 46):
            labels.putpixel((x, y), RULE)
            original.putpixel((x, y), (150, 140, 120))

    # Seal touching the roads (shares an edge with RULE but not with SKY).
    for y in range(46, 78):
        for x in range(68, 88):
            labels.putpixel((x, y), SEAL)
            original.putpixel((x, y), (168, 58, 44))
    return original, labels


def build_page_three() -> tuple[Image.Image, Image.Image]:
    """Back cover: paper with a seal and a narrow sky strip at the top."""

    width, height = 96, 96
    labels = Image.new("L", (width, height), PAPER)
    original = Image.new("RGB", (width, height), (242, 238, 226))
    for y in range(5):
        for x in range(width):
            labels.putpixel((x, y), SKY)
            original.putpixel((x, y), (47, 93, 138))
    for y in range(80, 92):
        for x in range(8, 20):
            labels.putpixel((x, y), SEAL)
            original.putpixel((x, y), (168, 58, 44))
    return original, labels


def write_pages(directory: Path) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    builders = [build_page_one, build_page_two, build_page_three]
    manifest = []
    for index, builder in enumerate(builders, start=1):
        original, labels = builder()
        rgb_path = directory / f"page{index}_rgb.png"
        label_path = directory / f"page{index}_labels.png"
        original.save(rgb_path)
        labels.save(label_path)
        manifest.append(
            {
                "page": index,
                "rgb": str(rgb_path),
                "labels": str(label_path),
                "size": list(original.size),
            }
        )
    labels_seen = sorted(
        {value for builder in builders for value in builder()[1].tobytes()}
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "pages": manifest,
                "palette": PALETTE,
                "min_distance": 140,
                "labels": labels_seen,
                "label_meanings": {
                    str(SKY): "sky",
                    str(PAPER): "paper",
                    str(TITLE): "title ink",
                    str(RULE): "rule / road",
                    str(SEAL): "seal",
                },
            },
            indent=2,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, nargs="?", default=Path("demo/pages"))
    args = parser.parse_args()
    pages = write_pages(args.out)
    for page in pages:
        print(f"wrote {page['rgb']} and {page['labels']}")


if __name__ == "__main__":
    main()
