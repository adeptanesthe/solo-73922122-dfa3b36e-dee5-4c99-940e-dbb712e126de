"""Page analysis, label rendering and legend construction."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image

from .imaging import MAX_LABELS, RGB, InputError, Page


@dataclass
class Analysis:
    labels: list[int]
    costs: dict[int, list[int]]
    # Number of pixels carrying each label across the whole set.
    pixel_counts: dict[int, int]
    # Undirected label pairs that share a pixel edge on at least one page.
    edges: set[tuple[int, int]]
    # Pages on which each edge occurs, used for conflict reporting.
    edge_pages: dict[tuple[int, int], list[int]]
    # Per-page label pixel counts, useful for the legend.
    per_page_counts: list[dict[int, int]]


def _add_edge(
    edge: tuple[int, int],
    page_index: int,
    per_page: set[tuple[int, int]],
    edge_pages: dict[tuple[int, int], list[int]],
) -> None:
    if edge in per_page:
        return
    per_page.add(edge)
    edge_pages.setdefault(edge, []).append(page_index)
    edge_pages[edge].sort()


def analyse_pages(pages: Sequence[Page], palette: Sequence[RGB]) -> Analysis:
    """Collect per-label colour statistics and the label adjacency graph."""
    stats: dict[int, list[int]] = {}
    edge_pages: dict[tuple[int, int], list[int]] = {}
    per_page_counts: list[dict[int, int]] = []

    for page in pages:
        page_counts: dict[int, int] = {}
        rgb = page.rgb
        labels = page.labels
        width, height = page.width, page.height

        for i, label in enumerate(labels):
            offset = i * 3
            r = rgb[offset]
            g = rgb[offset + 1]
            b = rgb[offset + 2]
            page_counts[label] = page_counts.get(label, 0) + 1
            row = stats.get(label)
            if row is None:
                stats[label] = [1, r, g, b, r * r + g * g + b * b]
            else:
                row[0] += 1
                row[1] += r
                row[2] += g
                row[3] += b
                row[4] += r * r + g * g + b * b

        per_page_counts.append(page_counts)
        if len(stats) > MAX_LABELS:
            raise InputError(
                f"the whole set may use at most {MAX_LABELS} label values, "
                f"found {len(stats)} by page {page.index}"
            )

        per_page: set[tuple[int, int]] = set()
        for y in range(height):
            base = y * width
            for x in range(width):
                a = labels[base + x]
                if x + 1 < width:
                    b = labels[base + x + 1]
                    if a != b:
                        _add_edge(
                            (a, b) if a < b else (b, a),
                            page.index,
                            per_page,
                            edge_pages,
                        )
                if y + 1 < height:
                    b = labels[base + width + x]
                    if a != b:
                        _add_edge(
                            (a, b) if a < b else (b, a),
                            page.index,
                            per_page,
                            edge_pages,
                        )

    labels = sorted(stats)
    costs: dict[int, list[int]] = {}
    for label in labels:
        count, sum_r, sum_g, sum_b, sum_squares = stats[label]
        row: list[int] = []
        for pr, pg, pb in palette:
            # sum (r-pr)^2 + (g-pg)^2 + (b-pb)^2, exactly, in integers.
            row.append(
                sum_squares
                + count * (pr * pr + pg * pg + pb * pb)
                - 2 * (pr * sum_r + pg * sum_g + pb * sum_b)
            )
        costs[label] = row

    return Analysis(
        labels=labels,
        costs=costs,
        pixel_counts={label: stats[label][0] for label in labels},
        edges=set(edge_pages),
        edge_pages=edge_pages,
        per_page_counts=per_page_counts,
    )


def total_error(analysis: Analysis, assignment: dict[int, int]) -> int:
    return sum(analysis.costs[label][colour] for label, colour in assignment.items())


def render_page(
    page: Page, assignment: dict[int, int], palette: Sequence[RGB]
) -> bytes:
    """Render one page as an RGB PNG using the label assignment."""
    lookup = {label: bytes(palette[c]) for label, c in assignment.items()}
    raw = bytearray(page.width * page.height * 3)
    for i, label in enumerate(page.labels):
        offset = i * 3
        raw[offset : offset + 3] = lookup[label]
    image = Image.frombytes("RGB", (page.width, page.height), bytes(raw))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def legend_entries(
    analysis: Analysis,
    assignment: dict[int, int],
    palette: Sequence[RGB],
    locks: dict[int, int],
) -> list[dict]:
    return [
        {
            "label": label,
            "colour": "#{:02x}{:02x}{:02x}".format(*palette[assignment[label]]),
            "rgb": list(palette[assignment[label]]),
            "locked": label in locks,
            "pixels": analysis.pixel_counts[label],
            "pages": [
                index
                for index, counts in enumerate(analysis.per_page_counts, start=1)
                if label in counts
            ],
        }
        for label in analysis.labels
    ]
