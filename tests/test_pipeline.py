from __future__ import annotations

import itertools
import random

from app.imaging import Page
from app.pipeline import analyse_pages
from app.solver import build_compatibility, solve_colouring


def make_page(index, width, height, labels, colours):
    rgb = bytearray()
    for y in range(height):
        for x in range(width):
            rgb.extend(colours[y][x])
    raw_labels = bytes(labels[y][x] for y in range(height) for x in range(width))
    return Page(
        index=index,
        width=width,
        height=height,
        rgb=bytes(rgb),
        labels=raw_labels,
    )


def brute_force(analysis, palette, min_distance):
    labels = analysis.labels
    compat = build_compatibility(palette, min_distance)
    index = {label: i for i, label in enumerate(labels)}
    edges = [(index[a], index[b]) for a, b in analysis.edges]
    best = None
    for combo in itertools.product(range(len(palette)), repeat=len(labels)):
        if any(not compat[combo[u]][combo[v]] for u, v in edges):
            continue
        cost = sum(analysis.costs[label][combo[i]] for i, label in enumerate(labels))
        if best is None or cost < best:
            best = cost
    return best


def test_analysis_matches_manual_pixel_sums() -> None:
    palette = [(0, 0, 0), (255, 255, 255)]
    page = make_page(
        1,
        2,
        1,
        [[3, 3]],
        [[(10, 20, 30), (200, 210, 220)]],
    )
    analysis = analyse_pages([page], palette)
    assert analysis.labels == [3]
    assert analysis.pixel_counts == {3: 2}
    assert analysis.edges == set()
    # Black: 10^2+20^2+30^2 plus 200^2+210^2+220^2.
    assert analysis.costs[3][0] == 1400 + 40000 + 44100 + 48400
    # White: (255-10)^2+(255-20)^2+(255-30)^2 plus the second pixel.
    assert analysis.costs[3][1] == (245**2 + 235**2 + 225**2) + (55**2 + 45**2 + 35**2)


def test_edges_include_horizontal_and_vertical_contact() -> None:
    label_rows = [[1, 2], [3, 4]]
    colours = [[(0, 0, 0)] * 2 for _ in range(2)]
    page = make_page(1, 2, 2, label_rows, colours)
    analysis = analyse_pages([page], [(0, 0, 0), (255, 255, 255)])
    assert analysis.edges == {(1, 2), (1, 3), (2, 4), (3, 4)}
    assert analysis.edge_pages[(1, 2)] == [1]


def test_edges_span_pages_without_cross_page_contact() -> None:
    # Label 1 and 2 touch on page 1; label 2 and 3 touch on page 2. Page 1's
    # last row must not be joined to page 2's first row.
    first = make_page(1, 2, 1, [[1, 2]], [[(0, 0, 0)] * 2])
    second = make_page(2, 2, 1, [[2, 3]], [[(0, 0, 0)] * 2])
    analysis = analyse_pages([first, second], [(0, 0, 0), (255, 255, 255)])
    assert analysis.edges == {(1, 2), (2, 3)}
    assert analysis.edge_pages[(2, 3)] == [2]


def test_random_pages_solver_is_exact() -> None:
    rng = random.Random(2024)
    palette = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255)]
    for trial in range(30):
        width = rng.randint(1, 4)
        height = rng.randint(1, 3)
        label_count = rng.randint(1, 4)
        label_rows = [
            [rng.randrange(1, label_count + 1) for _ in range(width)]
            for _ in range(height)
        ]
        colour_rows = [
            [
                (rng.randrange(256), rng.randrange(256), rng.randrange(256))
                for _ in range(width)
            ]
            for _ in range(height)
        ]
        page = make_page(1, width, height, label_rows, colour_rows)
        min_distance = rng.choice([1, 100, 200, 400])
        analysis = analyse_pages([page], palette)
        expected = brute_force(analysis, palette, min_distance)
        index = {label: i for i, label in enumerate(analysis.labels)}
        result = solve_colouring(
            [analysis.costs[label] for label in analysis.labels],
            [(index[a], index[b]) for a, b in analysis.edges],
            palette,
            min_distance,
        )
        if expected is None:
            assert result.assignment is None, trial
        else:
            assert result.total_cost == expected, trial
