from __future__ import annotations

import itertools
import random

import pytest

from app.solver import (
    build_compatibility,
    colour_distance,
    solve_colouring,
)


def brute_force(costs, edges, palette, min_distance, locked=None):
    locked = dict(locked or {})
    n = len(costs)
    k = len(palette)
    compat = build_compatibility(palette, min_distance)
    best = None
    domains = []
    for v in range(n):
        if v in locked:
            domains.append([locked[v]])
        else:
            domains.append(range(k))
    for combo in itertools.product(*domains):
        if any(not compat[combo[u]][combo[v]] for u, v in edges):
            continue
        cost = sum(costs[v][combo[v]] for v in range(n))
        if best is None or cost < best:
            best = cost
    return best


@pytest.mark.parametrize("seed", range(25))
def test_solver_matches_brute_force(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 7)
    k = rng.randint(2, 4)
    palette = [
        (rng.randrange(0, 256), rng.randrange(0, 256), rng.randrange(0, 256))
        for _ in range(k)
    ]
    if len(set(palette)) != k:
        palette = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255)][:k]
    min_distance = rng.choice([1, 40, 120, 300])
    costs = [[rng.randrange(0, 5000) for _ in range(k)] for _ in range(n)]
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.45:
                edges.add((u, v))
    locked = {}
    if rng.random() < 0.3 and n:
        label = rng.randrange(n)
        locked[label] = rng.randrange(k)
    edges = sorted(edges)

    expected = brute_force(costs, edges, palette, min_distance, locked)
    result = solve_colouring(costs, edges, palette, min_distance, locked)
    if expected is None:
        assert result.assignment is None
        assert result.conflict is not None
    else:
        assert result.assignment is not None
        assert result.total_cost == expected
        compat = build_compatibility(palette, min_distance)
        for u, v in edges:
            assert compat[result.assignment[u]][result.assignment[v]]
        for v, c in locked.items():
            assert result.assignment[v] == c


def test_locked_too_close_is_reported() -> None:
    palette = [(0, 0, 0), (0, 0, 10), (255, 255, 255)]
    costs = [[0, 5, 10], [10, 5, 0]]
    result = solve_colouring(costs, [(0, 1)], palette, 20, locked={0: 0, 1: 1})
    assert result.assignment is None
    assert result.conflict is not None
    assert result.conflict.kind == "locked_colours_too_close"


def test_independent_components_are_merged() -> None:
    palette = [(0, 0, 0), (255, 255, 255)]
    costs = [[0, 10], [10, 0], [0, 10]]
    result = solve_colouring(costs, [(0, 1)], palette, 50)
    assert result.assignment is not None
    assert result.assignment[0] == 0
    assert result.assignment[1] == 1
    assert result.stats.components == 2


def test_palette_too_small_for_clique() -> None:
    palette = [(0, 0, 0), (255, 255, 255)]
    costs = [[0, 1], [1, 0], [1, 1]]
    edges = [(0, 1), (1, 2), (0, 2)]
    result = solve_colouring(costs, edges, palette, 1)
    assert result.assignment is None
    assert result.conflict is not None


def test_distance_boundary_inclusive() -> None:
    a = (0, 0, 0)
    b = (3, 4, 0)
    assert colour_distance(a, b) == 5
    compat = build_compatibility([a, b], 5)
    assert compat[0][1] and compat[1][0]
    res = solve_colouring([[0, 100], [100, 0]], [(0, 1)], [a, b], 5)
    assert res.assignment == [0, 1]


def test_no_edges_means_all_cheapest() -> None:
    palette = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    costs = [[9, 2, 7], [5, 8, 1]]
    res = solve_colouring(costs, [], palette, 0.5)
    assert res.assignment == [1, 2]
    assert res.total_cost == 3
