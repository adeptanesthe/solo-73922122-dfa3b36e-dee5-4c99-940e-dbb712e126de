"""Exact colouring solver.

The problem is a small weighted graph-colouring problem with a fixed palette:

* every legend entry (grayscale value) is a vertex;
* two entries are joined by an edge when they touch along a pixel edge on any
  page;
* assigning palette colour ``c`` to vertex ``v`` costs ``costs[v][c]``, the
  sum of squared RGB errors of all pixels belonging to ``v``;
* neighbouring vertices must receive colours whose RGB Euclidean distance is
  at least ``min_distance`` (so they cannot use the same palette colour).

The solver first splits the graph into connected components (components are
independent because palette colours may be reused), then runs a
branch-and-bound search with greedy upper bounds, MRV branching, forward
checking and an admissible per-vertex lower bound.  It never enumerates all
``K**N`` colourings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import hypot, inf

RGB = tuple[int, int, int]


@dataclass
class Conflict:
    """Why a colouring could not be produced.

    ``vertices`` are indices into the cost matrix.  ``info`` is JSON-safe and
    is enriched with labels by the API layer.
    """

    kind: str
    vertices: list[int]
    info: dict


@dataclass
class SolveStats:
    components: int = 0
    search_nodes: int = 0
    greedy_cost: int | None = None


@dataclass
class SolveResult:
    # Palette indices per vertex, or None when the problem is infeasible.
    assignment: list[int] | None
    total_cost: int | None
    stats: SolveStats = field(default_factory=SolveStats)
    conflict: Conflict | None = None


def colour_distance(a: RGB, b: RGB) -> float:
    return hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def build_compatibility(
    palette: Sequence[RGB], min_distance: float
) -> list[list[bool]]:
    """compat[i][j] is true when colours i and j are far enough apart."""
    k = len(palette)
    compat = [[False] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            # A tiny tolerance absorbs float representation noise only; the
            # palette itself consists of integer RGB values.
            compat[i][j] = (
                colour_distance(palette[i], palette[j]) + 1e-9 >= min_distance
            )
    return compat


def solve_colouring(
    costs: Sequence[Sequence[int]],
    edges: Iterable[tuple[int, int]],
    palette: Sequence[RGB],
    min_distance: float,
    locked: dict[int, int] | None = None,
) -> SolveResult:
    """Find a minimum-cost feasible assignment.

    ``costs[v][c]`` is the integer cost of giving vertex ``v`` palette colour
    ``c``.  ``locked`` maps vertex to a palette index.
    """
    n = len(costs)
    if n == 0:
        return SolveResult([], 0)
    k = len(palette)
    if k == 0 or any(len(row) != k for row in costs):
        raise ValueError("invalid cost matrix")
    locked = dict(locked or {})
    for v, c in locked.items():
        if not (0 <= v < n) or not (0 <= c < k):
            raise ValueError("lock outside problem bounds")

    compat = build_compatibility(palette, min_distance)
    adjacency: list[set[int]] = [set() for _ in range(n)]
    clean_edges: set[tuple[int, int]] = set()
    for u, v in edges:
        if u == v:
            continue
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge outside problem bounds")
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in clean_edges:
            continue
        clean_edges.add((a, b))
        adjacency[a].add(b)
        adjacency[b].add(a)

    # A locked pair that touches is an immediate, explainable conflict.
    for u, v in sorted(clean_edges):
        if u in locked and v in locked:
            cu, cv = locked[u], locked[v]
            if not compat[cu][cv]:
                return SolveResult(
                    None,
                    None,
                    conflict=Conflict(
                        kind="locked_colours_too_close",
                        vertices=[u, v],
                        info={
                            "colour_indices": [cu, cv],
                            "distance": colour_distance(palette[cu], palette[cv]),
                            "min_distance": min_distance,
                        },
                    ),
                )

    # Connected components: the palette is unlimited, so assignments of
    # different components can be optimised independently.
    seen = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            v = stack.pop()
            component.append(v)
            for w in adjacency[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        components.append(sorted(component))

    assignment = [-1] * n
    total = 0
    stats = SolveStats(components=len(components))
    for component in components:
        local_index = {v: i for i, v in enumerate(component)}
        local_costs = [costs[v] for v in component]
        local_edges = [
            (local_index[u], local_index[v])
            for u, v in clean_edges
            if u in local_index and v in local_index
        ]
        local_locked = {
            local_index[v]: c for v, c in locked.items() if v in local_index
        }
        result = _solve_component(local_costs, local_edges, compat, local_locked, stats)
        if isinstance(result, Conflict):
            return SolveResult(
                None,
                None,
                stats,
                conflict=Conflict(
                    kind=result.kind,
                    vertices=[component[i] for i in result.vertices],
                    info=result.info,
                ),
            )
        local_assignment, local_cost = result
        total += local_cost
        for v, c in zip(component, local_assignment):
            assignment[v] = c
    return SolveResult(assignment, total, stats)


def _solve_component(
    costs: Sequence[Sequence[int]],
    edges: Sequence[tuple[int, int]],
    compat: Sequence[Sequence[bool]],
    locked: dict[int, int],
    global_stats: SolveStats,
) -> tuple[list[int], int] | Conflict:
    n = len(costs)
    k = len(costs[0])
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbours in adjacency:
        neighbours.sort()

    assigned = [-1] * n
    for v, c in locked.items():
        assigned[v] = c
    for u, v in edges:
        if (
            assigned[u] >= 0
            and assigned[v] >= 0
            and not compat[assigned[u]][assigned[v]]
        ):
            # Already handled globally; kept as a safety net.
            return Conflict("locked_colours_too_close", [u, v], {})

    base_cost = sum(costs[v][c] for v, c in locked.items())
    unlocked = [v for v in range(n) if assigned[v] < 0]
    if not unlocked:
        return assigned, base_cost

    # Root domains, filtered by the locked vertices.
    root_domains: dict[int, list[int]] = {}
    for v in unlocked:
        domain = _allowed(v, assigned, adjacency, compat, k)
        if not domain:
            return Conflict(
                "no_compatible_colour",
                [v],
                {"blocked_by": _blockers(v, assigned, adjacency)},
            )
        root_domains[v] = domain

    root_lower_bound = base_cost + sum(
        min(costs[v][c] for c in root_domains[v]) for v in unlocked
    )

    # Greedy upper bound from a few deterministic vertex orders.  This only
    # tightens pruning; correctness does not depend on it.
    best_assignment: list[int] | None = None
    best_cost = inf
    degree = {v: len(adjacency[v]) for v in unlocked}
    regret = {}
    for v in unlocked:
        ordered = sorted(costs[v])
        regret[v] = (ordered[1] - ordered[0]) if len(ordered) > 1 else 0

    orders = [
        sorted(unlocked, key=lambda v: (-degree[v], -regret[v], v)),
        sorted(unlocked, key=lambda v: (-regret[v], -degree[v], v)),
        sorted(unlocked, key=lambda v: (-degree[v], v)),
        list(unlocked),
    ]
    for order in orders:
        greedy = _greedy(order, assigned, base_cost, costs, adjacency, compat, k)
        if greedy is not None:
            greedy_assignment, greedy_cost = greedy
            if greedy_cost < best_cost:
                best_assignment, best_cost = greedy_assignment, greedy_cost
    if best_assignment is not None:
        global_stats.greedy_cost = (
            best_cost
            if global_stats.greedy_cost is None
            else min(global_stats.greedy_cost, best_cost)
        )

    # The greedy solution hits the lower bound: it is provably optimal.
    if best_assignment is not None and best_cost == root_lower_bound:
        return best_assignment, best_cost

    nodes = 0
    stop = False
    first_conflict: Conflict | None = None

    def search(current: list[int], cost: int) -> None:
        nonlocal best_assignment, best_cost, nodes, stop, first_conflict
        if stop:
            return
        nodes += 1
        if nodes > 50_000_000:  # defensive guard; not expected for N<=12
            raise RuntimeError("colouring search exceeded its node budget")

        remaining = [v for v in unlocked if current[v] < 0]
        domains: dict[int, list[int]] = {}
        lower = cost
        for v in remaining:
            domain = _allowed(v, current, adjacency, compat, k)
            if not domain:
                if first_conflict is None:
                    first_conflict = Conflict(
                        "no_compatible_colour",
                        [v],
                        {"blocked_by": _blockers(v, current, adjacency)},
                    )
                return
            domains[v] = domain
            lower += min(costs[v][c] for c in domain)
        if lower >= best_cost:
            return

        # Pairwise support check: cheap forward checking that prunes dead
        # branches before assigning a vertex.
        for u, v in edges:
            if current[u] >= 0 or current[v] >= 0:
                continue
            if not _domains_support(domains[u], domains[v], compat):
                if first_conflict is None:
                    first_conflict = Conflict("unsatisfiable_constraints", [u, v], {})
                return

        if not remaining:
            if cost < best_cost:
                best_assignment = list(current)
                best_cost = cost
                if best_cost == root_lower_bound:
                    stop = True
            return

        def variable_key(v: int) -> tuple[int, int, int, int]:
            dom = domains[v]
            if len(dom) > 1:
                ordered = sorted(costs[v][c] for c in dom)
                reg = ordered[1] - ordered[0]
            else:
                reg = 10**18
            return (len(dom), -len(adjacency[v]), -reg, v)

        vertex = min(remaining, key=variable_key)
        domain = domains[vertex]
        domain.sort(
            key=lambda c: (
                costs[vertex][c],
                -_support_count(c, vertex, domains, adjacency, compat),
                c,
            )
        )
        for c in domain:
            current[vertex] = c
            search(current, cost + costs[vertex][c])
            current[vertex] = -1
            if stop:
                return

    search(assigned, base_cost)
    global_stats.search_nodes += nodes
    if best_assignment is None:
        if first_conflict is not None:
            return first_conflict
        return Conflict("unsatisfiable_constraints", list(unlocked), {})
    return best_assignment, int(best_cost)


def _allowed(
    v: int,
    assigned: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    compat: Sequence[Sequence[bool]],
    k: int,
) -> list[int]:
    result = []
    for c in range(k):
        ok = True
        for neighbour in adjacency[v]:
            nc = assigned[neighbour]
            if nc >= 0 and not compat[c][nc]:
                ok = False
                break
        if ok:
            result.append(c)
    return result


def _blockers(
    v: int, assigned: Sequence[int], adjacency: Sequence[Sequence[int]]
) -> list[list[int]]:
    return [
        [neighbour, assigned[neighbour]]
        for neighbour in adjacency[v]
        if assigned[neighbour] >= 0
    ]


def _domains_support(
    left: Sequence[int], right: Sequence[int], compat: Sequence[Sequence[bool]]
) -> bool:
    for c in left:
        row = compat[c]
        for d in right:
            if row[d]:
                return True
    return False


def _support_count(
    c: int,
    vertex: int,
    domains: dict[int, list[int]],
    adjacency: Sequence[Sequence[int]],
    compat: Sequence[Sequence[bool]],
) -> int:
    """How many colour choices remain available to unassigned neighbours."""
    count = 0
    row = compat[c]
    for neighbour in adjacency[vertex]:
        dom = domains.get(neighbour)
        if dom is None:
            continue
        count += sum(1 for d in dom if row[d])
    return count


def _greedy(
    order: Sequence[int],
    assigned: Sequence[int],
    base_cost: int,
    costs: Sequence[Sequence[int]],
    adjacency: Sequence[Sequence[int]],
    compat: Sequence[Sequence[bool]],
    k: int,
) -> tuple[list[int], int] | None:
    current = list(assigned)
    total = base_cost
    for v in order:
        domain = _allowed(v, current, adjacency, compat, k)
        if not domain:
            return None
        chosen = min(domain, key=lambda c: (costs[v][c], c))
        current[v] = chosen
        total += costs[v][chosen]
    return current, total
