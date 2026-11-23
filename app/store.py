"""In-memory package store with atomic versioning."""

from __future__ import annotations

import io
import json
import threading
import uuid
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .imaging import RGB, Page, colour_to_hex
from .pipeline import Analysis, legend_entries, render_page
from .solver import SolveResult, colour_distance, solve_colouring


class NotFoundError(LookupError):
    pass


class VersionNotFoundError(NotFoundError):
    pass


class VersionConflictError(RuntimeError):
    def __init__(self, message: str, current_version: int) -> None:
        super().__init__(message)
        self.current_version = current_version


class InfeasibleError(RuntimeError):
    def __init__(self, result: SolveResult, details: dict[str, Any]) -> None:
        super().__init__("no feasible colouring")
        self.result = result
        self.details = details


@dataclass
class Bundle:
    number: int
    assignment: dict[int, int]
    total_error: int
    legend: list[dict]
    pages_png: list[bytes]
    legend_json: bytes
    legend_png: bytes
    created_at: str
    change: dict[str, Any]
    solver: dict[str, int]

    def zip_bytes(self, package_id: str) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, data in enumerate(self.pages_png, start=1):
                archive.writestr(f"page_{index:02d}.png", data)
            archive.writestr("legend.json", self.legend_json)
            archive.writestr("legend.png", self.legend_png)
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "package_id": package_id,
                        "version": self.number,
                        "created_at": self.created_at,
                        "total_error": self.total_error,
                        "solver": self.solver,
                        "files": [
                            f"page_{i:02d}.png"
                            for i in range(1, len(self.pages_png) + 1)
                        ]
                        + ["legend.json", "legend.png"],
                    },
                    indent=2,
                ).encode(),
            )
        return buffer.getvalue()


@dataclass
class Package:
    id: str
    palette: list[RGB]
    min_distance: float
    pages: list[Page]
    analysis: Analysis
    created_at: str
    locks: dict[int, int] = field(default_factory=dict)
    current: Bundle | None = None
    # Version numbers ever published; superseded numbers are remembered only
    # so that downloads can report a version conflict rather than a 404.
    published: set[int] = field(default_factory=set)
    # Serialises edits and downloads per package so a download can never see
    # a half-rendered bundle.
    lock: threading.RLock = field(default_factory=threading.RLock)


class PackageStore:
    def __init__(self) -> None:
        self._packages: dict[str, Package] = {}
        self._lock = threading.RLock()

    # -- creation ---------------------------------------------------------
    def create(
        self,
        pages: Sequence[Page],
        palette: Sequence[RGB],
        min_distance: float,
        analysis: Analysis,
        requested_locks: dict[int, int],
    ) -> dict[str, Any]:
        package_id = uuid.uuid4().hex
        package = Package(
            id=package_id,
            palette=list(palette),
            min_distance=min_distance,
            pages=list(pages),
            analysis=analysis,
            created_at=_now(),
        )
        bundle = _compute_bundle(
            package,
            requested_locks,
            number=1,
            change=_lock_delta({}, requested_locks, package.palette, kind="create"),
        )
        package.locks = dict(requested_locks)
        package.current = bundle
        package.published = {1}
        with self._lock:
            self._packages[package_id] = package
            return _summary_locked(package)

    # -- reads ------------------------------------------------------------
    def get(self, package_id: str) -> Package:
        with self._lock:
            package = self._packages.get(package_id)
        if package is None:
            raise NotFoundError(package_id)
        return package

    def describe(self, package_id: str) -> dict[str, Any]:
        package = self.get(package_id)
        with package.lock:
            return _summary_locked(package)

    def get_bundle(self, package_id: str, version: int) -> Bundle:
        package = self.get(package_id)
        with package.lock:
            current = package.current
            assert current is not None
            if version == current.number:
                return current
            if version in package.published:
                raise VersionConflictError(
                    f"version {version} has been superseded by version "
                    f"{current.number}; download the current version",
                    current.number,
                )
            if 1 <= version < current.number:
                raise VersionConflictError(
                    f"version {version} no longer exists; current version is "
                    f"{current.number}",
                    current.number,
                )
            raise VersionNotFoundError(
                f"version {version} has not been published; current version is "
                f"{current.number}"
            )

    def download(self, package_id: str, version: int) -> tuple[Bundle, bytes]:
        package = self.get(package_id)
        bundle = self.get_bundle(package_id, version)
        return bundle, bundle.zip_bytes(package.id)

    # -- edits ------------------------------------------------------------
    def set_lock(
        self, package_id: str, label: int, colour_index: int
    ) -> dict[str, Any]:
        package = self.get(package_id)
        with package.lock:
            current = package.current
            if current is None:
                raise NotFoundError(package_id)
            if label not in package.analysis.costs:
                raise KeyError(f"label {label} does not occur in this package")
            candidate_locks = dict(package.locks)
            candidate_locks[label] = colour_index
            next_number = current.number + 1
            bundle = _compute_bundle(
                package,
                candidate_locks,
                number=next_number,
                change={
                    **_lock_delta(
                        package.locks, candidate_locks, package.palette, kind="lock"
                    ),
                    "previous_version": current.number,
                },
            )
            # Commit only after every page and the legend have been rendered.
            package.locks = candidate_locks
            package.current = bundle
            package.published.add(next_number)
            return _summary_locked(package)

    def clear_lock(self, package_id: str, label: int) -> dict[str, Any]:
        package = self.get(package_id)
        with package.lock:
            current = package.current
            if current is None:
                raise NotFoundError(package_id)
            if label not in package.analysis.costs:
                raise KeyError(f"label {label} does not occur in this package")
            if label not in package.locks:
                raise KeyError(f"label {label} is not locked")
            candidate_locks = dict(package.locks)
            candidate_locks.pop(label)
            next_number = current.number + 1
            bundle = _compute_bundle(
                package,
                candidate_locks,
                number=next_number,
                change={
                    **_lock_delta(
                        package.locks, candidate_locks, package.palette, kind="unlock"
                    ),
                    "previous_version": current.number,
                },
            )
            package.locks = candidate_locks
            package.current = bundle
            package.published.add(next_number)
            return _summary_locked(package)


def _lock_delta(
    previous: dict[int, int],
    current: dict[int, int],
    palette: Sequence[RGB],
    kind: str,
) -> dict[str, Any]:
    """Describe what changed between two lock maps, with resolved colours."""
    added = [label for label in current if label not in previous]
    removed = [label for label in previous if label not in current]
    changed = [
        label
        for label in current
        if label in previous and previous[label] != current[label]
    ]
    focus = sorted(added + removed + changed)
    delta: dict[str, Any] = {
        "type": kind,
        "locks": {
            str(label): colour_to_hex(palette[index])
            for label, index in sorted(current.items())
        },
    }
    if focus:
        delta["label"] = focus[0]
        delta["colours"] = {
            str(label): {
                "before": (
                    colour_to_hex(palette[previous[label]])
                    if label in previous
                    else None
                ),
                "after": (
                    colour_to_hex(palette[current[label]]) if label in current else None
                ),
            }
            for label in focus
        }
    return delta


def _compute_bundle(
    package: Package,
    locks: dict[int, int],
    number: int,
    change: dict[str, Any],
) -> Bundle:
    labels = package.analysis.labels
    index = {label: i for i, label in enumerate(labels)}
    costs = [package.analysis.costs[label] for label in labels]
    edges = [(index[a], index[b]) for a, b in sorted(package.analysis.edges)]
    local_locks = {index[label]: colour_index for label, colour_index in locks.items()}
    result = solve_colouring(
        costs, edges, package.palette, package.min_distance, local_locks
    )
    if result.assignment is None:
        raise InfeasibleError(result, _conflict_details(package, result, locks))

    assignment = {label: result.assignment[index[label]] for label in labels}
    legend = legend_entries(package.analysis, assignment, package.palette, locks)
    pages_png = [
        render_page(page, assignment, package.palette) for page in package.pages
    ]
    created_at = _now()
    legend_payload = {
        "package_id": package.id,
        "version": number,
        "created_at": created_at,
        "palette": [colour_to_hex(colour) for colour in package.palette],
        "min_distance": package.min_distance,
        "total_error": result.total_cost,
        "solver": {
            "search_nodes": result.stats.search_nodes,
            "components": result.stats.components,
        },
        "locks": _locks_json(package, locks),
        "entries": legend,
        "pages": [
            {
                "page": page.index,
                "file": f"page_{page.index:02d}.png",
                "width": page.width,
                "height": page.height,
                "source_rgb": page.rgb_name,
                "source_labels": page.label_name,
            }
            for page in package.pages
        ],
    }
    legend_json = json.dumps(legend_payload, indent=2, ensure_ascii=False).encode()
    legend_png = _render_legend_png(legend, number)
    return Bundle(
        number=number,
        assignment=assignment,
        total_error=int(result.total_cost),
        legend=legend,
        pages_png=pages_png,
        legend_json=legend_json,
        legend_png=legend_png,
        created_at=created_at,
        change=change,
        solver={
            "search_nodes": result.stats.search_nodes,
            "components": result.stats.components,
        },
    )


def _conflict_details(
    package: Package, result: SolveResult, locks: dict[int, int]
) -> dict[str, Any]:
    """Turn a solver conflict into an operator-facing explanation.

    The report separates the two things an operator can act on: a minimum
    distance that is impossible for mutually touching labels, and locks that
    make an otherwise solvable set infeasible.
    """
    labels = package.analysis.labels
    conflict = result.conflict
    details: dict[str, Any] = {
        "conflict": True,
        "type": conflict.kind if conflict else "unsatisfiable_constraints",
        "message": "no colouring satisfies the minimum colour distance",
        "labels": [],
        "edges": [],
        "colour_pairs": [],
        "clique": None,
        "max_compatible_colours": _max_compatible_clique(
            package.palette, package.min_distance
        ),
        "palette": [colour_to_hex(colour) for colour in package.palette],
        "min_distance": package.min_distance,
        "locks": _locks_json(package, locks),
        "locks_cause": False,
    }
    if conflict is None:
        return details

    vertex_labels = [labels[v] for v in conflict.vertices]
    details["labels"] = vertex_labels
    if conflict.kind == "locked_colours_too_close":
        details["message"] = (
            "the requested lock puts two touching labels on colours that are too close"
        )
        details["colour_pairs"] = [
            {
                "labels": vertex_labels,
                "colours": [
                    colour_to_hex(package.palette[c])
                    for c in conflict.info.get("colour_indices", [])
                    if 0 <= c < len(package.palette)
                ],
                "distance": conflict.info.get("distance"),
                "min_distance": package.min_distance,
            }
        ]
    else:
        details["message"] = (
            "the minimum colour distance cannot be met with the available "
            "colours, even before considering error minimisation"
        )
        details["unassignable_label"] = vertex_labels[0] if vertex_labels else None
        details["blocked_by"] = [
            {
                "label": labels[neighbour],
                "colour": colour_to_hex(package.palette[colour_index]),
                "locked": labels[neighbour] in locks,
            }
            for neighbour, colour_index in conflict.info.get("blocked_by", [])
            if 0 <= neighbour < len(labels) and 0 <= colour_index < len(package.palette)
        ]

    # Touching pairs around the labels involved in the conflict, with the
    # pages on which they touch, so the operator can find the artwork.
    focus = set(vertex_labels)
    details["edges"] = [
        {
            "labels": [a, b],
            "pages": [page + 1 for page in package.analysis.edge_pages[(a, b)]],
        }
        for a, b in sorted(package.analysis.edges)
        if a in focus or b in focus
    ]

    # A clique larger than the palette's compatibility clique cannot be
    # coloured, regardless of costs.  At most 12 labels, so subsets are cheap.
    clique = _max_clique(package.analysis.edges, labels)
    allowed = details["max_compatible_colours"]
    if clique and len(clique) > allowed:
        details["clique"] = {
            "labels": clique,
            "pages": sorted(
                {
                    page + 1
                    for i, a in enumerate(clique)
                    for b in clique[i + 1 :]
                    for page in package.analysis.edge_pages.get(
                        (min(a, b), max(a, b)), []
                    )
                }
            ),
            "required_colours": len(clique),
            "available_colours": allowed,
        }
        details["labels"] = clique
        details["message"] = (
            f"{len(clique)} labels all touch each other, but only {allowed} of "
            f"the palette colours are mutually at least "
            f"{package.min_distance:g} apart"
        )

    # Would the same set be solvable without the locks?  That isolates the
    # cause for the operator and justifies keeping the old package on 409.
    if locks:
        index = {label: i for i, label in enumerate(labels)}
        costs = [package.analysis.costs[label] for label in labels]
        edges = [(index[a], index[b]) for a, b in sorted(package.analysis.edges)]
        unlocked = solve_colouring(
            costs, edges, package.palette, package.min_distance, {}
        )
        details["locks_cause"] = unlocked.assignment is not None
        if unlocked.assignment is not None:
            details["message"] = (
                "the requested locks make the set infeasible; unlocking would "
                "allow a colouring"
            )
    return details


def _max_clique(edges: set[tuple[int, int]], labels: Sequence[int]) -> list[int]:
    """Largest mutually touching label set (brute force over <=12 labels)."""
    n = len(labels)
    adjacency = [0] * n
    position = {label: i for i, label in enumerate(labels)}
    for a, b in edges:
        i, j = position[a], position[b]
        adjacency[i] |= 1 << j
        adjacency[j] |= 1 << i
    best_mask = 0
    best_size = 0
    for mask in range(1, 1 << n):
        size = mask.bit_count()
        if size <= best_size:
            continue
        ok = True
        for i in range(n):
            if mask >> i & 1 and (adjacency[i] & (mask & ~(1 << i))) != (
                mask & ~(1 << i)
            ):
                ok = False
                break
        if ok:
            best_mask, best_size = mask, size
    return [labels[i] for i in range(n) if best_mask >> i & 1]


def _max_compatible_clique(palette: Sequence[RGB], min_distance: float) -> int:
    """Largest set of palette colours that are pairwise far enough apart."""
    k = len(palette)
    best = 0
    for mask in range(1, 1 << k):
        size = mask.bit_count()
        if size <= best:
            continue
        members = [i for i in range(k) if mask >> i & 1]
        if all(
            colour_distance(palette[i], palette[j]) + 1e-9 >= min_distance
            for a, i in enumerate(members)
            for j in members[a + 1 :]
        ):
            best = size
    return best


def _locks_json(package: Package, locks: dict[int, int]) -> dict[str, str]:
    return {
        str(label): colour_to_hex(package.palette[colour_index])
        for label, colour_index in sorted(locks.items())
    }


def _render_legend_png(legend: Sequence[dict], version: int) -> bytes:
    row_height = 30
    width = 460
    header = 46
    height = header + row_height * max(len(legend), 1) + 12
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((12, 14), f"Legend - version {version}", fill=(0, 0, 0), font=font)
    for i, entry in enumerate(legend):
        y = header + i * row_height
        red, green, blue = entry["rgb"]
        draw.rectangle(
            [12, y + 3, 36, y + 27],
            fill=(red, green, blue),
            outline=(0, 0, 0),
        )
        lock = "  [locked]" if entry.get("locked") else ""
        text = (
            f"label {entry['label']:>3}    {entry['colour']}    "
            f"rgb({red},{green},{blue}){lock}"
        )
        draw.text((46, y + 7), text, fill=(0, 0, 0), font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _summary_locked(package: Package) -> dict[str, Any]:
    bundle = package.current
    assert bundle is not None
    return {
        "package_id": package.id,
        "version": bundle.number,
        "created_at": bundle.created_at,
        "palette": [colour_to_hex(colour) for colour in package.palette],
        "min_distance": package.min_distance,
        "total_error": bundle.total_error,
        "locks": _locks_json(package, package.locks),
        "entries": bundle.legend,
        "pages": [
            {
                "page": page.index,
                "width": page.width,
                "height": page.height,
                "png": f"page_{page.index:02d}.png",
                "source_rgb": page.rgb_name,
                "source_labels": page.label_name,
            }
            for page in package.pages
        ],
        "download_url": (
            f"/api/packages/{package.id}/versions/{bundle.number}/download"
        ),
        "change": bundle.change,
        "solver": bundle.solver,
    }
