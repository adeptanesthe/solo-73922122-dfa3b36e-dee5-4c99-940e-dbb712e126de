# E-paper colouring service

FastAPI service that maps a small set of numbered page regions onto a fixed
e-paper palette. It reads 1–4 RGB page PNGs plus matching 8-bit grayscale
label maps (side at most 128 px, at most 12 label values in the whole set),
and publishes a consistent package: recoloured PNG pages, a legend, and a
version number.

## What it guarantees

* One colour per label across the whole set (`same label -> same colour`).
* Labels whose pixels share an edge on any page must receive palette colours
  whose Euclidean RGB distance is at least `min_distance`.
* Among all feasible assignments it returns one with the minimum **total
  squared RGB error** over every pixel in the set. The search is branch and
  bound over the ≤12-label adjacency graph with a per-vertex lower bound, a
  greedy upper bound, MRV branching and forward checking; it never enumerates
  all `K**N` assignments.
* Locks (label → palette colour) are stored per package. Locking or unlocking
  any label re-solves the whole set, so colours on other pages can change.
* An edit is all-or-nothing. If the new locks have no feasible assignment the
  request returns `409` with the conflict reason, and the current package and
  locks are untouched.
* Every successful create/edit publishes a new integer version. Only the
  current version can be downloaded; a superseded version returns `409`
  `version_conflict` with the current version number.

## Run with Docker Compose

```bash
docker compose up --build
# service on http://localhost:8000, docs at /docs
```

## Run locally (without Docker)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000
```

Tests:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

## API

### `POST /api/packages` (multipart/form-data)

| field | meaning |
| --- | --- |
| `rgb_pages` | 1–4 PNG files, 8-bit RGB, each side ≤128 px |
| `label_pages` | same count, same sizes, 8-bit grayscale (`L`) |
| `palette` | JSON array of 2–6 colours, `#rrggbb` or `[r,g,b]` |
| `min_distance` | number > 0, minimum RGB Euclidean distance for touching labels |
| `locks` | optional JSON object `{"160": "#2f5d8a"}`; colours must be in the palette |

Response `201`: package id, version, legend entries (label, colour, pages),
total squared error, locks and the versioned download URL. `422` for invalid
input, `409` when no feasible colouring exists.

### `POST /api/packages/{id}/locks`
Body: `{"label": 160, "colour": "#2f5d8a"}`. Lock or re-lock one label.

### `DELETE /api/packages/{id}/locks/{label}`
Remove one lock.

Both edits publish a new version on success. On failure they return `409`
with `details` (conflict type, labels, colour pairs, whether the locks caused
it), `kept_version`, `kept_locks` and `kept_download_url`.

### Downloads (current version only)

* `GET /api/packages/{id}/versions/{v}/download` — ZIP with `page_NN.png`,
  `legend.json`, `legend.png`, `manifest.json`
* `GET /api/packages/{id}/versions/{v}/legend.json`
* `GET /api/packages/{id}/versions/{v}/pages/{n}`

Stale versions return `409`; versions never published return `404`.

## Demo: lock causes a cross-page colour change

```bash
python3 demo/make_demo_pages.py            # builds demo/pages/*.png + manifest
docker compose up --build                  # in another terminal
bash demo/request_demo.sh                  # BASE=... to target another host
```

The demo creates a three-page package with a five-colour palette. Locking the
seal label to the sky blue forces the rule label on page 1 and page 2 to move
to the ochre colour, because those labels touch on every page where both
appear. It then requests an impossible lock (rule and seal both blue while
touching), shows the `409` with preserved version/locks, and shows a stale
version download returning `409`.

## Solver

The label adjacency graph has at most 12 vertices. The solver splits it into
connected components, then for each component:

1. builds exact per-label costs `Σ (r-pr)² + (g-pg)² + (b-pb)²` over all
   pixels of that label (integer arithmetic);
2. computes an admissible lower bound `Σ min_c cost(label, c)` and a greedy
   upper bound from four vertex orders;
3. runs depth-first branch and bound with MRV variable ordering, forward
   checking on every edge, and a node budget as a defensive guard.

Greedy hitting the lower bound returns immediately, which is the common case.
Response bodies and `legend.json` include `solver.search_nodes` so the search
effort is visible.

### Response shape

```jsonc
{
  "package_id": "…",
  "version": 2,
  "total_error": 80647488,
  "palette": ["#101010", "…"],
  "min_distance": 200.0,
  "locks": {"160": "#2f5d8a"},
  "entries": [{"label": 160, "colour": "#2f5d8a", "rgb": [47,93,138],
               "locked": true, "pixels": 240, "pages": [1,2,3]}],
  "download_url": "/api/packages/…/versions/2/download",
  "change": {"type": "lock", "label": 160,
             "colours": {"160": {"before": "#d8b41e", "after": "#2f5d8a"}},
             "previous_version": 1},
  "solver": {"search_nodes": 0, "components": 1}
}
```

Conflict responses (`409`) carry `error.code = "infeasible"` with
`error.details` (type, involved labels, colour pairs, whether the locks caused
the failure, and the largest mutually touching clique when that is the
blocker), plus `kept_version`, `kept_locks` and `kept_download_url`.

## Limits and storage

PNG handling is in-memory via Pillow, and packages are kept in memory. The
service is single-process by design: versions are not shared between
replicas, and a restart clears packages. The input contract (≤4 pages,
≤128 px, ≤12 labels, 2–6 colours) keeps analysis and exact optimisation
cheap: a full 4 × 128 × 128 page set analyses in ~0.25 s and solves in
milliseconds.
