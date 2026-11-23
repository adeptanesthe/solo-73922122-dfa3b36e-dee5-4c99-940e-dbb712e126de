from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _colours(image: Image.Image) -> set[tuple[int, int, int]]:
    raw = image.tobytes()
    return {(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)}


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_pages(rgb_arrays, label_arrays):
    """Build parallel RGB and label PNG upload files."""
    rgb_files = []
    label_files = []
    for index, (rgb_rows, label_rows) in enumerate(
        zip(rgb_arrays, label_arrays), start=1
    ):
        height = len(rgb_rows)
        width = len(rgb_rows[0])
        rgb = Image.new("RGB", (width, height))
        rgb.putdata([tuple(px) for row in rgb_rows for px in row])
        labels = Image.new("L", (width, height))
        labels.putdata([value for row in label_rows for value in row])
        rgb_files.append(
            ("rgb_pages", (f"page{index}.png", png_bytes(rgb), "image/png"))
        )
        label_files.append(
            ("label_pages", (f"page{index}-labels.png", png_bytes(labels), "image/png"))
        )
    return rgb_files, label_files


def submit(client, rgb_files, label_files, palette, min_distance=60, locks=None):
    data = {
        "palette": __import__("json").dumps(palette),
        "min_distance": str(min_distance),
    }
    if locks is not None:
        data["locks"] = __import__("json").dumps(locks)
    return client.post(
        "/api/packages",
        data=data,
        files=rgb_files + label_files,
    )


TWO_BY_TWO_RGB = [
    [[(250, 250, 250), (10, 10, 10)], [(250, 250, 250), (10, 10, 10)]],
]
TWO_BY_TWO_LABELS = [[[1, 2], [1, 2]]]
PALETTE = ["#000000", "#ffffff"]


def test_create_and_download(client: TestClient) -> None:
    rgb_files, label_files = make_pages(TWO_BY_TWO_RGB, TWO_BY_TWO_LABELS)
    response = submit(client, rgb_files, label_files, PALETTE)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["version"] == 1
    assert len(body["entries"]) == 2
    colours = {entry["label"]: entry["colour"] for entry in body["entries"]}
    assert colours[1] == "#ffffff"
    assert colours[2] == "#000000"
    # Total squared error: two white pixels off by 5 per channel and two black
    # pixels off by 10 per channel: 2*(3*25) + 2*(3*100).
    assert body["total_error"] == 750

    package_id = body["package_id"]
    download = client.get(f"/api/packages/{package_id}/versions/1/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    names = sorted(archive.namelist())
    assert names == [
        "legend.json",
        "legend.png",
        "manifest.json",
        "page_01.png",
    ]
    page = Image.open(io.BytesIO(archive.read("page_01.png")))
    assert page.mode == "RGB"
    assert page.size == (2, 2)
    pixels = page.tobytes()
    assert pixels[0:3] == bytes((255, 255, 255))
    assert pixels[3:6] == bytes((0, 0, 0))


def test_ssim_style_error_is_minimal(client: TestClient) -> None:
    """A three-label line should pick the closest feasible colours."""
    rgb_rows = [[(250, 250, 250), (120, 120, 120), (4, 4, 4)]]
    label_rows = [[5, 6, 7]]
    rgb_files, label_files = make_pages([rgb_rows], [label_rows])
    response = submit(
        client,
        rgb_files,
        label_files,
        ["#000000", "#808080", "#ffffff"],
        min_distance=100,
    )
    assert response.status_code == 201, response.text
    body = response.json()
    # 75 for the white label, 192 for the grey label, 48 for the black label.
    assert body["total_error"] == 75 + 192 + 48


def test_locked_colour_is_respected(client: TestClient) -> None:
    rgb_files, label_files = make_pages(TWO_BY_TWO_RGB, TWO_BY_TWO_LABELS)
    response = submit(client, rgb_files, label_files, PALETTE, locks={"1": "#000000"})
    assert response.status_code == 201, response.text
    body = response.json()
    colours = {entry["label"]: entry["colour"] for entry in body["entries"]}
    assert colours[1] == "#000000"
    assert colours[2] == "#ffffff"
    assert body["locks"] == {"1": "#000000"}


def test_lock_conflict_keeps_version(client: TestClient) -> None:
    palette = ["#000000", "#0a0000", "#ffffff"]
    rgb_files, label_files = make_pages(TWO_BY_TWO_RGB, TWO_BY_TWO_LABELS)
    response = submit(
        client,
        rgb_files,
        label_files,
        palette,
        min_distance=20,
        locks={"1": "#000000"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    package_id = body["package_id"]
    version = body["version"]

    conflict = client.post(
        f"/api/packages/{package_id}/locks",
        json={"label": 2, "colour": "#0a0000"},
    )
    assert conflict.status_code == 409, conflict.text
    payload = conflict.json()["error"]
    assert payload["code"] == "infeasible"
    assert payload["kept_version"] == version
    assert payload["kept_locks"] == {"1": "#000000"}
    assert payload["details"]["labels"] == [1, 2]

    # The package still serves its previous version.
    still = client.get(f"/api/packages/{package_id}")
    assert still.status_code == 200
    assert still.json()["version"] == version
    assert still.json()["locks"] == {"1": "#000000"}
    archive = client.get(f"/api/packages/{package_id}/versions/{version}/download")
    assert archive.status_code == 200

    # Unknown palette colour is rejected without touching state.
    bad = client.post(
        f"/api/packages/{package_id}/locks",
        json={"label": 2, "colour": "#123456"},
    )
    assert bad.status_code == 422
    assert client.get(f"/api/packages/{package_id}").json()["version"] == version


def test_version_bumps_and_old_versions_conflict(client: TestClient) -> None:
    palette = ["#000000", "#0a0000", "#ffffff"]
    rgb_files, label_files = make_pages(TWO_BY_TWO_RGB, TWO_BY_TWO_LABELS)
    created = submit(client, rgb_files, label_files, palette, min_distance=20)
    assert created.status_code == 201, created.text
    package_id = created.json()["package_id"]

    locked = client.post(
        f"/api/packages/{package_id}/locks",
        json={"label": 1, "colour": "#0a0000"},
    )
    assert locked.status_code == 200, locked.text
    assert locked.json()["version"] == 2
    assert {e["label"]: e["colour"] for e in locked.json()["entries"]}[1] == "#0a0000"

    stale = client.get(f"/api/packages/{package_id}/versions/1/download")
    assert stale.status_code == 409
    error = stale.json()["error"]
    assert error["code"] == "version_conflict"
    assert error["current_version"] == 2

    fresh = client.get(f"/api/packages/{package_id}/versions/2/download")
    assert fresh.status_code == 200

    unlocked = client.delete(f"/api/packages/{package_id}/locks/1")
    assert unlocked.status_code == 200
    assert unlocked.json()["version"] == 3
    assert (
        client.get(f"/api/packages/{package_id}/versions/2/download").status_code == 409
    )


def test_cross_page_single_colour_per_label(client: TestClient) -> None:
    # Page 1: labels 1 and 2 touch. Page 2: label 1 only.
    rgb_arrays = [
        [[(250, 250, 250), (10, 10, 10)], [(250, 250, 250), (10, 10, 10)]],
        [[(240, 240, 240), (240, 240, 240)], [(240, 240, 240), (240, 240, 240)]],
    ]
    label_arrays = [
        [[1, 2], [1, 2]],
        [[1, 1], [1, 1]],
    ]
    rgb_files, label_files = make_pages(rgb_arrays, label_arrays)
    palette = ["#000000", "#ffffff"]
    created = submit(client, rgb_files, label_files, palette)
    assert created.status_code == 201, created.text
    package_id = created.json()["package_id"]

    locked = client.post(
        f"/api/packages/{package_id}/locks",
        json={"label": 1, "colour": "#000000"},
    )
    assert locked.status_code == 200, locked.text
    entries = {e["label"]: e for e in locked.json()["entries"]}
    assert entries[1]["colour"] == "#000000"
    assert entries[1]["pages"] == [1, 2]
    assert entries[2]["colour"] == "#ffffff"
    # Locking label 1 forced label 2 across pages even though page 2 never
    # shows label 2; the legend reflects one colour per label for the set.
    download = client.get(f"/api/packages/{package_id}/versions/2/download")
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    page1 = Image.open(io.BytesIO(archive.read("page_01.png")))
    page2 = Image.open(io.BytesIO(archive.read("page_02.png")))
    assert _colours(page1) == {(0, 0, 0), (255, 255, 255)}
    assert _colours(page2) == {(0, 0, 0)}


def test_rejects_bad_inputs(client: TestClient) -> None:
    rgb_files, label_files = make_pages(TWO_BY_TWO_RGB, TWO_BY_TWO_LABELS)
    # Grayscale original instead of RGB.
    gray = Image.new("L", (2, 2), 0)
    bad_rgb = [("rgb_pages", ("gray.png", png_bytes(gray), "image/png"))]
    resp = submit(client, bad_rgb, label_files, PALETTE)
    assert resp.status_code == 422
    assert "8-bit RGB" in resp.json()["error"]["message"]

    # Mismatched sizes.
    label_big = Image.new("L", (3, 3), 0)
    bad_label = [("label_pages", ("bad.png", png_bytes(label_big), "image/png"))]
    resp = submit(client, rgb_files, bad_label, PALETTE)
    assert resp.status_code == 422
    assert "does not match" in resp.json()["error"]["message"]

    # Too many colours.
    resp = submit(client, rgb_files, label_files, ["#000000"])
    assert resp.status_code == 422

    # Non-positive distance.
    resp = submit(client, rgb_files, label_files, PALETTE, min_distance=0)
    assert resp.status_code == 422

    # Lock colour outside palette.
    resp = submit(client, rgb_files, label_files, PALETTE, locks={"1": "#123456"})
    assert resp.status_code == 422

    # Size limit.
    big = Image.new("RGB", (129, 1), (0, 0, 0))
    big_labels = Image.new("L", (129, 1), 0)
    files = [
        ("rgb_pages", ("big.png", png_bytes(big), "image/png")),
        ("label_pages", ("big-l.png", png_bytes(big_labels), "image/png")),
    ]
    resp = submit(client, files[:1], files[1:], PALETTE)
    assert resp.status_code == 422
    assert "128" in resp.json()["error"]["message"]


def test_too_many_pages(client: TestClient) -> None:
    rgb_arrays = [[[(1, 1, 1)]] for _ in range(5)]
    label_arrays = [[[1]] for _ in range(5)]
    rgb_files, label_files = make_pages(rgb_arrays, label_arrays)
    resp = submit(client, rgb_files, label_files, PALETTE)
    assert resp.status_code == 422


def test_too_many_labels(client: TestClient) -> None:
    rows = [[[(100, 100, 100) for _ in range(13)]]]
    labels = [[list(range(1, 14))]]
    rgb_files, label_files = make_pages(rows, labels)
    resp = submit(client, rgb_files, label_files, PALETTE)
    assert resp.status_code == 422
    assert "12" in resp.json()["error"]["message"]


def test_checkerboard_needs_two_colours(client: TestClient) -> None:
    rgb_rows = [[(0, 0, 0), (255, 255, 255)], [(255, 255, 255), (0, 0, 0)]]
    label_rows = [[1, 2], [2, 1]]
    rgb_files, label_files = make_pages([rgb_rows], [label_rows])
    resp = submit(client, rgb_files, label_files, PALETTE)
    assert resp.status_code == 201, resp.text
    assert resp.json()["total_error"] == 0


def test_unlock_unknown_label(client: TestClient) -> None:
    rgb_files, label_files = make_pages(TWO_BY_TWO_RGB, TWO_BY_TWO_LABELS)
    created = submit(client, rgb_files, label_files, PALETTE)
    package_id = created.json()["package_id"]
    resp = client.delete(f"/api/packages/{package_id}/locks/1")
    assert resp.status_code == 422
    resp = client.delete(f"/api/packages/{package_id}/locks/99")
    assert resp.status_code == 422


def test_solve_is_skipped_when_lock_hits_lower_bound(client: TestClient) -> None:
    """Locking the cheapest colour must not cost extra search work."""
    rgb_rows = [[(255, 255, 255), (255, 255, 255), (0, 0, 0), (0, 0, 0)]]
    label_rows = [[1, 1, 2, 2]]
    rgb_files, label_files = make_pages([rgb_rows], [label_rows])
    palette = ["#000000", "#ffffff"]
    created = submit(client, rgb_files, label_files, palette)
    assert created.status_code == 201, created.text
    pid = created.json()["package_id"]
    locked = client.post(
        f"/api/packages/{pid}/locks", json={"label": 1, "colour": "#ffffff"}
    )
    assert locked.status_code == 200
    assert locked.json()["total_error"] == 0


def test_unpublished_version_is_404(client: TestClient) -> None:
    rgb_files, label_files = make_pages(TWO_BY_TWO_RGB, TWO_BY_TWO_LABELS)
    created = submit(client, rgb_files, label_files, PALETTE)
    pid = created.json()["package_id"]
    resp = client.get(f"/api/packages/{pid}/versions/99/download")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "version_not_found"


def test_rendered_pixels_are_palette_colours(client: TestClient) -> None:
    rgb_rows = [[(250, 250, 250), (10, 10, 10)]]
    label_rows = [[1, 2]]
    rgb_files, label_files = make_pages([rgb_rows], [label_rows])
    created = submit(client, rgb_files, label_files, PALETTE)
    assert created.status_code == 201, created.text
    body = created.json()
    archive = zipfile.ZipFile(io.BytesIO(client.get(body["download_url"]).content))
    page = Image.open(io.BytesIO(archive.read("page_01.png")))
    assert _colours(page) <= {(0, 0, 0), (255, 255, 255)}


def test_relock_records_before_and_after(client: TestClient) -> None:
    rgb_files, label_files = make_pages(TWO_BY_TWO_RGB, TWO_BY_TWO_LABELS)
    created = submit(client, rgb_files, label_files, PALETTE)
    pid = created.json()["package_id"]
    first = client.post(
        f"/api/packages/{pid}/locks", json={"label": 1, "colour": "#000000"}
    )
    assert first.status_code == 200
    second = client.post(
        f"/api/packages/{pid}/locks", json={"label": 1, "colour": "#ffffff"}
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["version"] == 3
    assert body["change"]["type"] == "lock"
    assert body["change"]["colours"]["1"] == {
        "before": "#000000",
        "after": "#ffffff",
    }
    assert {e["label"]: e["colour"] for e in body["entries"]}[1] == "#ffffff"


def test_clique_conflict_is_explained(client: TestClient) -> None:
    # Three labels in a row: 1-2 and 2-3 touch, so with only black/white far
    # enough apart the chain is colourable; make it a ring by wrapping the
    # row so 3 also touches 1.
    rgb_rows = [[(0, 0, 0)] * 4]
    label_rows = [[1, 2, 3, 1]]
    rgb_files, label_files = make_pages([rgb_rows], [label_rows])
    resp = submit(
        client,
        rgb_files,
        label_files,
        ["#000000", "#808080", "#ffffff"],
        min_distance=300,
    )
    # #808080 is 127 from black and 127 from white, 3 labels need 3 mutually
    # distant colours, but only black/white are far enough apart.
    assert resp.status_code == 409, resp.text
    error = resp.json()["error"]
    assert error["details"]["clique"]["required_colours"] == 3
    assert error["details"]["clique"]["available_colours"] == 2
    assert error["details"]["locks_cause"] is False
    assert error["details"]["locks"] == {}


def test_validation_errors_use_error_envelope(client: TestClient) -> None:
    resp = client.post(
        "/api/packages/does-not-exist/locks",
        json={"label": -1, "colour": "#000000"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_input"
    assert "label" in resp.json()["error"]["message"]

    resp = client.post(
        "/api/packages/does-not-exist/locks",
        json={"label": 1, "colour": "#000000"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "package_not_found"

    resp = client.post("/api/packages", data={})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_input"


def test_total_error_matches_rendered_pixels(client: TestClient) -> None:
    rgb_rows = [
        [(x * 41 % 256, y * 63 % 256, (x + y) * 37 % 256) for x in range(6)]
        for y in range(4)
    ]
    label_rows = [[1 if x < 3 else 2 for x in range(6)] for _ in range(4)]
    rgb_files, label_files = make_pages([rgb_rows], [label_rows])
    created = submit(
        client,
        rgb_files,
        label_files,
        ["#000000", "#ffffff", "#ff0000"],
        min_distance=150,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    archive = zipfile.ZipFile(io.BytesIO(client.get(body["download_url"]).content))
    rendered = Image.open(io.BytesIO(archive.read("page_01.png"))).tobytes()
    original = bytes(channel for row in rgb_rows for pixel in row for channel in pixel)
    recomputed = sum((a - b) ** 2 for a, b in zip(original, rendered))
    assert recomputed == body["total_error"]
