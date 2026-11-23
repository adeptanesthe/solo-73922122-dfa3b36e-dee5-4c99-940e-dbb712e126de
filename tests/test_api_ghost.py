"""End-to-end tests for the afterimage verification endpoint."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _png_gray(pixels, width, height):
    image = Image.new("L", (width, height))
    image.putdata(list(pixels))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_rgb(pixels, width, height):
    image = Image.new("RGB", (width, height))
    image.putdata(list(pixels))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _photo_png(new, old, width, height, ratio, gain, offset):
    pixels = []
    for new_value, old_value in zip(new, old):
        value = gain * (new_value + ratio * (old_value - new_value)) + offset
        pixels.append(max(0, min(255, round(value))))
    return _png_gray(pixels, width, height)


@pytest.fixture()
def package(client: TestClient):
    # 16x16 page: dark left half, bright right half.
    width = height = 16
    rgb = [
        (40, 40, 40) if x < 8 else (210, 210, 210)
        for _ in range(height)
        for x in range(width)
    ]
    labels = [1 if x < 8 else 2 for _ in range(height) for x in range(width)]
    response = client.post(
        "/api/packages",
        data={
            "palette": '["#000000", "#ffffff"]',
            "min_distance": "60",
        },
        files=[
            ("rgb_pages", ("page1.png", _png_rgb(rgb, width, height), "image/png")),
            ("label_pages", ("page1-l.png", _png_gray(labels, width, height), "image/png")),
        ],
    )
    assert response.status_code == 201, response.text
    return response.json()


def _old_image(width, height):
    # New image renders to black/white halves; the previously delivered image
    # differs in the central 8x8 zone and agrees elsewhere.
    old = []
    for y in range(height):
        for x in range(width):
            new_value = 0 if x < 8 else 255
            if 4 <= x < 12 and 4 <= y < 12:
                old.append(255 if new_value == 0 else 0)
            else:
                old.append(new_value)
    return old


def test_ghost_check_reflash_flow(client: TestClient, package) -> None:
    width = height = 16
    body = package
    pid = body["package_id"]
    version = body["version"]
    new = [0 if x < 8 else 255 for _ in range(height) for x in range(width)]
    old = _old_image(width, height)
    photo = _photo_png(new, old, width, height, ratio=0.30, gain=1.0, offset=0)

    response = client.post(
        f"/api/packages/{pid}/versions/{version}/pages/1/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("shot1.png", photo, "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["decision"] == "reflash"
    assert payload["residual_ratio"]["min"] >= 0.20 - 1e-6
    assert 2 <= len(payload["calibration_points"]) <= 8
    assert len({point["gray"] for point in payload["calibration_points"]}) >= 2
    roi = payload["region_of_interest"]
    assert roi["width"] <= 16 and roi["height"] <= 16
    assert roi["changed_pixels"] >= 1 and roi["gray_levels"] >= 2
    assert payload["photos"][0]["gain_a"] == pytest.approx(1.0, abs=0.03)
    assert payload["photos"][0]["max_abs_residual"] <= 2.0 + 1e-6
    assert payload["tolerance"] == 2.0
    assert payload["reflash_threshold"] == 0.20


def test_ghost_check_two_photos_and_no_reflash(client: TestClient, package) -> None:
    width = height = 16
    pid = package["package_id"]
    version = package["version"]
    new = [0 if x < 8 else 255 for _ in range(height) for x in range(width)]
    old = _old_image(width, height)
    first = _photo_png(new, old, width, height, 0.05, 0.9, 4)
    second = _photo_png(new, old, width, height, 0.05, 0.8, 6)

    response = client.post(
        f"/api/packages/{pid}/versions/{version}/pages/1/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("shot1.png", first, "image/png")),
            ("photos", ("shot2.png", second, "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["decision"] == "no_reflash"
    assert len(payload["photos"]) == 2
    assert payload["residual_ratio"]["max"] < 0.20


def test_evidence_conflict_returns_409(client: TestClient, package) -> None:
    width = height = 16
    pid = package["package_id"]
    version = package["version"]
    new = [0 if x < 8 else 255 for _ in range(height) for x in range(width)]
    old = _old_image(width, height)
    first = _photo_png(new, old, width, height, 0.0, 1.0, 0)
    second = _photo_png(new, old, width, height, 0.6, 1.0, 0)

    response = client.post(
        f"/api/packages/{pid}/versions/{version}/pages/1/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("shot1.png", first, "image/png")),
            ("photos", ("shot2.png", second, "image/png")),
        ],
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "evidence_conflict"
    assert len(error["photo_intervals"]) == 2


def test_stale_version_conflicts(client: TestClient, package) -> None:
    width = height = 16
    pid = package["package_id"]
    new = [0 if x < 8 else 255 for _ in range(height) for x in range(width)]
    old = _old_image(width, height)
    photo = _photo_png(new, old, width, height, 0.3, 1.0, 0)

    # Bump the package to version 2.
    locked = client.post(
        f"/api/packages/{pid}/locks",
        json={"label": 1, "colour": "#000000"},
    )
    assert locked.status_code == 200
    response = client.post(
        f"/api/packages/{pid}/versions/1/pages/1/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("shot1.png", photo, "image/png")),
        ],
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "version_conflict"
    assert error["current_version"] == 2


def test_unknown_package_and_page_are_404(client: TestClient, package) -> None:
    width = height = 16
    old = _old_image(width, height)
    response = client.post(
        "/api/packages/does-not-exist/versions/1/pages/1/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("shot1.png", _png_gray(old, width, height), "image/png")),
        ],
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "package_not_found"

    pid = package["package_id"]
    response = client.post(
        f"/api/packages/{pid}/versions/1/pages/9/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("shot1.png", _png_gray(old, width, height), "image/png")),
        ],
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "page_not_found"


def test_unknown_version_is_404_not_package_error(client: TestClient, package) -> None:
    width = height = 16
    pid = package["package_id"]
    old = _old_image(width, height)
    response = client.post(
        f"/api/packages/{pid}/versions/99/pages/1/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("shot1.png", _png_gray(old, width, height), "image/png")),
        ],
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "version_not_found"


def test_size_mismatch_is_422(client: TestClient, package) -> None:
    width = height = 16
    pid = package["package_id"]
    version = package["version"]
    old = _old_image(width, height)
    small = _png_gray([100] * 64, 8, 8)
    response = client.post(
        f"/api/packages/{pid}/versions/{version}/pages/1/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("shot1.png", small, "image/png")),
        ],
    )
    assert response.status_code == 422
    assert "does not match" in response.json()["error"]["message"]


def test_too_many_photos_is_422(client: TestClient, package) -> None:
    width = height = 16
    pid = package["package_id"]
    version = package["version"]
    old = _old_image(width, height)
    photo = _png_gray(old, width, height)
    response = client.post(
        f"/api/packages/{pid}/versions/{version}/pages/1/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("a.png", photo, "image/png")),
            ("photos", ("b.png", photo, "image/png")),
            ("photos", ("c.png", photo, "image/png")),
        ],
    )
    assert response.status_code == 422
    assert "photographs" in response.json()["error"]["message"]


def test_rgb_photo_is_accepted_and_converted(client: TestClient, package) -> None:
    width = height = 16
    pid = package["package_id"]
    version = package["version"]
    new = [0 if x < 8 else 255 for _ in range(height) for x in range(width)]
    old = _old_image(width, height)
    gray_photo = [
        max(0, min(255, round(n + 0.3 * (o - n)))) for n, o in zip(new, old)
    ]
    rgb_photo = [(g, g, g) for g in gray_photo]
    response = client.post(
        f"/api/packages/{pid}/versions/{version}/pages/1/ghost-check",
        files=[
            ("old_image", ("old.png", _png_gray(old, width, height), "image/png")),
            ("photos", ("shot1.png", _png_rgb(rgb_photo, width, height), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    assert response.json()["decision"] == "reflash"
