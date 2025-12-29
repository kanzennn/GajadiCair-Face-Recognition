import os
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

TEST_DATASET_DIR = "test_dataset"

def get_test_images():
    files = []
    for fname in os.listdir(TEST_DATASET_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            files.append(os.path.join(TEST_DATASET_DIR, fname))
    return files


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["status"] == "running"


@pytest.mark.parametrize("image_path", get_test_images())
def test_has_face(image_path):
    with open(image_path, "rb") as f:
        response = client.post(
            "/has-face",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")}
        )

    assert response.status_code == 200
    body = response.json()

    assert "has_face" in body
    assert body["has_face"] is True
    assert body["count"] >= 1
    assert isinstance(body["boxes"], list)


def test_allowed_gestures():
    response = client.get("/allowed-gestures")
    assert response.status_code == 200

    body = response.json()
    assert "allowed_gestures" in body
    assert body["total"] > 0
    assert "OK" in body["allowed_gestures"]


@pytest.mark.parametrize("image_path", get_test_images())
def test_recognize_face_and_gesture(image_path):
    with open(image_path, "rb") as f:
        response = client.post(
            "/recognize",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")}
        )

    assert response.status_code == 200
    body = response.json()

    assert "employee_id" in body
    assert body["confidence"] == "high"

    assert "face_location" in body
    assert all(k in body["face_location"] for k in ["x", "y", "w", "h"])

    assert "gestures_detected" in body
    assert isinstance(body["gestures_detected"], list)

    if len(body["gestures_detected"]) > 0:
        for g in body["gestures_detected"]:
            assert "hand" in g
            assert "gesture" in g
