from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from backend.api import prediction as prediction_api
from backend.database.models import Diagnosis, User


def register_and_login(client, username: str = "fieldtester") -> str:
    response = client.post(
        "/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": "secure-test-password"},
    )
    assert response.status_code == 201, response.text
    response = client.post(
        "/auth/login",
        json={"email": f"{username}@example.com", "password": "secure-test-password"},
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def make_png(size: tuple[int, int] = (96, 96)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, color=(48, 130, 70)).save(stream, format="PNG")
    return stream.getvalue()


def test_frontend_and_api_docs_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "CropCare" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/docs").status_code == 200


def test_register_login_and_current_user(client):
    token = register_and_login(client)
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["username"] == "fieldtester"


def test_login_requires_valid_email_and_password(client):
    register_and_login(client)
    response = client.post(
        "/auth/login",
        json={"email": "fieldtester@example.com", "password": "incorrect-password"},
    )
    assert response.status_code == 401

    missing_email = client.post("/auth/login", json={"password": "secure-test-password"})
    assert missing_email.status_code == 422


def test_duplicate_account_is_rejected(client):
    register_and_login(client)
    response = client.post(
        "/auth/register",
        json={"username": "fieldtester", "email": "other@example.com", "password": "secure-test-password"},
    )
    assert response.status_code == 400


def test_history_requires_authentication(client):
    assert client.get("/history").status_code == 401


def test_history_hides_legacy_absolute_windows_paths(client):
    token = register_and_login(client)
    session: Session = client.app.state.testing_session_local()
    try:
        user = session.query(User).filter_by(username="fieldtester").one()
        session.add(
            Diagnosis(
                user_id=user.id,
                crop="Tomato",
                disease="Healthy",
                confidence=0.98,
                image_path=r"C:\\private\\uploads\\legacy-leaf.png",
                disease_summary="legacy record",
            )
        )
        session.commit()
    finally:
        session.close()

    response = client.get("/history", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()[0]["image_path"] == "legacy-leaf.png"


def test_prediction_upload_persists_opaque_filename_and_history(client, tmp_path, monkeypatch):
    token = register_and_login(client)
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(prediction_api, "UPLOAD_DIR", upload_dir)

    class FakePredictor:
        def predict(self, image):
            assert image.size == (224, 224)
            return {"crop": "Tomato", "disease": "Healthy", "confidence": 0.98, "label": "Healthy_Tomato"}

    monkeypatch.setattr(prediction_api, "get_predictor", lambda: FakePredictor())
    response = client.post(
        "/predictions",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("../../leaf.png", make_png(), "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    stored_name = payload["image_path"]
    assert stored_name == Path(stored_name).name
    assert "/" not in stored_name and "\\" not in stored_name
    assert (upload_dir / stored_name).is_file()

    history = client.get("/history", headers={"Authorization": f"Bearer {token}"})
    assert history.status_code == 200
    assert len(history.json()) == 1
    assert history.json()[0]["disease"] == "Healthy"
    assert history.json()[0]["image_path"] == stored_name


def test_oversized_upload_is_rejected_before_image_decode(client, monkeypatch):
    token = register_and_login(client)
    response = client.post(
        "/predictions",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("large.png", b"x" * (5 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 413


def test_oversized_multipart_body_is_rejected_before_auth_or_parsing(client):
    response = client.post(
        "/predictions",
        files={"file": ("large.png", b"x" * (7 * 1024 * 1024), "image/png")},
    )
    assert response.status_code == 413


def test_invalid_image_and_extension_are_rejected(client, monkeypatch):
    token = register_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}
    bad_image = client.post("/predictions", headers=headers, files={"file": ("leaf.png", b"not an image", "image/png")})
    assert bad_image.status_code == 400
    bad_extension = client.post("/predictions", headers=headers, files={"file": ("leaf.gif", make_png(), "image/gif")})
    assert bad_extension.status_code == 400
