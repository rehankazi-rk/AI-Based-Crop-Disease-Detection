from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from backend.services import predictor
from backend.services import image_processor
from backend.services.image_processor import validate_image_file


def test_image_pixel_limit_rejects_extreme_dimensions(tmp_path, monkeypatch):
    from PIL import Image

    image_path = tmp_path / "large-dimensions.png"
    Image.new("RGB", (32, 32)).save(image_path)
    monkeypatch.setattr(image_processor, "MAX_IMAGE_PIXELS", 100)
    valid, message, dimensions = validate_image_file("leaf.png", image_path.read_bytes())
    assert not valid
    assert dimensions is None
    assert message


def test_predictor_is_constructed_once(monkeypatch):
    calls = []

    class FakePredictor:
        def __init__(self):
            calls.append(self)

    monkeypatch.setattr(predictor, "TransferLearningPredictor", FakePredictor)
    predictor.get_predictor.cache_clear()
    try:
        first = predictor.get_predictor()
        second = predictor.get_predictor()
        assert first is second
        assert len(calls) == 1
    finally:
        predictor.get_predictor.cache_clear()


def test_production_refuses_missing_or_placeholder_jwt_secret():
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["APP_ENV"] = "production"
    env.pop("SECRET_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", "import backend.api.auth"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SECRET_KEY" in result.stderr
