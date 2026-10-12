from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["APP_ENV"] = "testing"
os.environ["SECRET_KEY"] = "test-only-secret-key-not-for-production-0123456789"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from backend.database.database import Base, get_db
    import main as application

    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'test.sqlite').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    def create_test_tables():
        Base.metadata.create_all(bind=test_engine)

    application.app.dependency_overrides[get_db] = override_get_db
    application.app.state.testing_session_local = TestingSessionLocal
    monkeypatch.setattr(application, "init_db", create_test_tables)
    try:
        with TestClient(application.app) as test_client:
            yield test_client
    finally:
        application.app.dependency_overrides.clear()
        del application.app.state.testing_session_local
        test_engine.dispose()
