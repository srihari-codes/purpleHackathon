"""
conftest.py — Shared pytest fixtures.
Key insight: Each sqlite:///:memory: connection gets a separate DB.
We use a temp file so all connections (lifespan init_db + test sessions) share tables.
"""

import os
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

import app.database as db_module
from app.database import Base, get_db
from app.main import app


@pytest.fixture(scope="function")
def db_engine(monkeypatch, tmp_path):
    db_file = str(tmp_path / "test.db")
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)

    # Patch the module-level engine + SessionLocal so init_db() hits the same file
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)
    monkeypatch.setattr(db_module, "DATABASE_URL", db_url)

    yield engine

    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="function")
def test_client(db_engine, db_session):
    TestSession = sessionmaker(bind=db_engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client, db_session
    app.dependency_overrides.clear()


