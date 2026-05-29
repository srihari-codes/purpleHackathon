"""
database.py — SQLite setup via SQLAlchemy (async).
Creates all tables and provides a session factory.
"""

import os
from sqlalchemy import (
    Column, String, Float, Boolean, Integer, DateTime, Text, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./store_intelligence.db")

# Sync engine (FastAPI startup is sync; we use thread-safe sessions)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────

class EventRecord(Base):
    __tablename__ = "events"

    event_id = Column(String, primary_key=True, index=True)
    store_id = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False)
    visitor_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    zone_id = Column(String, nullable=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    confidence = Column(Float, default=1.0)
    metadata_json = Column(JSON, nullable=True)
    ingested_at = Column(DateTime, nullable=True)


class POSTransaction(Base):
    __tablename__ = "pos_transactions"

    transaction_id = Column(String, primary_key=True)
    store_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    basket_value_inr = Column(Float, nullable=False)


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def init_db():
    """Create all tables. Safe to call multiple times."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created / verified.")
    except OperationalError as exc:
        logger.error("Failed to initialise database: %s", exc)
        raise


def get_db():
    """FastAPI dependency: yield a DB session, close on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
