"""
[LAYER_START] Session 10: Database Engine & Session Management
Provides SQLAlchemy engine, session factory, and table creation.

Dev: SQLite (default — zero config, file-based)
Prod: PostgreSQL (swap via db_config.yaml)
"""

import logging
from pathlib import Path
from typing import Generator, Optional

import yaml
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_DB_CONFIG_PATH = "configs/db_config.yaml"

# Default SQLite URL (relative to project root)
DEFAULT_SQLITE_URL = "sqlite:///data/depresso.db"


def _load_db_config(config_path: str = DEFAULT_DB_CONFIG_PATH) -> dict:
    """Load database config from YAML, falling back to defaults."""
    path = Path(config_path)
    if not path.exists():
        logger.info(f"[DB] No config at {path}, using SQLite defaults")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _enable_sqlite_wal(dbapi_conn, connection_record):
    """Enable WAL mode for SQLite (better concurrent read performance)."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine(config_path: str = DEFAULT_DB_CONFIG_PATH):
    """
    Create SQLAlchemy engine from config.

    Reads db_config.yaml for connection URL and pool settings.
    Defaults to SQLite at data/depresso.db.
    """
    config = _load_db_config(config_path)
    url = config.get("url", DEFAULT_SQLITE_URL)
    pool_size = config.get("pool_size", 5)
    echo = config.get("echo", False)

    # Ensure parent directory exists for SQLite
    if url.startswith("sqlite:///"):
        db_path = Path(url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine_kwargs = {"echo": echo}

    # SQLite doesn't support pool_size/max_overflow
    if not url.startswith("sqlite"):
        engine_kwargs["pool_size"] = pool_size
        engine_kwargs["max_overflow"] = config.get("max_overflow", 10)
        engine_kwargs["pool_pre_ping"] = True

    engine = create_engine(url, **engine_kwargs)

    # Enable WAL for SQLite
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _enable_sqlite_wal)

    logger.info(f"[DB] Engine created: {url.split('?')[0]}")
    return engine


def get_session_factory(engine) -> sessionmaker:
    """Create a session factory bound to the given engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_session(session_factory: sessionmaker) -> Generator[Session, None, None]:
    """
    Context-manager-style session generator.

    Usage:
        factory = get_session_factory(engine)
        with next(get_session(factory)) as session:
            session.add(...)
            session.commit()
    """
    session = session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(config_path: str = DEFAULT_DB_CONFIG_PATH) -> sessionmaker:
    """
    Full database initialization: engine + tables + session factory.

    Returns:
        sessionmaker instance ready to create sessions.
    """
    from src.db.models import Base  # deferred to avoid circular import

    engine = get_engine(config_path)
    Base.metadata.create_all(engine)
    factory = get_session_factory(engine)
    logger.info("[DB] Database initialized — all tables created")
    return factory
