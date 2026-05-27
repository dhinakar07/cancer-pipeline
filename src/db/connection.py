"""
connection.py — PostgreSQL connection helpers for cancer-pipeline.

Provides:
  - get_engine()      → SQLAlchemy engine (used by pandas read_sql / to_sql)
  - get_connection()  → raw psycopg2 connection (used for DDL / bulk inserts)
  - execute_sql()     → convenience wrapper for one-off statements

All connection parameters are read from src/config.py which loads .env.
"""

import psycopg2
import psycopg2.extras          # for execute_values (bulk insert)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from loguru import logger

from src.config import DB_URL, DB_CONN_PARAMS


# ── SQLAlchemy engine (singleton pattern) ───────────────────
_engine: Engine | None = None


def get_engine(pool_size: int = 5, max_overflow: int = 10) -> Engine:
    """
    Return a (cached) SQLAlchemy engine connected to the cancer_pipeline DB.

    The engine is created once and reused across calls (connection pooling).

    Args:
        pool_size:    Number of persistent connections in the pool.
        max_overflow: Extra connections allowed beyond pool_size.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine
    if _engine is None:
        logger.info(f"Creating SQLAlchemy engine → {DB_URL.split('@')[-1]}")
        _engine = create_engine(
            DB_URL,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,   # test connection health before each use
        )
    return _engine


def get_connection() -> psycopg2.extensions.connection:
    """
    Return a raw psycopg2 connection.

    The caller is responsible for committing / rolling back and closing.

    Example:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM cancer.raw_records")
            conn.commit()
        finally:
            conn.close()
    """
    logger.debug(f"Opening psycopg2 connection to {DB_CONN_PARAMS['dbname']}")
    return psycopg2.connect(**DB_CONN_PARAMS)


def execute_sql(sql: str, params: tuple | None = None) -> None:
    """
    Execute a single SQL statement (DDL or DML) and commit.

    Args:
        sql:    SQL string, may contain %s placeholders.
        params: Optional tuple of values for parameterised queries.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
        logger.debug("SQL executed and committed successfully.")
    except Exception as exc:
        conn.rollback()
        logger.error(f"SQL execution failed: {exc}")
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """
    Quick health-check: returns True if the DB is reachable.

    Useful as a preflight check at the start of each Airflow task.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection OK.")
        return True
    except Exception as exc:
        logger.error(f"Database connection FAILED: {exc}")
        return False
