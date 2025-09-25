"""
DB Loader Agent
- Ensures the PostgreSQL database is reachable using SQLAlchemy.
- Reads DATABASE_URL from the provided state or environment.
- Returns a dict suitable for LangGraph state updates.
"""
from __future__ import annotations

import os
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


load_dotenv()


def get_database_url(state: Dict[str, Any] | None = None) -> str | None:
    if state and state.get("database_url"):
        return state["database_url"]
    return os.environ.get("DATABASE_URL")


def create_sqlalchemy_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True, future=True)


def check_database_connection(database_url: str) -> Tuple[bool, str]:
    try:
        engine = create_sqlalchemy_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except SQLAlchemyError as exc:
        return False, f"DB connection failed: {exc}"


def run(state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Node-style entrypoint. Returns keys:
      - database_url
      - db_status
    """
    database_url = get_database_url(state)
    if not database_url:
        return {"db_status": "DATABASE_URL not set"}

    ok, msg = check_database_connection(database_url)
    result: Dict[str, Any] = {"database_url": database_url, "db_status": msg}
    return result


if __name__ == "__main__":
    output = run({})
    print(output) 