"""
Schema/Stats Agent
- Introspects schema via SQLAlchemy inspector
- Computes light-weight stats per table/column
- Infers simple foreign key hints and column roles (measure/dimension/id)
- Returns report text along with structured schema, profiling, fk_hints, and roles
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


load_dotenv()


def _create_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True, future=True)


def describe_schema(database_url: str) -> Dict[str, Any]:
    engine = _create_engine(database_url)
    inspector = inspect(engine)
    schema_info: Dict[str, Any] = {}
    tables = inspector.get_table_names()  # include all tables

    with engine.connect() as conn:
        for table in tables:
            columns_meta = inspector.get_columns(table)
            columns: List[Dict[str, Any]] = [
                {"name": c.get("name"), "type": str(c.get("type"))} for c in columns_meta
            ]
            try:
                count_value = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                row_count = int(list(count_value)[0][0])
            except Exception:
                row_count = None
            schema_info[table] = {"columns": columns, "row_count": row_count}
    return schema_info


def compute_basic_stats(database_url: str, schema: Dict[str, Any], max_tables: int = 8, max_numeric_cols: int = 5) -> Dict[str, Any]:
    engine = _create_engine(database_url)
    profiling: Dict[str, Any] = {}

    with engine.connect() as conn:
        for table_idx, (table, meta) in enumerate(schema.items()):
            if table_idx >= max_tables:
                break
            numeric_cols: List[str] = []
            for col in meta.get("columns", []):
                t = (col.get("type") or "").lower()
                if any(k in t for k in ["int", "numeric", "float", "double", "real", "dec"]):
                    numeric_cols.append(col["name"])
            numeric_cols = numeric_cols[:max_numeric_cols]
            stats_for_table: Dict[str, Any] = {}
            for col in numeric_cols:
                try:
                    q = text(f"SELECT AVG({col}) AS mean, MIN({col}) AS min, MAX({col}) AS max FROM {table}")
                    row = conn.execute(q).mappings().first()
                    stats_for_table[col] = {
                        "mean": float(row["mean"]) if row["mean"] is not None else None,
                        "min": float(row["min"]) if row["min"] is not None else None,
                        "max": float(row["max"]) if row["max"] is not None else None,
                    }
                except Exception:
                    stats_for_table[col] = {"mean": None, "min": None, "max": None}
            if stats_for_table:
                profiling[table] = stats_for_table
    return profiling


def infer_fk_hints(schema: Dict[str, Any]) -> List[Dict[str, str]]:
    """Infer simple FK hints by name matching on *_id columns.
    Returns a list of {from_table, from_col, to_table, to_col}.
    """
    hints: List[Dict[str, str]] = []
    table_to_cols = {
        t: [c["name"] for c in meta.get("columns", [])]
        for t, meta in schema.items()
    }
    
    for from_table, cols in table_to_cols.items():
        for col in cols:
            lower = col.lower()
            if lower == "id":
                continue
            if lower.endswith("_id"):
                base = lower[:-3]
                # First try exact table name match
                if base in table_to_cols:
                    to_table = base
                    to_col = col if col in table_to_cols[base] else "id" if "id" in table_to_cols[base] else None
                    if to_col:
                        hints.append({
                            "from_table": from_table,
                            "from_col": col,
                            "to_table": to_table,
                            "to_col": to_col,
                        })
                else:
                    # Handle known missing tables with custom logic
                    if base == "customer":
                        # Skip customer_id references when customer table is missing
                        # This prevents incorrect FK hints like rental.customer_id -> payment.customer_id
                        continue
                    
                    # For other missing references, try to find a table with the column
                    found = False
                    for candidate_table, ccols in table_to_cols.items():
                        if candidate_table == from_table:  # Skip self-references
                            continue
                        if col in ccols:
                            hints.append({
                                "from_table": from_table,
                                "from_col": col,
                                "to_table": candidate_table,
                                "to_col": col,
                            })
                            found = True
                            break
                    
                    # If no exact column match, try primary key
                    if not found:
                        for candidate_table, ccols in table_to_cols.items():
                            if candidate_table == from_table:  # Skip self-references
                                continue
                            if "id" in ccols:
                                hints.append({
                                    "from_table": from_table,
                                    "from_col": col,
                                    "to_table": candidate_table,
                                    "to_col": "id",
                                })
                                break
    return hints


def infer_roles(schema: Dict[str, Any]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for table, meta in schema.items():
        for col_meta in meta.get("columns", []):
            col = col_meta.get("name")
            ctype = (col_meta.get("type") or "").lower()
            key = f"{table}.{col}"
            lower = col.lower()
            if lower == "id" or lower.endswith("_id"):
                roles[key] = "id"
            elif any(k in ctype for k in ["int", "numeric", "float", "double", "real", "dec"]):
                roles[key] = "measure"
            else:
                roles[key] = "dimension"
    return roles


def build_report(schema: Dict[str, Any], profiling: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Dataset Overview")
    for table, meta in schema.items():
        lines.append(f"- {table}: rows={meta.get('row_count')}, cols={len(meta.get('columns', []))}")
    lines.append("")
    if profiling:
        lines.append("Basic Stats (sample)")
        for table, cols in profiling.items():
            lines.append(f"- {table}")
            for col, stats in cols.items():
                lines.append(f"  - {col}: mean={stats.get('mean')}, min={stats.get('min')}, max={stats.get('max')}")
    return "\n".join(lines)


def run(state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    database_url = (state or {}).get("database_url") or os.environ.get("DATABASE_URL")
    if not database_url:
        return {"error": "DATABASE_URL not set"}

    schema = describe_schema(database_url)
    profiling = compute_basic_stats(database_url, schema)
    fk_hints = infer_fk_hints(schema)
    roles = infer_roles(schema)
    report = build_report(schema, profiling)
    return {
        "schema": schema,
        "profiling": profiling,
        "fk_hints": fk_hints,
        "roles": roles,
        "report": report,
    }


if __name__ == "__main__":
    result = run()
    print(result["report"]) 