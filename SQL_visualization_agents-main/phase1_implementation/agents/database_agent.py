"""
Database Connection & Schema Agent (LangGraph Implementation)
Combines functionality from db_loader.py and schema_stats.py
"""

import os
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables from .env file FIRST
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from ..state_schemas import DatabaseState, DVDRentalSystemState, convert_numpy_types
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from state_schemas import DatabaseState, DVDRentalSystemState, convert_numpy_types


# ============================================================================
# Core Database Functions (from existing code)
# ============================================================================

def _create_engine(database_url: str) -> Engine:
    """Create SQLAlchemy engine with connection pooling"""
    return create_engine(database_url, pool_pre_ping=True, future=True)


def _check_database_connection(database_url: str) -> tuple[bool, str]:
    """Test database connection"""
    try:
        engine = _create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connection successful"
    except SQLAlchemyError as exc:
        return False, f"DB connection failed: {exc}"


def _describe_schema(database_url: str) -> Dict[str, Any]:
    """Extract table schema information"""
    engine = _create_engine(database_url)
    inspector = inspect(engine)
    schema_info: Dict[str, Any] = {}
    tables = inspector.get_table_names()

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


def _compute_basic_stats(database_url: str, schema: Dict[str, Any], max_tables: int = 8, max_numeric_cols: int = 5) -> Dict[str, Any]:
    """Compute basic statistics for numeric columns"""
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
                        "mean": convert_numpy_types(float(row["mean"])) if row["mean"] is not None else None,
                        "min": convert_numpy_types(float(row["min"])) if row["min"] is not None else None,
                        "max": convert_numpy_types(float(row["max"])) if row["max"] is not None else None,
                    }
                except Exception:
                    stats_for_table[col] = {"mean": None, "min": None, "max": None}
            if stats_for_table:
                profiling[table] = stats_for_table
    return profiling


def _infer_fk_hints(schema: Dict[str, Any]) -> List[Dict[str, str]]:
    """Infer foreign key relationships by column naming patterns"""
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


def _infer_roles(schema: Dict[str, Any]) -> Dict[str, str]:
    """Classify columns as measure/dimension/id"""
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


def _build_report(schema: Dict[str, Any], profiling: Dict[str, Any]) -> str:
    """Generate human-readable schema summary"""
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


# ============================================================================
# LangGraph Nodes
# ============================================================================

def validate_connection(state: DatabaseState) -> Dict[str, Any]:
    """Node: Validate database connection"""
    database_url = state.get("database_url")
    if not database_url:
        return {
            "connection_status": "ERROR: No database URL provided",
            "error_log": ["Database URL is required"]
        }
    
    success, message = _check_database_connection(database_url)
    return {
        "connection_status": message,
        "error_log": [] if success else [message]
    }


def discover_schema(state: DatabaseState) -> Dict[str, Any]:
    """Node: Discover database schema"""
    database_url = state.get("database_url")
    if not database_url or "failed" in state.get("connection_status", "").lower():
        return {
            "schema": {},
            "error_log": ["Cannot discover schema: no valid database connection"]
        }
    
    try:
        schema = _describe_schema(database_url)
        return {"schema": schema}
    except Exception as e:
        return {
            "schema": {},
            "error_log": [f"Schema discovery failed: {str(e)}"]
        }


def infer_relationships(state: DatabaseState) -> Dict[str, Any]:
    """Node: Infer foreign key relationships"""
    schema = state.get("schema", {})
    if not schema:
        return {
            "fk_hints": [],
            "error_log": ["Cannot infer relationships: no schema available"]
        }
    
    try:
        fk_hints = _infer_fk_hints(schema)
        return {"fk_hints": fk_hints}
    except Exception as e:
        return {
            "fk_hints": [],
            "error_log": [f"Relationship inference failed: {str(e)}"]
        }


def classify_columns(state: DatabaseState) -> Dict[str, Any]:
    """Node: Classify columns as measure/dimension/id"""
    schema = state.get("schema", {})
    if not schema:
        return {
            "roles": {},
            "error_log": ["Cannot classify columns: no schema available"]
        }
    
    try:
        roles = _infer_roles(schema)
        return {"roles": roles}
    except Exception as e:
        return {
            "roles": {},
            "error_log": [f"Column classification failed: {str(e)}"]
        }


def compute_statistics(state: DatabaseState) -> Dict[str, Any]:
    """Node: Compute basic statistics"""
    database_url = state.get("database_url")
    schema = state.get("schema", {})
    
    if not database_url or not schema:
        return {
            "profiling": {},
            "report": "No statistics available",
            "error_log": ["Cannot compute statistics: missing database connection or schema"]
        }
    
    try:
        profiling = _compute_basic_stats(database_url, schema)
        report = _build_report(schema, profiling)
        return {
            "profiling": convert_numpy_types(profiling),
            "report": report
        }
    except Exception as e:
        return {
            "profiling": {},
            "report": f"Statistics computation failed: {str(e)}",
            "error_log": [f"Statistics computation failed: {str(e)}"]
        }


# ============================================================================
# Sub-graph Creation
# ============================================================================

def create_database_agent() -> StateGraph:
    """Create the database agent sub-graph"""
    builder = StateGraph(DatabaseState)
    
    # Add nodes
    builder.add_node("validate_connection", validate_connection)
    builder.add_node("discover_schema", discover_schema)
    builder.add_node("infer_relationships", infer_relationships)
    builder.add_node("classify_columns", classify_columns)
    builder.add_node("compute_statistics", compute_statistics)
    
    # Add edges (sequential flow)
    builder.add_edge(START, "validate_connection")
    builder.add_edge("validate_connection", "discover_schema")
    builder.add_edge("discover_schema", "infer_relationships")
    builder.add_edge("infer_relationships", "classify_columns")
    builder.add_edge("classify_columns", "compute_statistics")
    builder.add_edge("compute_statistics", END)
    
    # Compile with memory for persistence
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ============================================================================
# System Integration Functions
# ============================================================================

def run_database_agent(system_state: DVDRentalSystemState) -> Dict[str, Any]:
    """Run database agent and return updates for system state"""
    # Extract database-specific state
    db_state = {
        "database_url": system_state["database_url"],
        "connection_status": "",
        "schema": {},
        "fk_hints": [],
        "roles": {},
        "profiling": {},
        "report": "",
        "error_log": []
    }
    
    # Create and run the agent
    agent = create_database_agent()
    
    # Execute the database agent workflow
    thread = {"configurable": {"thread_id": f"db_agent_{int(time.time())}"}}
    result = agent.invoke(db_state, thread)
    
    # Return updates for the main system state
    return {
        "connection_status": result.get("connection_status", ""),
        "schema": result.get("schema", {}),
        "fk_hints": result.get("fk_hints", []),
        "roles": result.get("roles", {}),
        "profiling": result.get("profiling", {}),
        "report": result.get("report", ""),
        "errors": result.get("error_log", []),
        "current_stage": "database_complete"
    }
