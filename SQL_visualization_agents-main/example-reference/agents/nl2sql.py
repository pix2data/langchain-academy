"""
Prompt-to-SQL Agent (simplified)
- Builds a tiny semantic plan from roles and request
- Generates up to two SQL candidates using fk_hints and a constrained skeleton
- Executes the selected SQL elsewhere (Verify+Plot)
"""
from __future__ import annotations

import os
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


SAFE_SQL_MAX_LIMIT = 1000


def _create_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True, future=True)


def _dialect_from_url(database_url: str) -> str:
    if database_url.startswith("sqlite"):
        return "sqlite"
    return "postgres"


def _extract_sql_from_text(response_text: str) -> str:
    t = response_text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        if t.endswith("```"):
            t = t[:-3]
    t = re.sub(r"^(sql\s*:|SQL\s*:)", "", t).strip()
    t = re.sub(r"^(here is the sql\s*:|Here is the SQL\s*:)", "", t).strip()
    m = re.search(r"(?is)\b(select|with)\b", t)
    if m:
        t = t[m.start():]
    if ";" in t:
        t = t.split(";", 1)[0]
    return t.strip()


def _ensure_select_only(sql_query: str) -> str:
    sql = sql_query.strip().strip(";")
    lowered = sql.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT queries are allowed")
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "truncate"]
    if any(f" {w} " in f" {lowered} " for w in forbidden):
        raise ValueError("Disallowed SQL command detected")
    # Check for LIMIT with word boundaries to avoid false positives
    import re
    if not re.search(r'\blimit\b', lowered):
        return sql + " LIMIT " + str(SAFE_SQL_MAX_LIMIT)
    return sql


def _summarize_schema(schema: Dict[str, Any]) -> str:
    lines: List[str] = []
    for table, meta in schema.items():
        cols = ", ".join(c["name"] for c in meta.get("columns", []))
        lines.append(f"- {table}({cols})")
    return "\n".join(lines)


def _summarize_fk_hints(fk_hints: List[Dict[str, str]]) -> str:
    out = []
    for h in fk_hints[:40]:
        # Format as SQL JOIN to make it clearer
        out.append(f"JOIN {h['to_table']} ON {h['from_table']}.{h['from_col']} = {h['to_table']}.{h['to_col']}")
    return "\n".join(out)


def _summarize_roles(roles: Dict[str, str]) -> str:
    # keep brief
    items = list(roles.items())[:80]
    return "\n".join(f"{k}: {v}" for k, v in items)


def _build_plan(user_request: str, roles: Dict[str, str]) -> Dict[str, Any]:
    lower = user_request.lower()
    # crude measure guess: prefer columns tagged as measure and containing synonyms
    measure_synonyms = ["amount", "total", "sum", "revenue", "sales", "payment", "count"]
    candidate_measures = [k for k, v in roles.items() if v == "measure"]
    # pick a plausible one
    chosen_measure = None
    for m in candidate_measures:
        if any(s in m.lower() for s in measure_synonyms):
            chosen_measure = m
            break
    if not chosen_measure and candidate_measures:
        chosen_measure = candidate_measures[0]
    # dims: pick a couple of dimensions
    dims = [k for k, v in roles.items() if v == "dimension"]
    dims = dims[:2]
    return {"measure": chosen_measure, "dimensions": dims}


def _generate_candidate_sqls(user_request: str, schema: Dict[str, Any], fk_hints: List[Dict[str, str]], roles: Dict[str, str], database_url: str) -> List[str]:
    dialect = _dialect_from_url(database_url)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    system = (
        "You write one safe SQL SELECT per request."
        " CRITICAL: Use ONLY the provided FK hints for ALL JOIN conditions."
        " NEVER create joins that are not explicitly listed in the FK hints."
        " Example: If you need to join film to rental, you MUST go through inventory:"
        " film -> inventory (via film.film_id = inventory.film_id)"
        " inventory -> rental (via inventory.inventory_id = rental.inventory_id)"
        " NEVER join film.film_id directly to rental.inventory_id - this is WRONG."
        " Output exactly one SQL statement; no comments, no code fences."
    )
    schema_txt = _summarize_schema(schema)
    fk_txt = _summarize_fk_hints(fk_hints)
    roles_txt = _summarize_roles(roles)

    base_prompt = (
        f"Dialect: {dialect}\n"
        f"Schema:\n{schema_txt}\n\n"
        f"Valid JOIN patterns (use EXACTLY these, no others):\n{fk_txt}\n\n"
        f"Column roles:\n{roles_txt}\n\n"
        f"Task: {user_request}\n"
        "Constraints: SELECT-only; LIMIT 1000 max; GROUP BY when aggregating."
    )
    cands: List[str] = []
    for variant in ["focused", "alt_join"]:
        user = base_prompt + ("\nVariant: prefer minimal joins" if variant == "focused" else "\nVariant: alternate join if plausible")
        msg = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        sql = _ensure_select_only(_extract_sql_from_text(msg.content))
        cands.append(sql)
    # Deduplicate
    uniq: List[str] = []
    seen = set()
    for s in cands:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq[:2] if uniq else cands[:1]


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    user_request = state.get("user_request")
    schema = state.get("schema") or {}
    fk_hints = state.get("fk_hints") or []
    roles = state.get("roles") or {}
    database_url = state.get("database_url") or os.environ.get("DATABASE_URL")

    if not user_request:
        return {"error": "user_request is required"}
    if not database_url:
        return {"error": "DATABASE_URL not set"}
    if not schema:
        return {"error": "schema is required"}

    plan = _build_plan(user_request, roles)
    sql_candidates = _generate_candidate_sqls(user_request, schema, fk_hints, roles, database_url)

    return {"semantic_plan": plan, "sql_candidates": sql_candidates}


if __name__ == "__main__":
    print("This module now returns up to two SQL candidates; execution happens in Verify+Plot stage.") 