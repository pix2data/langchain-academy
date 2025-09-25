"""
Natural Language to SQL Agent (LangGraph Implementation)
Based on existing nl2sql.py functionality with LangGraph orchestration
"""

import os
import re
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables from .env file FIRST
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from ..state_schemas import NL2SQLState, DVDRentalSystemState
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from state_schemas import NL2SQLState, DVDRentalSystemState

SAFE_SQL_MAX_LIMIT = 10000


# ============================================================================
# Core SQL Generation Functions (from existing code)
# ============================================================================

def _create_engine(database_url: str) -> Engine:
    """Create SQLAlchemy engine"""
    return create_engine(database_url, pool_pre_ping=True, future=True)


def _dialect_from_url(database_url: str) -> str:
    """Determine SQL dialect from database URL"""
    if database_url.startswith("sqlite"):
        return "sqlite"
    return "postgres"


def _extract_sql_from_text(response_text: str) -> str:
    """Extract SQL query from LLM response text"""
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
    """Validate SQL safety and add LIMIT if needed"""
    sql = sql_query.strip().strip(";")
    lowered = sql.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT queries are allowed")
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "truncate"]
    if any(f" {w} " in f" {lowered} " for w in forbidden):
        raise ValueError("Disallowed SQL command detected")
    # Check for LIMIT with word boundaries to avoid false positives
    if not re.search(r'\blimit\b', lowered):
        return sql + " LIMIT " + str(SAFE_SQL_MAX_LIMIT)
    return sql


def _summarize_schema(schema: Dict[str, Any]) -> str:
    """Create concise schema summary for LLM"""
    lines: List[str] = []
    for table, meta in schema.items():
        cols = ", ".join(c["name"] for c in meta.get("columns", []))
        lines.append(f"- {table}({cols})")
    return "\n".join(lines)


def _summarize_fk_hints(fk_hints: List[Dict[str, str]]) -> str:
    """Create FK hints summary for LLM"""
    out = []
    for h in fk_hints[:40]:
        # Format as SQL JOIN to make it clearer
        out.append(f"JOIN {h['to_table']} ON {h['from_table']}.{h['from_col']} = {h['to_table']}.{h['to_col']}")
    return "\n".join(out)


def _summarize_roles(roles: Dict[str, str]) -> str:
    """Create column roles summary for LLM"""
    # keep brief
    items = list(roles.items())[:80]
    return "\n".join(f"{k}: {v}" for k, v in items)


def _build_semantic_plan(user_request: str, roles: Dict[str, str]) -> Dict[str, Any]:
    """Build semantic plan identifying measures and dimensions"""
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


def _generate_sql_candidates(user_request: str, schema: Dict[str, Any], 
                           fk_hints: List[Dict[str, str]], roles: Dict[str, str], 
                           database_url: str) -> List[str]:
    """Generate multiple SQL candidate queries"""
    dialect = _dialect_from_url(database_url)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
        "Constraints: SELECT-only; LIMIT 10000 max; GROUP BY when aggregating."
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


# ============================================================================
# LangGraph Nodes
# ============================================================================

def parse_user_intent(state: NL2SQLState) -> Dict[str, Any]:
    """Node: Parse user intent and create semantic plan"""
    user_request = state.get("user_request", "")
    roles = state.get("roles", {})
    
    if not user_request:
        return {
            "semantic_plan": {},
            "messages": [AIMessage(content="No user request provided")]
        }
    
    try:
        semantic_plan = _build_semantic_plan(user_request, roles)
        return {
            "semantic_plan": semantic_plan,
            "messages": [AIMessage(content=f"Identified semantic plan: {semantic_plan}")]
        }
    except Exception as e:
        return {
            "semantic_plan": {},
            "messages": [AIMessage(content=f"Failed to parse user intent: {str(e)}")]
        }


def generate_sql_candidates(state: NL2SQLState) -> Dict[str, Any]:
    """Node: Generate multiple SQL candidate queries"""
    user_request = state.get("user_request", "")
    schema = state.get("schema", {})
    fk_hints = state.get("fk_hints", [])
    roles = state.get("roles", {})
    
    # Get database URL from environment or state
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        # Try to find the database file
        from pathlib import Path
        possible_paths = [
            Path(__file__).parent.parent.parent / "example-reference" / "dvdrental.sqlite",
            Path("example-reference") / "dvdrental.sqlite",
            Path("../example-reference/dvdrental.sqlite")
        ]
        
        for path in possible_paths:
            if path.exists():
                database_url = f"sqlite:///{path.resolve()}"
                break
        else:
            database_url = "sqlite:///example-reference/dvdrental.sqlite"
    
    if not user_request or not schema:
        return {
            "sql_candidates": [],
            "messages": [AIMessage(content="Cannot generate SQL: missing user request or schema")]
        }
    
    try:
        candidates = _generate_sql_candidates(user_request, schema, fk_hints, roles, database_url)
        return {
            "sql_candidates": candidates,
            "messages": [AIMessage(content=f"Generated {len(candidates)} SQL candidates:\n" + 
                                 "\n".join(f"{i+1}. {sql}" for i, sql in enumerate(candidates)))]
        }
    except Exception as e:
        return {
            "sql_candidates": [],
            "messages": [AIMessage(content=f"SQL generation failed: {str(e)}")]
        }


def validate_sql_safety(state: NL2SQLState) -> Dict[str, Any]:
    """Node: Validate SQL candidates for safety"""
    candidates = state.get("sql_candidates", [])
    
    if not candidates:
        return {
            "validation_results": {"status": "no_candidates"},
            "messages": [AIMessage(content="No SQL candidates to validate")]
        }
    
    validation_results = {
        "status": "validated",
        "safe_candidates": [],
        "validation_errors": []
    }
    
    for i, sql in enumerate(candidates):
        try:
            # Validate SQL safety
            safe_sql = _ensure_select_only(sql)
            validation_results["safe_candidates"].append(safe_sql)
        except Exception as e:
            validation_results["validation_errors"].append(f"Candidate {i+1}: {str(e)}")
    
    return {
        "validation_results": validation_results,
        "sql_candidates": validation_results["safe_candidates"],  # Update with safe versions
        "messages": [AIMessage(content=f"Validated {len(validation_results['safe_candidates'])} safe SQL candidates")]
    }


def rank_candidates(state: NL2SQLState) -> Dict[str, Any]:
    """Node: Rank and select best SQL candidate"""
    candidates = state.get("sql_candidates", [])
    
    if not candidates:
        return {
            "selected_sql": "",
            "confidence_score": 0.0,
            "messages": [AIMessage(content="No valid SQL candidates to rank")]
        }
    
    # Simple ranking: prefer shorter queries (less complex joins)
    # In a more sophisticated implementation, we could execute and score results
    ranked_candidates = sorted(candidates, key=len)
    selected_sql = ranked_candidates[0]
    
    # Confidence based on number of candidates and query complexity
    confidence = min(0.9, 0.5 + (len(candidates) * 0.2))
    if len(selected_sql) > 500:  # Very long query
        confidence *= 0.8
    
    return {
        "selected_sql": selected_sql,
        "confidence_score": confidence,
        "messages": [AIMessage(content=f"Selected SQL (confidence: {confidence:.2f}):\n{selected_sql}")]
    }


def human_approval(state: NL2SQLState) -> Dict[str, Any]:
    """Node: Human-in-the-loop approval point"""
    # This is a no-op node that should be interrupted on
    # Allows human review of the generated SQL
    confidence = state.get("confidence_score", 0.0)
    selected_sql = state.get("selected_sql", "")
    
    return {
        "messages": [AIMessage(content=f"SQL query ready for review (confidence: {confidence:.2f}):\n{selected_sql}\n\nWaiting for human approval...")]
    }


def refine_sql(state: NL2SQLState) -> Dict[str, Any]:
    """Node: Refine SQL based on human feedback"""
    feedback = state.get("human_feedback", "")
    current_sql = state.get("selected_sql", "")
    
    if not feedback:
        return {
            "messages": [AIMessage(content="No feedback provided, keeping current SQL")]
        }
    
    # In a more sophisticated implementation, we could use LLM to refine the SQL
    # For now, we'll just log the feedback
    return {
        "messages": [AIMessage(content=f"Received feedback: {feedback}\nCurrent SQL maintained: {current_sql}")]
    }


# ============================================================================
# Routing Logic
# ============================================================================

def should_get_human_approval(state: NL2SQLState) -> str:
    """Determine if human approval is needed"""
    confidence = state.get("confidence_score", 0.0)
    
    # Require human approval for low confidence queries
    if confidence < 0.9999:
        return "human_approval"
    
    return END


def should_refine_sql(state: NL2SQLState) -> str:
    """Determine if SQL needs refinement based on feedback"""
    feedback = state.get("human_feedback")
    
    if feedback and feedback.strip():
        return "refine_sql"
    
    return END


# ============================================================================
# Sub-graph Creation
# ============================================================================

def create_nl2sql_agent() -> StateGraph:
    """Create the NL2SQL agent sub-graph"""
    builder = StateGraph(NL2SQLState)
    
    # Add nodes
    builder.add_node("parse_user_intent", parse_user_intent)
    builder.add_node("generate_sql_candidates", generate_sql_candidates)
    builder.add_node("validate_sql_safety", validate_sql_safety)
    builder.add_node("rank_candidates", rank_candidates)
    builder.add_node("human_approval", human_approval)
    builder.add_node("refine_sql", refine_sql)
    
    # Add edges
    builder.add_edge(START, "parse_user_intent")
    builder.add_edge("parse_user_intent", "generate_sql_candidates")
    builder.add_edge("generate_sql_candidates", "validate_sql_safety")
    builder.add_edge("validate_sql_safety", "rank_candidates")
    
    # Conditional edges for human approval
    builder.add_conditional_edges(
        "rank_candidates",
        should_get_human_approval,
        ["human_approval", END]
    )
    
    # Handle refinement loop
    builder.add_conditional_edges(
        "human_approval",
        should_refine_sql,
        ["refine_sql", END]
    )
    
    # Refinement back to ranking
    builder.add_edge("refine_sql", "rank_candidates")
    
    # Compile with memory for persistence
    memory = MemorySaver()
    return builder.compile(
        interrupt_before=['human_approval'],  # Breakpoint for human review
        checkpointer=memory
    )


# ============================================================================
# System Integration Functions
# ============================================================================

def run_nl2sql_agent(system_state: DVDRentalSystemState) -> Dict[str, Any]:
    """Run NL2SQL agent and return updates for system state"""
    # Extract NL2SQL-specific state
    nl2sql_state = {
        "messages": [],
        "user_request": system_state["user_request"],
        "schema": system_state.get("schema", {}),
        "fk_hints": system_state.get("fk_hints", []),
        "roles": system_state.get("roles", {}),
        "semantic_plan": {},
        "sql_candidates": [],
        "selected_sql": "",
        "confidence_score": 0.0,
        "validation_results": {},
        "human_feedback": system_state.get("human_feedback")
    }
    
    # Create and run the agent
    agent = create_nl2sql_agent()
    
    # Execute the NL2SQL agent workflow
    thread = {"configurable": {"thread_id": f"nl2sql_agent_{int(time.time())}"}}
    result = agent.invoke(nl2sql_state, thread)
    
    # Return updates for the main system state
    return {
        "semantic_plan": result.get("semantic_plan", {}),
        "sql_candidates": result.get("sql_candidates", []),
        "selected_sql": result.get("selected_sql", ""),
        "confidence_score": result.get("confidence_score", 0.0),
        "current_stage": "nl2sql_complete"
    }
