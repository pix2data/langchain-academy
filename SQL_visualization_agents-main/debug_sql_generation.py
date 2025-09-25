#!/usr/bin/env python3
"""
Debug SQL Generation - Find out why no SQL is being generated
"""

import sys
import os
from pathlib import Path

# Add the phase1_implementation directory to path
sys.path.append(str(Path(__file__).parent / "phase1_implementation"))

from state_schemas import create_initial_state
from agents.database_agent import run_database_agent
from agents.nl2sql_agent import _generate_sql_candidates, _summarize_schema, _summarize_fk_hints, _summarize_roles


def debug_sql_generation():
    """Debug why SQL generation is failing"""
    print("=== Debugging SQL Generation ===\n")
    
    # Step 1: Set up state with database discovery
    print("1. Setting up database connection...")
    state = create_initial_state(
        "plot scatter for movie length vs renting date",
        "sqlite:///example-reference/dvdrental.sqlite"
    )
    
    # Run database discovery
    db_result = run_database_agent(state)
    state.update(db_result)
    
    if db_result.get("current_stage") == "database_error":
        print(f"❌ Database error: {db_result.get('errors', [])}")
        return
    
    print("✅ Database connected successfully")
    print(f"Schema tables: {list(state.get('schema', {}).keys())}")
    print(f"FK hints count: {len(state.get('fk_hints', []))}")
    print(f"Roles count: {len(state.get('roles', {}))}")
    
    # Step 2: Check OpenAI API key
    print("\n2. Checking OpenAI API configuration...")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("This is likely why no SQL is being generated.")
        print("Solution: Set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return
    elif len(openai_key) < 10:
        print(f"❌ OPENAI_API_KEY seems invalid (too short): {openai_key[:10]}...")
        return
    else:
        print(f"✅ OpenAI API key found: {openai_key[:10]}...{openai_key[-4:]}")
    
    # Step 3: Test SQL generation with detailed output
    print("\n3. Testing SQL generation...")
    
    user_request = state["user_request"]
    schema = state.get("schema", {})
    fk_hints = state.get("fk_hints", [])
    roles = state.get("roles", {})
    database_url = state["database_url"]
    
    print(f"User request: {user_request}")
    print(f"Database URL: {database_url}")
    
    # Show what will be sent to LLM
    print("\n4. LLM Input Summary:")
    schema_txt = _summarize_schema(schema)
    fk_txt = _summarize_fk_hints(fk_hints)
    roles_txt = _summarize_roles(roles)
    
    print(f"Schema summary (first 500 chars):")
    print(f"  {schema_txt[:500]}...")
    print(f"FK hints (first 3):")
    for i, hint in enumerate(fk_hints[:3]):
        print(f"  {i+1}. {hint['from_table']}.{hint['from_col']} -> {hint['to_table']}.{hint['to_col']}")
    print(f"Roles (first 5):")
    for i, (col, role) in enumerate(list(roles.items())[:5]):
        print(f"  {i+1}. {col}: {role}")
    
    # Step 5: Try generating SQL
    print("\n5. Attempting SQL generation...")
    try:
        candidates = _generate_sql_candidates(user_request, schema, fk_hints, roles, database_url)
        
        if candidates:
            print(f"✅ Generated {len(candidates)} SQL candidates:")
            for i, sql in enumerate(candidates):
                print(f"  {i+1}. {sql}")
        else:
            print("❌ No SQL candidates generated!")
            print("This could be due to:")
            print("  - OpenAI API rate limiting")
            print("  - Model not understanding the request")
            print("  - Schema/FK hints not providing enough information")
            
    except Exception as e:
        print(f"❌ SQL generation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_sql_generation() 