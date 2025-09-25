#!/usr/bin/env python3
"""
Simple Human Approval Interface for DVD Rental SQL System
Demonstrates how to handle human feedback in the current architecture
"""

import sys
from pathlib import Path

# Add the phase1_implementation directory to path  
sys.path.append(str(Path(__file__).parent / "phase1_implementation"))

from state_schemas import create_initial_state
from agents.database_agent import run_database_agent
from agents.nl2sql_agent import run_nl2sql_agent, resume_nl2sql_agent
from agents.data_analysis_agent import run_data_analysis_agent
from agents.visualization_agent import run_visualization_agent


def manual_workflow_with_approval(user_request: str):
    """
    Demonstrate manual workflow with human approval points
    """
    print(f"=== Processing: {user_request} ===\n")
    
    # Step 1: Initialize and discover database
    print("1. Database Discovery...")
    state = create_initial_state(user_request, "sqlite:///example-reference/dvdrental.sqlite")
    db_result = run_database_agent(state)
    state.update(db_result)
    
    if db_result.get("current_stage") == "database_error":
        print(f"❌ Database error: {db_result.get('errors', [])}")
        return
    
    print("✅ Database connected and schema discovered")
    
    # Step 2: Generate SQL with human approval
    print("\n2. SQL Generation...")
    sql_result = run_nl2sql_agent(state)
    state.update(sql_result)
    
    if sql_result.get("current_stage") == "awaiting_human_approval":
        print("🔍 Human approval required!")
        print(f"Confidence: {sql_result.get('confidence_score', 0):.4f}")
        print(f"Generated SQL:")
        print(f"  {sql_result.get('selected_sql', 'No SQL')}")
        
        # Human decision point
        approval = input("\nApprove this SQL? (y/n/edit): ").strip().lower()
        
        if approval == 'n':
            print("❌ SQL rejected by human. Stopping workflow.")
            return
        elif approval == 'edit':
            feedback = input("Provide feedback for SQL improvement: ")
            # Resume with feedback
            thread_id = sql_result.get("thread_id")
            if thread_id:
                print("🔄 Resuming with feedback...")
                resumed_result = resume_nl2sql_agent(thread_id, feedback)
                state.update(resumed_result)
                print(f"Updated SQL: {resumed_result.get('selected_sql', 'No SQL')}")
            else:
                print("❌ Cannot resume - no thread ID")
                return
        else:
            print("✅ SQL approved by human")
    
    elif sql_result.get("current_stage") == "nl2sql_complete":
        print("✅ SQL generated successfully (high confidence)")
    else:
        print(f"❌ SQL generation failed: {sql_result.get('errors', [])}")
        return
    
    # Step 3: Execute data analysis
    print("\n3. Data Execution...")
    data_result = run_data_analysis_agent(state)
    state.update(data_result)
    
    if data_result.get("current_stage") == "data_error":
        print(f"❌ Data execution error: {data_result.get('errors', [])}")
        return
    
    print("✅ Data executed and analyzed")
    
    # Step 4: Create visualization
    print("\n4. Visualization Creation...")
    viz_result = run_visualization_agent(state)
    state.update(viz_result)
    
    if viz_result.get("current_stage") == "viz_error":
        print(f"❌ Visualization error: {viz_result.get('errors', [])}")
        return
    
    print("✅ Visualization created successfully!")
    
    # Show final results
    print(f"\n📊 Final Results:")
    print(f"SQL: {state.get('selected_sql', 'N/A')}")
    print(f"Data shape: {state.get('data_characteristics', {}).get('shape', 'N/A')}")
    print(f"Visualization: {state.get('plot_specification', {}).get('chart_type', 'N/A')} chart")
    
    return state


def demo_manual_approval():
    """Demo the manual approval process"""
    # Test with film query that should trigger human approval
    result = manual_workflow_with_approval("Show me films with their release years")
    
    if result:
        print("\n🎉 Workflow completed successfully!")
    else:
        print("\n❌ Workflow was stopped or failed")


if __name__ == "__main__":
    demo_manual_approval() 