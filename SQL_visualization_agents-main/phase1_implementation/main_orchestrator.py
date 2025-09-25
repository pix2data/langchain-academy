"""
Main Orchestrator for DVD Rental SQL Visualization System
Coordinates all 4 agents using LangGraph sub-graph patterns
"""

import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()

try:
    # Try relative imports first (when used as a package)
    from .state_schemas import DVDRentalSystemState, create_initial_state, dict_to_dataframe, dict_to_figure
    from .agents.database_agent import run_database_agent
    from .agents.nl2sql_agent import run_nl2sql_agent
    from .agents.data_analysis_agent import run_data_analysis_agent
    from .agents.visualization_agent import run_visualization_agent
except ImportError:
    # Fall back to absolute imports (when running directly)
    from state_schemas import DVDRentalSystemState, create_initial_state, dict_to_dataframe, dict_to_figure
    from agents.database_agent import run_database_agent
    from agents.nl2sql_agent import run_nl2sql_agent
    from agents.data_analysis_agent import run_data_analysis_agent
    from agents.visualization_agent import run_visualization_agent


# ============================================================================
# Main Orchestrator Nodes
# ============================================================================

def initialize_system(state: DVDRentalSystemState) -> Dict[str, Any]:
    """Node: Initialize system and validate inputs"""
    user_request = state.get("user_request", "")
    database_url = state.get("database_url", "")
    
    if not user_request:
        return {
            "current_stage": "error",
            "errors": ["No user request provided"]
        }
    
    if not database_url:
        # Set default database URL
        database_url = "sqlite:///SQL_visualization_agents/example-reference/dvdrental.sqlite"
    
    return {
        "database_url": database_url,
        "current_stage": "initialized",
        "stages_completed": ["initialization"],
        "total_processing_time": 0.0
    }


def run_database_discovery(state: DVDRentalSystemState) -> Dict[str, Any]:
    """Node: Run database connection and schema discovery agent"""
    start_time = time.time()
    
    try:
        updates = run_database_agent(state)
        processing_time = time.time() - start_time
        
        # Check if database connection was successful
        if "failed" in updates.get("connection_status", "").lower():
            return {
                **updates,
                "current_stage": "database_error",
                "total_processing_time": state.get("total_processing_time", 0.0) + processing_time
            }
        
        return {
            **updates,
            "stages_completed": state.get("stages_completed", []) + ["database_discovery"],
            "total_processing_time": state.get("total_processing_time", 0.0) + processing_time
        }
    except Exception as e:
        return {
            "current_stage": "database_error",
            "errors": [f"Database discovery failed: {str(e)}"],
            "total_processing_time": state.get("total_processing_time", 0.0) + (time.time() - start_time)
        }


def run_sql_generation(state: DVDRentalSystemState) -> Dict[str, Any]:
    """Node: Run natural language to SQL agent"""
    start_time = time.time()
    
    try:
        updates = run_nl2sql_agent(state)
        processing_time = time.time() - start_time
        
        # Check if SQL generation was successful
        if not updates.get("selected_sql"):
            return {
                **updates,
                "current_stage": "sql_error",
                "errors": state.get("errors", []) + ["No valid SQL generated"],
                "total_processing_time": state.get("total_processing_time", 0.0) + processing_time
            }
        
        return {
            **updates,
            "stages_completed": state.get("stages_completed", []) + ["sql_generation"],
            "total_processing_time": state.get("total_processing_time", 0.0) + processing_time
        }
    except Exception as e:
        return {
            "current_stage": "sql_error",
            "errors": state.get("errors", []) + [f"SQL generation failed: {str(e)}"],
            "total_processing_time": state.get("total_processing_time", 0.0) + (time.time() - start_time)
        }


def run_data_execution(state: DVDRentalSystemState) -> Dict[str, Any]:
    """Node: Run data execution and analysis agent"""
    start_time = time.time()
    
    try:
        updates = run_data_analysis_agent(state)
        processing_time = time.time() - start_time
        
        # Check if data execution was successful
        query_results = updates.get("query_results")
        if query_results is None or not query_results or (isinstance(query_results, dict) and not query_results.get("data")):
            return {
                **updates,
                "current_stage": "data_error",
                "errors": state.get("errors", []) + ["No data returned from query execution"],
                "total_processing_time": state.get("total_processing_time", 0.0) + processing_time
            }
        
        return {
            **updates,
            "stages_completed": state.get("stages_completed", []) + ["data_execution"],
            "total_processing_time": state.get("total_processing_time", 0.0) + processing_time
        }
    except Exception as e:
        return {
            "current_stage": "data_error",
            "errors": state.get("errors", []) + [f"Data execution failed: {str(e)}"],
            "total_processing_time": state.get("total_processing_time", 0.0) + (time.time() - start_time)
        }


def run_visualization_creation(state: DVDRentalSystemState) -> Dict[str, Any]:
    """Node: Run visualization agent"""
    start_time = time.time()
    
    try:
        updates = run_visualization_agent(state)
        processing_time = time.time() - start_time
        
        return {
            **updates,
            "stages_completed": state.get("stages_completed", []) + ["visualization"],
            "total_processing_time": state.get("total_processing_time", 0.0) + processing_time,
            "current_stage": "completed"
        }
    except Exception as e:
        return {
            "current_stage": "visualization_error",
            "errors": state.get("errors", []) + [f"Visualization creation failed: {str(e)}"],
            "total_processing_time": state.get("total_processing_time", 0.0) + (time.time() - start_time)
        }


def handle_error(state: DVDRentalSystemState) -> Dict[str, Any]:
    """Node: Handle errors and provide user feedback"""
    current_stage = state.get("current_stage", "unknown")
    errors = state.get("errors", [])
    
    error_message = f"System encountered an error at stage: {current_stage}\n"
    error_message += "Errors:\n"
    for i, error in enumerate(errors, 1):
        error_message += f"{i}. {error}\n"
    
    return {
        "current_stage": "error_handled",
        "error_message": error_message
    }


# ============================================================================
# Routing Logic
# ============================================================================

def route_after_database(state: DVDRentalSystemState) -> str:
    """Route after database discovery"""
    current_stage = state.get("current_stage", "")
    
    if "error" in current_stage:
        return "handle_error"
    
    return "run_sql_generation"


def route_after_sql(state: DVDRentalSystemState) -> str:
    """Route after SQL generation"""
    current_stage = state.get("current_stage", "")
    
    if "error" in current_stage:
        return "handle_error"
    
    return "run_data_execution"


def route_after_data(state: DVDRentalSystemState) -> str:
    """Route after data execution"""
    current_stage = state.get("current_stage", "")
    
    if "error" in current_stage:
        return "handle_error"
    
    return "run_visualization_creation"


def route_after_visualization(state: DVDRentalSystemState) -> str:
    """Route after visualization"""
    current_stage = state.get("current_stage", "")
    
    if "error" in current_stage:
        return "handle_error"
    
    return END


# ============================================================================
# Main Graph Creation
# ============================================================================

def create_main_orchestrator() -> StateGraph:
    """Create the main orchestrator graph"""
    builder = StateGraph(DVDRentalSystemState)
    
    # Add nodes
    builder.add_node("initialize_system", initialize_system)
    builder.add_node("run_database_discovery", run_database_discovery)
    builder.add_node("run_sql_generation", run_sql_generation)
    builder.add_node("run_data_execution", run_data_execution)
    builder.add_node("run_visualization_creation", run_visualization_creation)
    builder.add_node("handle_error", handle_error)
    
    # Add edges
    builder.add_edge(START, "initialize_system")
    builder.add_edge("initialize_system", "run_database_discovery")
    
    # Conditional routing between stages
    builder.add_conditional_edges(
        "run_database_discovery",
        route_after_database,
        ["run_sql_generation", "handle_error"]
    )
    
    builder.add_conditional_edges(
        "run_sql_generation",
        route_after_sql,
        ["run_data_execution", "handle_error"]
    )
    
    builder.add_conditional_edges(
        "run_data_execution",
        route_after_data,
        ["run_visualization_creation", "handle_error"]
    )
    
    builder.add_conditional_edges(
        "run_visualization_creation",
        route_after_visualization,
        [END, "handle_error"]
    )
    
    # Error handling
    builder.add_edge("handle_error", END)
    
    # Compile with memory for persistence
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ============================================================================
# High-Level API
# ============================================================================

class DVDRentalVisualizationSystem:
    """High-level interface for the DVD rental visualization system"""
    
    def __init__(self):
        self.graph = create_main_orchestrator()
    
    def process_request(self, user_request: str, database_url: str = None) -> Dict[str, Any]:
        """
        Process a user request for SQL visualization
        
        Args:
            user_request: Natural language question about DVD rental data
            database_url: Optional database URL (defaults to local SQLite)
            
        Returns:
            Dictionary containing visualization results and metadata
        """
        # Create initial state
        initial_state = create_initial_state(user_request, database_url)
        
        # Generate unique thread ID
        thread = {"configurable": {"thread_id": f"main_{int(time.time())}"}}
        
        # Execute the main graph
        try:
            result = self.graph.invoke(initial_state, thread)
            
            # Convert serialized data back for user consumption
            query_results_dict = result.get("query_results", {})
            generated_plot_dict = result.get("generated_plot", {})
            
            # For backwards compatibility, convert back to pandas DataFrame for external API
            query_results_df = dict_to_dataframe(query_results_dict) if query_results_dict else None
            
            return {
                "success": result.get("current_stage") == "completed",
                "user_request": user_request,
                "selected_sql": result.get("selected_sql", ""),
                "plot_html": result.get("plot_html", ""),
                "plot_specification": result.get("plot_specification", {}),
                "data_quality_score": result.get("data_quality_score", 0.0),
                "processing_time": result.get("total_processing_time", 0.0),
                "stages_completed": result.get("stages_completed", []),
                "errors": result.get("errors", []),
                "current_stage": result.get("current_stage", "unknown"),
                "schema_report": result.get("report", ""),
                "alternative_plots": result.get("alternative_plots", []),
                # Internal data for debugging (serialized forms)
                "_query_results_df": query_results_df,
                "_query_results_dict": query_results_dict,
                "_generated_plot_dict": generated_plot_dict
            }
        except Exception as e:
            return {
                "success": False,
                "user_request": user_request,
                "errors": [f"System execution failed: {str(e)}"],
                "current_stage": "system_error"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and capabilities"""
        return {
            "system_name": "DVD Rental SQL Visualization System",
            "version": "1.0.0 (Phase 1)",
            "agents": [
                "Database Connection & Schema Agent",
                "Natural Language to SQL Agent", 
                "Data Execution & Analysis Agent",
                "Intelligent Visualization Agent"
            ],
            "supported_databases": ["SQLite"],
            "supported_visualizations": ["bar", "line", "scatter", "histogram", "box", "table"],
            "features": [
                "Automatic schema discovery",
                "Foreign key relationship inference",
                "Safe SQL generation with limits",
                "Multiple SQL candidate generation",
                "LLM-driven plot type selection",
                "Interactive Plotly visualizations"
            ]
        }


# ============================================================================
# Example Usage and Testing
# ============================================================================

def run_example():
    """Run an example query to test the system"""
    system = DVDRentalVisualizationSystem()
    
    # Example queries for the DVD rental database
    test_queries = [
        "Show me the top 10 grossing film categories",
        "What are the most popular film ratings?",
        "Compare rental counts by store",
        "Show customer rental frequency distribution"
    ]
    
    print("DVD Rental SQL Visualization System - Phase 1 Demo")
    print("=" * 60)
    
    # System status
    status = system.get_system_status()
    print(f"System: {status['system_name']} v{status['version']}")
    print(f"Agents: {len(status['agents'])}")
    print()
    
    # Process example query
    query = test_queries[0]
    print(f"Processing query: {query}")
    print("-" * 40)
    
    result = system.process_request(query)
    
    print(f"Success: {result['success']}")
    print(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
    print(f"Stages completed: {result.get('stages_completed', [])}")
    
    if result['success']:
        print(f"Generated SQL: {result.get('selected_sql', 'N/A')}")
        print(f"Plot type: {result.get('plot_specification', {}).get('plot_type', 'N/A')}")
        print(f"Data quality: {result.get('data_quality_score', 0):.2f}")
    else:
        print(f"Errors: {result.get('errors', [])}")
    
    return result


if __name__ == "__main__":
    # Run example when script is executed directly
    run_example()
