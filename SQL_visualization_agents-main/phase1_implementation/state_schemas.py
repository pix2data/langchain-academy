"""
State Schemas for DVD Rental SQL Visualization System
Defines TypedDict schemas for LangGraph state management
"""

from typing import Dict, List, Any, Optional, Annotated, Union
from typing_extensions import TypedDict
import operator
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from langgraph.graph import MessagesState
import numpy as np

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Custom Types for Serializable State
# ============================================================================

# Use dict representation for DataFrames to make them serializable
DataFrameDict = Dict[str, Any]  # Will store DataFrame as dict with 'data' and 'columns' keys
PlotlyFigureDict = Dict[str, Any]  # Will store Plotly figure as dict


# ============================================================================
# Individual Agent States (Sub-graph States)
# ============================================================================

class DatabaseState(TypedDict):
    """State for Database Connection & Schema Agent"""
    database_url: str  # SQLite connection string
    connection_status: str  # Connection validation result
    schema: Dict[str, Any]  # Table schemas and metadata
    fk_hints: List[Dict[str, str]]  # Foreign key relationships
    roles: Dict[str, str]  # Column roles (measure/dimension/id)
    profiling: Dict[str, Any]  # Basic statistics per table
    report: str  # Human-readable schema summary
    error_log: List[str]  # Error tracking


class NL2SQLState(MessagesState):
    """State for Natural Language to SQL Agent"""
    user_request: str  # Natural language question
    schema: Dict[str, Any]  # From database agent
    fk_hints: List[Dict[str, str]]  # Foreign key relationships
    roles: Dict[str, str]  # Column roles (measure/dimension/id)
    semantic_plan: Dict[str, Any]  # High-level query plan
    sql_candidates: List[str]  # Generated SQL options
    selected_sql: str  # Final chosen SQL query
    confidence_score: float  # Query generation confidence
    validation_results: Dict[str, Any]  # SQL safety validation
    human_feedback: Optional[str]  # Human query refinement


class DataAnalysisState(TypedDict):
    """State for Data Execution & Analysis Agent"""
    sql_candidates: List[str]  # From NL2SQL agent
    selected_sql: str  # Best SQL to execute
    query_results: DataFrameDict  # Executed query data (serializable dict)
    data_characteristics: Dict[str, Any]  # Data shape, types, ranges
    basic_statistics: Dict[str, Any]  # Mean, count, unique values
    data_quality_score: float  # Quality assessment of results
    analysis_summary: str  # Human-readable data summary
    execution_errors: List[str]  # Query execution issues


class VisualizationState(MessagesState):
    """State for Intelligent Visualization Agent"""
    user_request: str  # Original user question
    query_results: DataFrameDict  # Data from execution agent (serializable dict)
    data_characteristics: Dict[str, Any]  # Data analysis from previous agent
    plot_specification: Dict[str, Any]  # LLM-determined plot config
    generated_plot: PlotlyFigureDict  # Plotly figure as dict (serializable)
    plot_html: str  # HTML representation for display
    plot_reasoning: str  # Explanation of chart choice
    alternative_plots: List[Dict]  # Other visualization options
    human_feedback: Optional[str]  # User feedback on visualization


# ============================================================================
# Main System State (Orchestrator State)
# ============================================================================

class DVDRentalSystemState(TypedDict):
    """Main orchestrator state combining all agent outputs"""
    
    # User Input
    user_request: str  # Natural language question about DVD rental data
    
    # Database & Schema (Agent 1)
    database_url: str  # SQLite connection to dvdrental.sqlite
    connection_status: str
    schema: Dict[str, Any]  # Table definitions
    fk_hints: List[Dict[str, str]]  # Foreign key relationships
    roles: Dict[str, str]  # Column classifications
    
    # SQL Generation (Agent 2)
    semantic_plan: Dict[str, Any]  # Query planning
    sql_candidates: List[str]  # Multiple SQL options
    selected_sql: str  # Final chosen query
    
    # Data Execution (Agent 3)
    query_results: DataFrameDict  # Executed data (serializable dict)
    data_characteristics: Dict[str, Any]  # Data analysis
    data_quality_score: float
    
    # Visualization (Agent 4)
    plot_specification: Dict[str, Any]  # Chart configuration
    generated_plot: PlotlyFigureDict  # Final visualization (serializable dict)
    plot_html: str  # Displayable chart
    
    # Control Flow
    current_stage: str
    errors: Annotated[List[str], operator.add]  # Accumulate errors across agents
    human_feedback: Optional[str]  # Human-in-the-loop input
    
    # Progress tracking
    stages_completed: List[str]
    total_processing_time: float


# ============================================================================
# DataFrame Serialization Utilities
# ============================================================================

def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types for serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        # Check for pandas NA values (but only for scalar values, not arrays)
        try:
            if pd.isna(obj) and not isinstance(obj, (list, tuple, dict, np.ndarray)):
                return None
        except (ValueError, TypeError):
            # pd.isna() might fail on some types, just continue
            pass
        return obj

def dataframe_to_dict(df: pd.DataFrame) -> DataFrameDict:
    """Convert pandas DataFrame to serializable dict"""
    if df.empty:
        return {"data": [], "columns": [], "index": []}
    
    # Convert DataFrame to records and ensure all NumPy types are converted to native Python types
    data_records = df.to_dict('records')
    serializable_data = []
    
    for record in data_records:
        serializable_record = {}
        for key, value in record.items():
            # Convert NumPy types to native Python types
            if isinstance(value, np.integer):
                serializable_record[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_record[key] = float(value)
            elif isinstance(value, np.bool_):
                serializable_record[key] = bool(value)
            elif isinstance(value, np.ndarray):
                serializable_record[key] = value.tolist()
            elif pd.isna(value):
                serializable_record[key] = None
            else:
                serializable_record[key] = value
        serializable_data.append(serializable_record)
    
    # Also convert index values if they contain NumPy types
    index_list = []
    for idx in df.index:
        if isinstance(idx, np.integer):
            index_list.append(int(idx))
        elif isinstance(idx, np.floating):
            index_list.append(float(idx))
        elif isinstance(idx, np.bool_):
            index_list.append(bool(idx))
        else:
            index_list.append(idx)
    
    return {
        "data": serializable_data,
        "columns": df.columns.tolist(),
        "index": index_list,
        "dtypes": df.dtypes.astype(str).to_dict()
    }


def dict_to_dataframe(df_dict: DataFrameDict) -> pd.DataFrame:
    """Convert dict back to pandas DataFrame"""
    if not df_dict or not df_dict.get("data"):
        return pd.DataFrame()
    
    df = pd.DataFrame(df_dict["data"])
    if "columns" in df_dict:
        df.columns = df_dict["columns"]
    if "index" in df_dict:
        df.index = df_dict["index"]
    
    return df


def figure_to_dict(fig: go.Figure) -> PlotlyFigureDict:
    """Convert Plotly figure to serializable dict"""
    if fig is None:
        return {}
    return fig.to_dict()


def dict_to_figure(fig_dict: PlotlyFigureDict) -> go.Figure:
    """Convert dict back to Plotly figure"""
    if not fig_dict:
        return go.Figure()
    return go.Figure(fig_dict)


# ============================================================================
# State Utilities
# ============================================================================

def create_initial_state(user_request: str, database_url: str = None) -> DVDRentalSystemState:
    """Create initial state for the system"""
    if database_url is None:
        # Default to local SQLite database - try multiple possible paths
        from pathlib import Path
        possible_paths = [
            Path(__file__).parent.parent / "example-reference" / "dvdrental.sqlite",
            Path("example-reference") / "dvdrental.sqlite",
            Path("SQL_visualization_agents/example-reference/dvdrental.sqlite"),
            Path("../example-reference/dvdrental.sqlite")
        ]
        
        for path in possible_paths:
            if path.exists():
                database_url = f"sqlite:///{path.resolve()}"
                break
        else:
            # Fallback to a relative path
            database_url = "sqlite:///example-reference/dvdrental.sqlite"
    
    return DVDRentalSystemState(
        user_request=user_request,
        database_url=database_url,
        connection_status="",
        schema={},
        fk_hints=[],
        roles={},
        semantic_plan={},
        sql_candidates=[],
        selected_sql="",
        query_results={},
        data_characteristics={},
        data_quality_score=0.0,
        plot_specification={},
        generated_plot={},
        plot_html="",
        current_stage="initialization",
        errors=[],
        human_feedback=None,
        stages_completed=[],
        total_processing_time=0.0
    )


def extract_database_state(system_state: DVDRentalSystemState) -> DatabaseState:
    """Extract database agent state from system state"""
    return DatabaseState(
        database_url=system_state["database_url"],
        connection_status=system_state.get("connection_status", ""),
        schema=system_state.get("schema", {}),
        fk_hints=system_state.get("fk_hints", []),
        roles=system_state.get("roles", {}),
        profiling=system_state.get("profiling", {}),
        report=system_state.get("report", ""),
        error_log=[]
    )


def extract_nl2sql_state(system_state: DVDRentalSystemState) -> NL2SQLState:
    """Extract NL2SQL agent state from system state"""
    return NL2SQLState(
        messages=[],  # MessagesState requirement
        user_request=system_state["user_request"],
        schema=system_state.get("schema", {}),
        fk_hints=system_state.get("fk_hints", []),
        roles=system_state.get("roles", {}),
        semantic_plan=system_state.get("semantic_plan", {}),
        sql_candidates=system_state.get("sql_candidates", []),
        selected_sql=system_state.get("selected_sql", ""),
        confidence_score=0.0,
        validation_results={},
        human_feedback=system_state.get("human_feedback")
    )


def extract_data_analysis_state(system_state: DVDRentalSystemState) -> DataAnalysisState:
    """Extract data analysis agent state from system state"""
    return DataAnalysisState(
        sql_candidates=system_state.get("sql_candidates", []),
        selected_sql=system_state.get("selected_sql", ""),
        query_results=system_state.get("query_results", {}),
        data_characteristics=system_state.get("data_characteristics", {}),
        basic_statistics={},
        data_quality_score=system_state.get("data_quality_score", 0.0),
        analysis_summary="",
        execution_errors=[]
    )


def extract_visualization_state(system_state: DVDRentalSystemState) -> VisualizationState:
    """Extract visualization agent state from system state"""
    return VisualizationState(
        messages=[],  # MessagesState requirement
        user_request=system_state["user_request"],
        query_results=system_state.get("query_results", {}),
        data_characteristics=system_state.get("data_characteristics", {}),
        plot_specification=system_state.get("plot_specification", {}),
        generated_plot=system_state.get("generated_plot", {}),
        plot_html=system_state.get("plot_html", ""),
        plot_reasoning="",
        alternative_plots=[],
        human_feedback=system_state.get("human_feedback")
    )
