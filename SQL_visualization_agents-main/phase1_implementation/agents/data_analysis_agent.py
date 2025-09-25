"""
Data Execution & Analysis Agent (LangGraph Implementation)
New agent to bridge SQL generation and visualization
"""

import os
import time
from typing import Dict, Any, List
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Import SQL timeout utilities
try:
    from ..sql_timeout_handler import (
        timeout_sql_execution, analyze_query_complexity, suggest_query_optimization,
        create_safe_query_alternative, safe_sql_execution, SQLTimeoutError, logger
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from sql_timeout_handler import (
        timeout_sql_execution, analyze_query_complexity, suggest_query_optimization,
        create_safe_query_alternative, safe_sql_execution, SQLTimeoutError, logger
    )

# Load environment variables from .env file FIRST
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from ..state_schemas import DataAnalysisState, DVDRentalSystemState, dataframe_to_dict, dict_to_dataframe, convert_numpy_types
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from state_schemas import DataAnalysisState, DVDRentalSystemState, dataframe_to_dict, dict_to_dataframe, convert_numpy_types


# ============================================================================
# Core Data Analysis Functions
# ============================================================================

def _create_engine(database_url: str) -> Engine:
    """Create SQLAlchemy engine"""
    return create_engine(database_url, pool_pre_ping=True, future=True)


@timeout_sql_execution(timeout_seconds=120)  # 2 minute timeout
def _execute_sql(engine: Engine, sql: str) -> pd.DataFrame:
    """Execute SQL query and return DataFrame with timeout protection"""
    logger.info(f"Executing SQL: {sql[:100]}...")
    
    # Analyze query complexity before execution
    complexity = analyze_query_complexity(sql)
    logger.info(f"Query complexity: {complexity['risk_level']} (score: {complexity['complexity_score']})")
    
    # If query is very risky, create a safer alternative
    if complexity["risk_level"] == "very_high":
        logger.warning("Query is very complex, creating safer alternative")
        optimization = suggest_query_optimization(sql, complexity)
        if optimization["should_warn_user"]:
            logger.warning(f"Query optimization suggestions: {optimization['suggestions']}")
            # Use the optimized query instead
            sql = optimization["optimized_sql"]
            logger.info(f"Using optimized SQL: {sql[:100]}...")
    
    with engine.connect() as conn:
        start_time = time.time()
        try:
            result = pd.read_sql(text(sql), conn)
            elapsed = time.time() - start_time
            logger.info(f"SQL executed successfully in {elapsed:.2f} seconds, returned {len(result)} rows")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"SQL execution failed after {elapsed:.2f} seconds: {e}")
            raise


def _score_dataframe(df: pd.DataFrame) -> float:
    """Score DataFrame quality for ranking SQL candidates"""
    if df is None or df.empty:
        return 0.0
    
    rows, cols = df.shape
    if rows <= 0 or cols <= 0:
        return 0.1
    
    # Size score: more rows is better (up to a point)
    size_score = min(rows / 50.0, 1.0)
    
    # Variety score: mix of numeric and categorical is good
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    var_score = 0.0
    if num_cols:
        var_score = 0.5
    
    # Completeness score: fewer nulls is better
    null_ratio = convert_numpy_types(df.isnull().sum().sum()) / (rows * cols) if rows * cols > 0 else 1.0
    completeness_score = 1.0 - null_ratio
    
    return float((size_score + var_score + completeness_score) / 3.0)


def _analyze_data_characteristics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze DataFrame characteristics for visualization planning"""
    if df.empty:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "columns": [],
            "has_numeric": False,
            "has_categorical": False,
            "has_temporal": False
        }
    
    characteristics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": [],
        "has_numeric": False,
        "has_categorical": False,
        "has_temporal": False
    }
    
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_values": convert_numpy_types(df[col].nunique()),
            "null_count": convert_numpy_types(df[col].isnull().sum()),
            "sample_values": convert_numpy_types(df[col].dropna().head(3).tolist())
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["type"] = "numeric"
            col_info["min"] = convert_numpy_types(df[col].min()) if not df[col].empty else None
            col_info["max"] = convert_numpy_types(df[col].max()) if not df[col].empty else None
            col_info["mean"] = convert_numpy_types(df[col].mean()) if not df[col].empty else None
            characteristics["has_numeric"] = True
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["type"] = "datetime"
            characteristics["has_temporal"] = True
        elif any(keyword in col.lower() for keyword in ["date", "time", "year", "month", "day"]):
            col_info["type"] = "temporal"
            characteristics["has_temporal"] = True
        else:
            col_info["type"] = "categorical"
            characteristics["has_categorical"] = True
            
        characteristics["columns"].append(col_info)
    
    return characteristics


def _compute_basic_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic statistics for the dataset"""
    if df.empty:
        return {}
    
    stats = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "numeric_columns": [],
        "categorical_columns": []
    }
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats = {
                "name": col,
                "mean": convert_numpy_types(df[col].mean()) if not df[col].empty else None,
                "median": convert_numpy_types(df[col].median()) if not df[col].empty else None,
                "std": convert_numpy_types(df[col].std()) if not df[col].empty else None,
                "min": convert_numpy_types(df[col].min()) if not df[col].empty else None,
                "max": convert_numpy_types(df[col].max()) if not df[col].empty else None,
                "null_count": convert_numpy_types(df[col].isnull().sum())
            }
            stats["numeric_columns"].append(col_stats)
        else:
            col_stats = {
                "name": col,
                "unique_count": convert_numpy_types(df[col].nunique()),
                "most_common": convert_numpy_types(df[col].mode().tolist()[:3]) if not df[col].empty else [],
                "null_count": convert_numpy_types(df[col].isnull().sum())
            }
            stats["categorical_columns"].append(col_stats)
    
    return stats


# ============================================================================
# LangGraph Nodes
# ============================================================================

def execute_sql_candidates(state: DataAnalysisState) -> Dict[str, Any]:
    """Node: Execute SQL candidates and select best result with timeout protection"""
    candidates = state.get("sql_candidates", [])
    
    # Try to find the database file
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
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
    
    if not candidates:
        logger.warning("No SQL candidates provided for execution")
        return {
            "selected_sql": "",
            "query_results": {},
            "data_quality_score": 0.0,
            "execution_errors": ["No SQL candidates provided"]
        }
    
    logger.info(f"Executing {len(candidates)} SQL candidates")
    engine = _create_engine(database_url)
    best_result = {"score": -1.0, "sql": "", "df": pd.DataFrame()}
    execution_errors = []
    timeout_occurred = False
    
    for i, sql in enumerate(candidates):
        try:
            logger.info(f"Trying SQL candidate {i+1}/{len(candidates)}")
            
            # Pre-analyze complexity
            complexity = analyze_query_complexity(sql)
            logger.info(f"Candidate {i+1} complexity: {complexity['risk_level']}")
            
            # If the query is extremely complex, try a safer alternative first
            if complexity["risk_level"] == "very_high":
                logger.warning(f"Candidate {i+1} is very complex, trying safer alternative")
                safe_sql = create_safe_query_alternative(sql)
                
                try:
                    df = _execute_sql(engine, safe_sql)
                    score = _score_dataframe(df)
                    
                    if score > best_result["score"]:
                        best_result = {"score": score, "sql": safe_sql, "df": df}
                        execution_errors.append(f"Used safer alternative for candidate {i+1} due to complexity")
                    continue  # Skip the original complex query
                    
                except Exception as safe_e:
                    logger.warning(f"Safer alternative also failed: {safe_e}")
                    # Continue to try the original query
            
            # Execute the original query
            df = _execute_sql(engine, sql)
            score = _score_dataframe(df)
            
            if score > best_result["score"]:
                best_result = {"score": score, "sql": sql, "df": df}
                
        except SQLTimeoutError as e:
            timeout_occurred = True
            error_msg = f"SQL candidate {i+1} timed out: {str(e)}"
            logger.error(error_msg)
            execution_errors.append(error_msg)
            
            # For timeout, suggest a much simpler query
            try:
                logger.info(f"Creating emergency fallback for candidate {i+1}")
                emergency_sql = sql.replace("LIMIT 10000", "LIMIT 10").replace("GROUP BY", "-- GROUP BY")
                if "GROUP BY" in emergency_sql:
                    # Remove GROUP BY and aggregations for emergency query
                    emergency_sql = f"SELECT * FROM ({sql.split('FROM')[0]} FROM {sql.split('FROM')[1].split('GROUP BY')[0]}) LIMIT 10"
                
                df = _execute_sql(engine, emergency_sql)
                score = _score_dataframe(df) * 0.5  # Penalty for emergency query
                
                if score > best_result["score"]:
                    best_result = {"score": score, "sql": emergency_sql, "df": df}
                    execution_errors.append(f"Used emergency fallback for timed-out candidate {i+1}")
                    
            except Exception as emergency_e:
                logger.error(f"Emergency fallback also failed: {emergency_e}")
                execution_errors.append(f"Emergency fallback for candidate {i+1} also failed")
            
            continue
            
        except Exception as e:
            error_msg = f"SQL candidate {i+1} failed: {str(e)}"
            logger.error(error_msg)
            execution_errors.append(error_msg)
            continue
    
    # Prepare results
    if best_result["score"] > 0:
        logger.info(f"Successfully executed SQL with quality score: {best_result['score']:.2f}")
        result = {
            "selected_sql": best_result["sql"],
            "query_results": dataframe_to_dict(best_result["df"]),
            "data_quality_score": best_result["score"],
            "execution_errors": execution_errors
        }
        
        if timeout_occurred:
            result["warnings"] = ["Some queries timed out due to complexity. Results may be simplified."]
        
        return result
    else:
        logger.error("All SQL candidates failed to execute")
        return {
            "selected_sql": candidates[0] if candidates else "",
            "query_results": dataframe_to_dict(pd.DataFrame()),
            "data_quality_score": 0.0,
            "execution_errors": execution_errors + ["All SQL candidates failed to execute"],
            "warnings": ["Query too complex - consider simplifying your request"] if timeout_occurred else []
        }


def analyze_data_characteristics(state: DataAnalysisState) -> Dict[str, Any]:
    """Node: Analyze data characteristics for visualization planning"""
    df_dict = state.get("query_results", {})
    df = dict_to_dataframe(df_dict)
    
    if df.empty:
        return {
            "data_characteristics": {
                "total_rows": 0,
                "total_columns": 0,
                "columns": []
            }
        }
    
    try:
        characteristics = _analyze_data_characteristics(df)
        return {"data_characteristics": convert_numpy_types(characteristics)}
    except Exception as e:
        return {
            "data_characteristics": {},
            "execution_errors": [f"Data analysis failed: {str(e)}"]
        }


def compute_basic_statistics(state: DataAnalysisState) -> Dict[str, Any]:
    """Node: Compute basic statistics"""
    df_dict = state.get("query_results", {})
    df = dict_to_dataframe(df_dict)
    
    if df.empty:
        return {
            "basic_statistics": {},
            "analysis_summary": "No data available for analysis"
        }
    
    try:
        stats = _compute_basic_statistics(df)
        
        # Generate summary
        summary_parts = [
            f"Dataset contains {stats['row_count']} rows and {stats['column_count']} columns.",
        ]
        
        if stats["numeric_columns"]:
            summary_parts.append(f"Numeric columns: {', '.join([c['name'] for c in stats['numeric_columns']])}")
        
        if stats["categorical_columns"]:
            summary_parts.append(f"Categorical columns: {', '.join([c['name'] for c in stats['categorical_columns']])}")
        
        analysis_summary = " ".join(summary_parts)
        
        return {
            "basic_statistics": convert_numpy_types(stats),
            "analysis_summary": analysis_summary
        }
    except Exception as e:
        return {
            "basic_statistics": {},
            "analysis_summary": f"Statistical analysis failed: {str(e)}",
            "execution_errors": [f"Statistics computation failed: {str(e)}"]
        }


def assess_data_quality(state: DataAnalysisState) -> Dict[str, Any]:
    """Node: Assess overall data quality"""
    df_dict = state.get("query_results", {})
    df = dict_to_dataframe(df_dict)
    current_score = state.get("data_quality_score", 0.0)
    
    if df.empty:
        return {
            "data_quality_score": 0.0,
            "analysis_summary": "No data available - quality score: 0.0"
        }
    
    try:
        # Recompute quality score with additional factors
        base_score = current_score
        
        # Penalty for too few rows
        if len(df) < 5:
            base_score *= 0.5
        
        # Penalty for too many nulls
        null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_ratio > 0.5:
            base_score *= 0.7
        
        # Bonus for having both numeric and categorical data
        has_numeric = any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
        has_categorical = any(not pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
        if has_numeric and has_categorical:
            base_score *= 1.1
        
        final_score = min(1.0, base_score)
        
        quality_assessment = f"Data quality score: {final_score:.2f} "
        if final_score > 0.8:
            quality_assessment += "(Excellent)"
        elif final_score > 0.6:
            quality_assessment += "(Good)"
        elif final_score > 0.4:
            quality_assessment += "(Fair)"
        else:
            quality_assessment += "(Poor)"
        
        return {
            "data_quality_score": final_score,
            "analysis_summary": state.get("analysis_summary", "") + f" {quality_assessment}"
        }
    except Exception as e:
        return {
            "data_quality_score": 0.0,
            "analysis_summary": f"Quality assessment failed: {str(e)}",
            "execution_errors": [f"Quality assessment failed: {str(e)}"]
        }


# ============================================================================
# Sub-graph Creation
# ============================================================================

def create_data_analysis_agent() -> StateGraph:
    """Create the data analysis agent sub-graph"""
    builder = StateGraph(DataAnalysisState)
    
    # Add nodes
    builder.add_node("execute_sql_candidates", execute_sql_candidates)
    builder.add_node("analyze_data_characteristics", analyze_data_characteristics)
    builder.add_node("compute_basic_statistics", compute_basic_statistics)
    builder.add_node("assess_data_quality", assess_data_quality)
    
    # Add edges (sequential flow)
    builder.add_edge(START, "execute_sql_candidates")
    builder.add_edge("execute_sql_candidates", "analyze_data_characteristics")
    builder.add_edge("analyze_data_characteristics", "compute_basic_statistics")
    builder.add_edge("compute_basic_statistics", "assess_data_quality")
    builder.add_edge("assess_data_quality", END)
    
    # Compile with memory for persistence
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ============================================================================
# System Integration Functions
# ============================================================================

def run_data_analysis_agent(system_state: DVDRentalSystemState) -> Dict[str, Any]:
    """Run data analysis agent and return updates for system state"""
    # Extract data analysis-specific state
    analysis_state = {
        "sql_candidates": system_state.get("sql_candidates", []),
        "selected_sql": "",
        "query_results": {},
        "data_characteristics": {},
        "basic_statistics": {},
        "data_quality_score": 0.0,
        "analysis_summary": "",
        "execution_errors": []
    }
    
    # Create and run the agent
    agent = create_data_analysis_agent()
    
    # Execute the data analysis agent workflow
    thread = {"configurable": {"thread_id": f"data_analysis_agent_{int(time.time())}"}}
    result = agent.invoke(analysis_state, thread)
    
    # Return updates for the main system state
    return {
        "selected_sql": result.get("selected_sql", ""),
        "query_results": result.get("query_results", {}),
        "data_characteristics": result.get("data_characteristics", {}),
        "data_quality_score": result.get("data_quality_score", 0.0),
        "basic_statistics": result.get("basic_statistics", {}),
        "analysis_summary": result.get("analysis_summary", ""),
        "errors": result.get("execution_errors", []),
        "current_stage": "data_analysis_complete"
    }
