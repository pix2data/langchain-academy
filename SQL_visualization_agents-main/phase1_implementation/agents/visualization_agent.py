"""
Intelligent Visualization Agent (LangGraph Implementation)
Based on existing plotter.py functionality with LangGraph orchestration
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables from .env file FIRST
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from ..state_schemas import VisualizationState, DVDRentalSystemState, dict_to_dataframe, figure_to_dict, dict_to_figure
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from state_schemas import VisualizationState, DVDRentalSystemState, dict_to_dataframe, figure_to_dict, dict_to_figure


# ============================================================================
# Core Visualization Functions (from existing plotter.py)
# ============================================================================

def _get_plot_specification(user_request: str, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to determine the best plot specification based on user intent and data"""
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    system_prompt = """You are a data visualization expert. Given a user request and data characteristics, 
determine the best plot configuration. You must respond with ONLY a valid JSON object with these exact fields:

{
  "plot_type": "bar|line|scatter|histogram|box|table",
  "x_column": "exact_column_name_from_data",
  "y_column": "exact_column_name_from_data", 
  "color_column": "exact_column_name_from_data or null",
  "barmode": "group|stack|overlay|null",
  "orientation": "v|h|null",
  "reasoning": "brief explanation of choice"
}

CRITICAL: You must use ONLY the exact column names provided in the data. Do NOT make up column names.

Available plot types and when to use them:
- bar: comparing categories, showing totals/counts (use barmode: "group" for side-by-side, "stack" for stacked)
- line: showing trends over time or continuous data
- scatter: showing relationships between two numeric variables
- histogram: showing distribution of a single numeric variable
- box: showing distribution and outliers
- table: when data is better shown as a table

Barmode options:
- "group": bars side by side (good for comparing multiple series)
- "stack": bars stacked on top (good for showing parts of a whole)
- "overlay": bars overlapping (rarely used)
- null: single series bar chart

IMPORTANT: Only use column names that exist in the provided data. Look at the "Column Details" section below for exact column names."""

    data_summary = f"""
User Request: {user_request}

Data Characteristics:
- Rows: {data_characteristics.get('total_rows', 0)}
- Columns: {data_characteristics.get('total_columns', 0)}

Column Details:
"""
    
    for col in data_characteristics.get('columns', []):
        data_summary += f"- {col['name']}: {col['type']}, {col['unique_values']} unique values, sample: {col['sample_values']}\n"
    
    try:
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=data_summary)
        ])
        
        # Extract JSON from response
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
            
        plot_spec = json.loads(content)
        
        # Validate required fields
        required_fields = ["plot_type", "x_column", "y_column"]
        for field in required_fields:
            if field not in plot_spec:
                raise ValueError(f"Missing required field: {field}")
                
        return plot_spec
        
    except Exception as e:
        # Fallback to simple heuristic if LLM fails
        print(f"LLM plot specification failed: {e}")
        return _fallback_plot_spec(data_characteristics)


def _fallback_plot_spec(data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback heuristic when LLM fails"""
    columns = data_characteristics.get('columns', [])
    
    numeric_cols = [c for c in columns if c['type'] == 'numeric']
    categorical_cols = [c for c in columns if c['type'] == 'categorical']
    temporal_cols = [c for c in columns if c['type'] in ['datetime', 'temporal']]
    
    if temporal_cols and numeric_cols:
        return {
            "plot_type": "line",
            "x_column": temporal_cols[0]['name'],
            "y_column": numeric_cols[0]['name'],
            "color_column": None,
            "barmode": None,
            "orientation": "v",
            "reasoning": "Fallback: temporal data with numeric values"
        }
    elif categorical_cols and numeric_cols:
        return {
            "plot_type": "bar", 
            "x_column": categorical_cols[0]['name'],
            "y_column": numeric_cols[0]['name'],
            "color_column": None,
            "barmode": None,
            "orientation": "v",
            "reasoning": "Fallback: categorical vs numeric comparison"
        }
    elif len(numeric_cols) >= 2:
        return {
            "plot_type": "scatter",
            "x_column": numeric_cols[0]['name'],
            "y_column": numeric_cols[1]['name'],
            "color_column": None,
            "barmode": None,
            "orientation": None,
            "reasoning": "Fallback: two numeric variables"
        }
    elif len(numeric_cols) == 1:
        return {
            "plot_type": "histogram",
            "x_column": numeric_cols[0]['name'],
            "y_column": "count",
            "color_column": None,
            "barmode": None,
            "orientation": None,
            "reasoning": "Fallback: single numeric distribution"
        }
    else:
        return {
            "plot_type": "table",
            "x_column": columns[0]['name'] if columns else "data",
            "y_column": columns[1]['name'] if len(columns) > 1 else "value",
            "color_column": None,
            "barmode": None,
            "orientation": None,
            "reasoning": "Fallback: data not suitable for plotting"
        }


def _validate_and_fix_columns(df: pd.DataFrame, plot_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix column names in plot specification to match actual data"""
    available_cols = list(df.columns)
    fixed_spec = plot_spec.copy()
    
    def find_best_column(desired_name: str, column_type: str = None) -> str:
        """Find the best matching column for the desired name"""
        if not desired_name or desired_name == "null":
            return None
        
        # Exact match first
        if desired_name in available_cols:
            return desired_name
        
        # Partial match based on keywords
        desired_lower = desired_name.lower()
        
        # Common mappings for expected vs actual column names
        column_mappings = {
            "customer_segment": ["active", "activebool", "first_name", "customer_id"],
            "total_revenue": ["total_revenue", "amount", "revenue", "sum"],
            "revenue": ["total_revenue", "amount", "revenue", "sum"],
            "month": ["month", "payment_date", "date"],
            "rating": ["rating", "film_rating"],
            "category": ["name", "category", "title"],
            "film_rating": ["rating", "film_rating"]
        }
        
        # Try mapping first
        if desired_lower in column_mappings:
            for candidate in column_mappings[desired_lower]:
                for col in available_cols:
                    if candidate.lower() in col.lower():
                        return col
        
        # Fuzzy matching by type
        if column_type == "numeric":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                return numeric_cols[0]
        elif column_type == "categorical":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                return categorical_cols[0]
        
        # Keyword-based fuzzy matching
        for col in available_cols:
            if any(keyword in col.lower() for keyword in desired_lower.split('_')):
                return col
        
        # Last resort: return first available column of appropriate type
        if column_type == "numeric":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            return numeric_cols[0] if numeric_cols else available_cols[0] if available_cols else None
        elif column_type == "categorical":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            return categorical_cols[0] if categorical_cols else available_cols[0] if available_cols else None
        
        return available_cols[0] if available_cols else None
    
    # Fix x_column
    x_col = plot_spec.get("x_column")
    if x_col and x_col not in available_cols:
        fixed_x = find_best_column(x_col, "categorical")
        if fixed_x:
            fixed_spec["x_column"] = fixed_x
            print(f"Fixed x_column: '{x_col}' → '{fixed_x}'")
    
    # Fix y_column  
    y_col = plot_spec.get("y_column")
    if y_col and y_col not in available_cols:
        fixed_y = find_best_column(y_col, "numeric")
        if fixed_y:
            fixed_spec["y_column"] = fixed_y
            print(f"Fixed y_column: '{y_col}' → '{fixed_y}'")
    
    # Fix color_column
    color_col = plot_spec.get("color_column")
    if color_col and color_col != "null" and color_col not in available_cols:
        fixed_color = find_best_column(color_col, "categorical")
        if fixed_color:
            fixed_spec["color_column"] = fixed_color
            print(f"Fixed color_column: '{color_col}' → '{fixed_color}'")
        else:
            fixed_spec["color_column"] = None
    
    return fixed_spec


def _create_intelligent_figure(df: pd.DataFrame, plot_spec: Dict[str, Any]) -> go.Figure:
    """Create a plotly figure based on the intelligent plot specification"""
    
    # Validate and fix column names first
    fixed_spec = _validate_and_fix_columns(df, plot_spec)
    
    plot_type = fixed_spec.get("plot_type")
    x_col = fixed_spec.get("x_column")
    y_col = fixed_spec.get("y_column") 
    color_col = fixed_spec.get("color_column")
    barmode = fixed_spec.get("barmode")
    orientation = fixed_spec.get("orientation", "v")
    
    # Handle None values
    color_col = color_col if color_col != "null" and color_col is not None else None
    barmode = barmode if barmode != "null" and barmode is not None else None
    
    # Additional safety check
    if x_col not in df.columns:
        print(f"Warning: x_column '{x_col}' not found, using first column")
        x_col = df.columns[0] if len(df.columns) > 0 else None
    
    if y_col not in df.columns:
        print(f"Warning: y_column '{y_col}' not found, using first numeric column")
        numeric_cols = df.select_dtypes(include=['number']).columns
        y_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1] if len(df.columns) > 1 else None
    
    try:
        if plot_type == "bar":
            if orientation == "h":
                fig = px.bar(df, x=y_col, y=x_col, color=color_col, 
                           orientation='h', barmode=barmode)
            else:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                           barmode=barmode)
        elif plot_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col)
        elif plot_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
        elif plot_type == "histogram":
            fig = px.histogram(df, x=x_col, color=color_col)
        elif plot_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=color_col)
        elif plot_type == "table":
            # Create a simple table visualization
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns)),
                cells=dict(values=[df[col] for col in df.columns])
            )])
        else:
            # Create a simple bar chart as default
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                fig = px.bar(df, x=df.columns[0], y=numeric_cols[0])
            else:
                fig = go.Figure()
        
        # Update layout with better styling
        fig.update_layout(
            title=plot_spec.get("reasoning", "Data Visualization"),
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_white",
            showlegend=True if color_col else False
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure if creation fails
        print(f"Figure creation failed even after column fixing: {e}")
        print(f"Available columns: {list(df.columns)}")
        print(f"Attempted spec: x='{x_col}', y='{y_col}', color='{color_col}'")
        
        # Emergency fallback: create a simple table
        try:
            return go.Figure(data=[go.Table(
                header=dict(values=list(df.columns)),
                cells=dict(values=[df[col] for col in df.columns])
            )])
        except:
            return go.Figure()


def _generate_alternative_plots(data_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate alternative plot suggestions"""
    columns = data_characteristics.get('columns', [])
    numeric_cols = [c for c in columns if c['type'] == 'numeric']
    categorical_cols = [c for c in columns if c['type'] == 'categorical']
    
    alternatives = []
    
    # If we have categorical and numeric, suggest different combinations
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols[:2]:
            for num_col in numeric_cols[:2]:
                alternatives.append({
                    "plot_type": "bar",
                    "x_column": cat_col['name'],
                    "y_column": num_col['name'],
                    "description": f"Bar chart: {cat_col['name']} vs {num_col['name']}"
                })
    
    # If we have multiple numeric columns, suggest scatter plots
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:3]:
                alternatives.append({
                    "plot_type": "scatter",
                    "x_column": col1['name'],
                    "y_column": col2['name'],
                    "description": f"Scatter plot: {col1['name']} vs {col2['name']}"
                })
    
    return alternatives[:5]  # Limit to 5 alternatives


# ============================================================================
# LangGraph Nodes
# ============================================================================

def analyze_visualization_intent(state: VisualizationState) -> Dict[str, Any]:
    """Node: Parse user's visualization goals from original request"""
    user_request = state.get("user_request", "")
    data_characteristics = state.get("data_characteristics", {})
    
    if not user_request:
        return {
            "messages": [AIMessage(content="No user request provided for visualization analysis")]
        }
    
    # Simple intent analysis based on keywords
    intent_keywords = {
        "trend": ["over time", "trend", "change", "progression", "timeline"],
        "comparison": ["compare", "versus", "vs", "between", "difference"],
        "distribution": ["distribution", "spread", "histogram", "frequency"],
        "relationship": ["relationship", "correlation", "association", "connection"],
        "ranking": ["top", "bottom", "highest", "lowest", "best", "worst", "rank"]
    }
    
    detected_intent = "general"
    for intent, keywords in intent_keywords.items():
        if any(keyword in user_request.lower() for keyword in keywords):
            detected_intent = intent
            break
    
    return {
        "plot_reasoning": f"Detected user intent: {detected_intent}",
        "messages": [AIMessage(content=f"Analyzing visualization intent: {detected_intent}")]
    }


def get_plot_specification(state: VisualizationState) -> Dict[str, Any]:
    """Node: LLM determines optimal plot type and configuration"""
    user_request = state.get("user_request", "")
    data_characteristics = state.get("data_characteristics", {})
    
    if not data_characteristics or not data_characteristics.get("columns"):
        return {
            "plot_specification": {
                "plot_type": "table",
                "x_column": "data",
                "y_column": "value",
                "reasoning": "No data characteristics available"
            },
            "messages": [AIMessage(content="No data available for plot specification")]
        }
    
    try:
        plot_spec = _get_plot_specification(user_request, data_characteristics)
        return {
            "plot_specification": plot_spec,
            "plot_reasoning": plot_spec.get("reasoning", "Plot specification generated"),
            "messages": [AIMessage(content=f"Selected plot type: {plot_spec['plot_type']} - {plot_spec.get('reasoning', '')}")]
        }
    except Exception as e:
        fallback_spec = _fallback_plot_spec(data_characteristics)
        return {
            "plot_specification": fallback_spec,
            "plot_reasoning": f"Fallback specification used due to error: {str(e)}",
            "messages": [AIMessage(content=f"Using fallback plot specification: {fallback_spec['plot_type']}")]
        }


def create_visualization(state: VisualizationState) -> Dict[str, Any]:
    """Node: Generate Plotly chart based on specification"""
    query_results_dict = state.get("query_results", {})
    query_results = dict_to_dataframe(query_results_dict)
    plot_specification = state.get("plot_specification", {})
    
    if query_results.empty:
        return {
            "generated_plot": figure_to_dict(go.Figure()),
            "plot_html": "<p>No data available for visualization</p>",
            "messages": [AIMessage(content="Cannot create visualization: no data available")]
        }
    
    try:
        fig = _create_intelligent_figure(query_results, plot_specification)
        plot_html = fig.to_html(include_plotlyjs='cdn', div_id="plotly-div")
        
        return {
            "generated_plot": figure_to_dict(fig),
            "plot_html": plot_html,
            "messages": [AIMessage(content=f"Created {plot_specification.get('plot_type', 'unknown')} visualization")]
        }
    except Exception as e:
        return {
            "generated_plot": figure_to_dict(go.Figure()),
            "plot_html": f"<p>Visualization creation failed: {str(e)}</p>",
            "messages": [AIMessage(content=f"Visualization creation failed: {str(e)}")]
        }


def generate_alternatives(state: VisualizationState) -> Dict[str, Any]:
    """Node: Create alternative plot suggestions"""
    data_characteristics = state.get("data_characteristics", {})
    
    if not data_characteristics:
        return {
            "alternative_plots": [],
            "messages": [AIMessage(content="No alternatives available without data characteristics")]
        }
    
    try:
        alternatives = _generate_alternative_plots(data_characteristics)
        return {
            "alternative_plots": alternatives,
            "messages": [AIMessage(content=f"Generated {len(alternatives)} alternative visualization options")]
        }
    except Exception as e:
        return {
            "alternative_plots": [],
            "messages": [AIMessage(content=f"Failed to generate alternatives: {str(e)}")]
        }


def human_review(state: VisualizationState) -> Dict[str, Any]:
    """Node: Human-in-the-loop visualization review"""
    # This is a no-op node that should be interrupted on
    plot_type = state.get("plot_specification", {}).get("plot_type", "unknown")
    
    return {
        "messages": [AIMessage(content=f"Visualization ({plot_type}) ready for review. Waiting for human feedback...")]
    }


def refine_visualization(state: VisualizationState) -> Dict[str, Any]:
    """Node: Modify visualization based on user feedback"""
    feedback = state.get("human_feedback", "")
    current_spec = state.get("plot_specification", {})
    
    if not feedback:
        return {
            "messages": [AIMessage(content="No feedback provided, keeping current visualization")]
        }
    
    # Simple feedback processing (in a more sophisticated implementation, 
    # we could use LLM to interpret feedback and modify the plot spec)
    feedback_lower = feedback.lower()
    
    if "horizontal" in feedback_lower or "flip" in feedback_lower:
        current_spec["orientation"] = "h"
    elif "vertical" in feedback_lower:
        current_spec["orientation"] = "v"
    
    if "stacked" in feedback_lower or "stack" in feedback_lower:
        current_spec["barmode"] = "stack"
    elif "grouped" in feedback_lower or "group" in feedback_lower:
        current_spec["barmode"] = "group"
    
    return {
        "plot_specification": current_spec,
        "messages": [AIMessage(content=f"Applied feedback: {feedback}")]
    }


# ============================================================================
# Routing Logic
# ============================================================================

def should_get_human_review(state: VisualizationState) -> str:
    """Determine if human review is needed"""
    # For Phase 1, we'll skip human review by default
    # In future phases, this could be based on complexity or user preference
    return END


def should_refine_visualization(state: VisualizationState) -> str:
    """Determine if visualization needs refinement based on feedback"""
    feedback = state.get("human_feedback")
    
    if feedback and feedback.strip():
        return "refine_visualization"
    
    return END


# ============================================================================
# Sub-graph Creation
# ============================================================================

def create_visualization_agent() -> StateGraph:
    """Create the visualization agent sub-graph"""
    builder = StateGraph(VisualizationState)
    
    # Add nodes
    builder.add_node("analyze_visualization_intent", analyze_visualization_intent)
    builder.add_node("get_plot_specification", get_plot_specification)
    builder.add_node("create_visualization", create_visualization)
    builder.add_node("generate_alternatives", generate_alternatives)
    builder.add_node("human_review", human_review)
    builder.add_node("refine_visualization", refine_visualization)
    
    # Add edges
    builder.add_edge(START, "analyze_visualization_intent")
    builder.add_edge("analyze_visualization_intent", "get_plot_specification")
    builder.add_edge("get_plot_specification", "create_visualization")
    builder.add_edge("create_visualization", "generate_alternatives")
    
    # Conditional edges for human review (disabled for Phase 1)
    builder.add_conditional_edges(
        "generate_alternatives",
        should_get_human_review,
        ["human_review", END]
    )
    
    # Handle refinement loop
    builder.add_conditional_edges(
        "human_review",
        should_refine_visualization,
        ["refine_visualization", END]
    )
    
    # Refinement back to creation
    builder.add_edge("refine_visualization", "create_visualization")
    
    # Compile with memory for persistence
    memory = MemorySaver()
    return builder.compile(
        interrupt_before=[],  # No interrupts for Phase 1
        checkpointer=memory
    )


# ============================================================================
# System Integration Functions
# ============================================================================

def run_visualization_agent(system_state: DVDRentalSystemState) -> Dict[str, Any]:
    """Run visualization agent and return updates for system state"""
    # Extract visualization-specific state
    viz_state = {
        "messages": [],
        "user_request": system_state["user_request"],
        "query_results": system_state.get("query_results", {}),
        "data_characteristics": system_state.get("data_characteristics", {}),
        "plot_specification": {},
        "generated_plot": {},
        "plot_html": "",
        "plot_reasoning": "",
        "alternative_plots": [],
        "human_feedback": system_state.get("human_feedback")
    }
    
    # Create and run the agent
    agent = create_visualization_agent()
    
    # Execute the visualization agent workflow
    thread = {"configurable": {"thread_id": f"viz_agent_{int(time.time())}"}}
    result = agent.invoke(viz_state, thread)
    
    # Return updates for the main system state
    return {
        "plot_specification": result.get("plot_specification", {}),
        "generated_plot": result.get("generated_plot", {}),
        "plot_html": result.get("plot_html", ""),
        "plot_reasoning": result.get("plot_reasoning", ""),
        "alternative_plots": result.get("alternative_plots", []),
        "current_stage": "visualization_complete"
    }
