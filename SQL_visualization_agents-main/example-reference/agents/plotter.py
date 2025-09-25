"""
Intelligent Plot Agent
- Uses LLM to understand user intent and data characteristics
- Intelligently selects plot type, mode, and configuration
- Supports bar charts (grouped/stacked), line charts, scatter plots, histograms, box plots
- Executes SQL candidates and chooses the best result
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


def _execute_sql(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def _score_df(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    rows, cols = df.shape
    if rows <= 0 or cols <= 0:
        return 0.1
    size_score = min(rows / 50.0, 1.0)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    var_score = 0.0
    if num_cols:
        var_score = 0.5
    return size_score + var_score


def _choose_best(engine, candidates: List[str]) -> Dict[str, Any]:
    best = {"score": -1.0}
    for sql in candidates:
        try:
            df = _execute_sql(engine, sql)
            score = _score_df(df)
            if score > best.get("score", -1):
                best = {"sql": sql, "df": df, "score": score}
        except Exception:
            continue
    return best if best.get("score", 0) > 0 else {}


def _analyze_data_characteristics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the dataframe to understand its characteristics for plotting"""
    characteristics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": []
    }
    
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_values": df[col].nunique(),
            "null_count": df[col].isnull().sum(),
            "sample_values": df[col].dropna().head(3).tolist()
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["type"] = "numeric"
            col_info["min"] = float(df[col].min()) if not df[col].empty else None
            col_info["max"] = float(df[col].max()) if not df[col].empty else None
            col_info["mean"] = float(df[col].mean()) if not df[col].empty else None
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["type"] = "datetime"
        elif any(keyword in col.lower() for keyword in ["date", "time", "year", "month", "day"]):
            col_info["type"] = "temporal"
        else:
            col_info["type"] = "categorical"
            
        characteristics["columns"].append(col_info)
    
    return characteristics


def _get_plot_specification(user_request: str, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to determine the best plot specification based on user intent and data"""
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    system_prompt = """You are a data visualization expert. Given a user request and data characteristics, 
determine the best plot configuration. You must respond with ONLY a valid JSON object with these exact fields:

{
  "plot_type": "bar|line|scatter|histogram|box|table",
  "x_column": "column_name",
  "y_column": "column_name", 
  "color_column": "column_name or null",
  "barmode": "group|stack|overlay|null",
  "orientation": "v|h|null",
  "reasoning": "brief explanation of choice"
}

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

Choose columns that best match the user's intent. Prefer meaningful column names for axes."""

    data_summary = f"""
User Request: {user_request}

Data Characteristics:
- Rows: {data_characteristics['total_rows']}
- Columns: {data_characteristics['total_columns']}

Column Details:
"""
    
    for col in data_characteristics['columns']:
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
    columns = data_characteristics['columns']
    
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
            "reasoning": "Fallback: categorical with numeric values"
        }
    else:
        return {
            "plot_type": "table",
            "x_column": None,
            "y_column": None,
            "color_column": None,
            "barmode": None,
            "orientation": None,
            "reasoning": "Fallback: data not suitable for plotting"
        }


def _create_intelligent_figure(df: pd.DataFrame, plot_spec: Dict[str, Any]):
    """Create a plotly figure based on the intelligent plot specification"""
    
    plot_type = plot_spec.get("plot_type")
    x_col = plot_spec.get("x_column")
    y_col = plot_spec.get("y_column") 
    color_col = plot_spec.get("color_column")
    barmode = plot_spec.get("barmode")
    orientation = plot_spec.get("orientation", "v")
    
    # Handle None values
    color_col = color_col if color_col != "null" else None
    barmode = barmode if barmode != "null" else None
    
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
        else:
            return None
            
        # Update layout with better styling
        fig.update_layout(
            title=plot_spec.get("reasoning", "Data Visualization"),
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating figure: {e}")
        return None


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    user_request = state.get("user_request") or "Visualization"
    database_url = state.get("database_url")
    candidates = state.get("sql_candidates") or []

    if not database_url:
        return {"error": "DATABASE_URL not set"}
    if not candidates:
        return {"error": "No sql_candidates provided"}

    engine = create_engine(database_url, pool_pre_ping=True, future=True)
    best = _choose_best(engine, candidates)

    if not best:
        return {"error": "All candidates failed; please refine the request"}

    df = best["df"]
    
    # Analyze data characteristics
    data_characteristics = _analyze_data_characteristics(df)
    
    # Get intelligent plot specification from LLM
    plot_spec = _get_plot_specification(user_request, data_characteristics)
    
    # Create the figure
    fig = _create_intelligent_figure(df, plot_spec)

    if fig is None:
        # Fallback to table if plotting fails
        html = df.to_html(index=False)
        return {
            "chosen_sql": best["sql"], 
            "plot_spec": plot_spec, 
            "figure_html": html,
            "data_characteristics": data_characteristics
        }

    html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return {
        "chosen_sql": best["sql"], 
        "plot_spec": plot_spec, 
        "figure": fig, 
        "figure_html": html,
        "data_characteristics": data_characteristics
    }


if __name__ == "__main__":
    print("Intelligent Plot Agent: Uses LLM to determine optimal visualization based on user intent and data characteristics.") 