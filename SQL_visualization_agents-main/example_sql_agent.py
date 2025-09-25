"""
Example Implementation: SQL Query Generation Agent
Based on LangGraph patterns from the research assistant example
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import pandas as pd

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# ============================================================================
# State Definitions (following LangGraph patterns)
# ============================================================================

class SQLQueryState(MessagesState):
    """State for SQL query generation agent - extends MessagesState for conversation"""
    available_tables: Dict[str, Dict]  # Table schemas from ingestion agent
    relationships: List[Dict]  # Foreign key relationships
    user_question: str  # Natural language query
    generated_sql: str  # Generated SQL query
    query_explanation: str  # Human-readable explanation
    sample_data: Optional[Dict[str, Any]]  # Sample query results
    confidence_score: float  # Query generation confidence
    validation_status: str  # "pending", "approved", "rejected"
    human_feedback: Optional[str]  # Human feedback for query refinement

class SQLQuery(BaseModel):
    """Structured output for SQL generation"""
    sql_query: str = Field(description="The generated SQL query")
    explanation: str = Field(description="Explanation of the query logic")
    confidence: float = Field(description="Confidence score 0-1")
    tables_used: List[str] = Field(description="List of tables referenced in query")

# ============================================================================
# Tools (following LangGraph tool patterns)
# ============================================================================

@tool
def analyze_table_schema(table_info: Dict[str, Dict]) -> str:
    """
    Analyze database schema to understand table structures and relationships.
    
    Args:
        table_info: Dictionary containing table schemas
        
    Returns:
        Formatted schema analysis
    """
    analysis = []
    for table_name, schema in table_info.items():
        columns = schema.get('columns', [])
        primary_keys = schema.get('primary_keys', [])
        foreign_keys = schema.get('foreign_keys', [])
        
        analysis.append(f"Table: {table_name}")
        analysis.append(f"  Columns: {', '.join(columns)}")
        if primary_keys:
            analysis.append(f"  Primary Keys: {', '.join(primary_keys)}")
        if foreign_keys:
            analysis.append(f"  Foreign Keys: {', '.join([f'{fk[0]} -> {fk[1]}' for fk in foreign_keys])}")
    
    return "\n".join(analysis)

@tool
def validate_sql_syntax(sql_query: str) -> Dict[str, Any]:
    """
    Validate SQL query syntax and basic structure.
    
    Args:
        sql_query: SQL query string to validate
        
    Returns:
        Validation results with syntax check and suggestions
    """
    # Simplified validation - in real implementation, use proper SQL parser
    issues = []
    score = 1.0
    
    if not sql_query.strip().upper().startswith(('SELECT', 'WITH')):
        issues.append("Query should start with SELECT or WITH")
        score -= 0.3
    
    if ';' not in sql_query and not sql_query.strip().endswith(';'):
        issues.append("Consider ending query with semicolon")
        score -= 0.1
    
    # Check for potential injection patterns
    dangerous_patterns = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']
    for pattern in dangerous_patterns:
        if pattern in sql_query.upper():
            issues.append(f"Query contains potentially dangerous operation: {pattern}")
            score -= 0.5
    
    return {
        "is_valid": len(issues) == 0 or score > 0.5,
        "score": max(0, score),
        "issues": issues,
        "suggestions": ["Use proper WHERE clauses for filtering", "Consider using LIMIT for large datasets"]
    }

@tool  
def execute_sample_query(sql_query: str, table_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute query on sample data to validate results.
    
    Args:
        sql_query: SQL query to execute
        table_data: Sample table data for testing
        
    Returns:
        Sample results and execution status
    """
    try:
        # In real implementation, use actual SQL engine
        # This is a simplified mock
        return {
            "success": True,
            "sample_rows": [
                {"column1": "sample_value1", "column2": 123},
                {"column1": "sample_value2", "column2": 456}
            ],
            "row_count": 150,
            "execution_time_ms": 45
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "suggestion": "Check table names and column references"
        }

# ============================================================================
# Agent Nodes (following LangGraph node patterns)
# ============================================================================

def analyze_schema_and_relationships(state: SQLQueryState) -> Dict[str, Any]:
    """
    Analyze available database schema and relationships.
    Similar to research assistant's context gathering phase.
    """
    table_info = state.get("available_tables", {})
    relationships = state.get("relationships", [])
    
    # Use tool to analyze schema
    schema_analysis = analyze_table_schema.invoke({"table_info": table_info})
    
    # Add relationship information
    relationship_info = "\n\nTable Relationships:\n"
    for rel in relationships:
        relationship_info += f"  {rel.get('from_table')}.{rel.get('from_column')} -> {rel.get('to_table')}.{rel.get('to_column')}\n"
    
    full_context = schema_analysis + relationship_info
    
    return {
        "messages": [
            SystemMessage(content=f"Database Schema Context:\n{full_context}")
        ]
    }

def generate_sql_query(state: SQLQueryState) -> Dict[str, Any]:
    """
    Generate SQL query from natural language question.
    Following the research assistant's question generation pattern.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Get user question and context
    user_question = state.get("user_question", "")
    messages = state.get("messages", [])
    
    # SQL generation instructions
    sql_instructions = """You are an expert SQL developer. Your task is to convert natural language questions into SQL queries.

Guidelines:
1. Use only the tables and columns mentioned in the schema context
2. Write clean, efficient SQL queries
3. Include proper WHERE clauses for filtering
4. Use appropriate JOINs when multiple tables are needed
5. Consider using LIMIT for potentially large result sets
6. Provide a clear explanation of your query logic

Generate a SQL query that answers this question: {question}

Respond with structured output including the query, explanation, and confidence score."""
    
    # Use structured output like in research assistant
    structured_llm = llm.with_structured_output(SQLQuery)
    
    system_message = SystemMessage(content=sql_instructions.format(question=user_question))
    query_result = structured_llm.invoke([system_message] + messages)
    
    return {
        "generated_sql": query_result.sql_query,
        "query_explanation": query_result.explanation,
        "confidence_score": query_result.confidence,
        "messages": [AIMessage(content=f"Generated SQL Query:\n{query_result.sql_query}\n\nExplanation: {query_result.explanation}")]
    }

def validate_query(state: SQLQueryState) -> Dict[str, Any]:
    """
    Validate the generated SQL query.
    Similar to research assistant's validation steps.
    """
    sql_query = state.get("generated_sql", "")
    
    # Validate syntax
    validation_result = validate_sql_syntax.invoke({"sql_query": sql_query})
    
    # Execute sample query
    sample_result = execute_sample_query.invoke({
        "sql_query": sql_query,
        "table_data": state.get("available_tables", {})
    })
    
    # Update confidence based on validation
    base_confidence = state.get("confidence_score", 0.5)
    validation_score = validation_result["score"]
    execution_success = sample_result["success"]
    
    final_confidence = base_confidence * validation_score * (1.0 if execution_success else 0.3)
    
    validation_message = f"""Query Validation Results:
Syntax Score: {validation_score:.2f}
Execution: {"Success" if execution_success else "Failed"}
Final Confidence: {final_confidence:.2f}

Sample Results: {sample_result.get('row_count', 0)} rows
"""
    
    return {
        "confidence_score": final_confidence,
        "sample_data": sample_result,
        "validation_status": "approved" if final_confidence > 0.7 else "needs_review",
        "messages": [AIMessage(content=validation_message)]
    }

def human_feedback_node(state: SQLQueryState) -> Dict[str, Any]:
    """
    Human-in-the-loop node for query approval.
    Following breakpoint pattern from research assistant.
    """
    # This is a no-op node that should be interrupted on
    # Allows human review of the generated query
    pass

def refine_query(state: SQLQueryState) -> Dict[str, Any]:
    """
    Refine query based on human feedback.
    Similar to research assistant's refinement process.
    """
    feedback = state.get("human_feedback", "")
    current_query = state.get("generated_sql", "")
    
    if not feedback:
        return {}  # No changes needed
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    refinement_prompt = f"""
    Please refine this SQL query based on the feedback provided:
    
    Current Query:
    {current_query}
    
    Feedback:
    {feedback}
    
    Provide an improved query that addresses the feedback while maintaining correctness.
    """
    
    structured_llm = llm.with_structured_output(SQLQuery)
    refined_result = structured_llm.invoke([HumanMessage(content=refinement_prompt)])
    
    return {
        "generated_sql": refined_result.sql_query,
        "query_explanation": refined_result.explanation,
        "confidence_score": refined_result.confidence,
        "validation_status": "pending",
        "messages": [AIMessage(content=f"Refined SQL Query:\n{refined_result.sql_query}")]
    }

# ============================================================================
# Routing Logic (following LangGraph conditional patterns)
# ============================================================================

def should_get_human_approval(state: SQLQueryState) -> str:
    """
    Determine if human approval is needed.
    Following research assistant's approval logic.
    """
    confidence = state.get("confidence_score", 0.0)
    validation_status = state.get("validation_status", "pending")
    
    # Require human approval for low confidence or failed validation
    if confidence < 0.7 or validation_status == "needs_review":
        return "human_feedback"
    
    return END

def should_refine_query(state: SQLQueryState) -> str:
    """
    Determine if query needs refinement based on feedback.
    """
    feedback = state.get("human_feedback")
    validation_status = state.get("validation_status", "pending")
    
    if feedback and validation_status != "approved":
        return "refine_query"
    
    return END

# ============================================================================
# Graph Construction (following LangGraph graph patterns)
# ============================================================================

def create_sql_agent() -> StateGraph:
    """
    Create the SQL query generation agent graph.
    Following the research assistant's graph construction pattern.
    """
    # Build the graph
    builder = StateGraph(SQLQueryState)
    
    # Add nodes
    builder.add_node("analyze_schema", analyze_schema_and_relationships)
    builder.add_node("generate_query", generate_sql_query)
    builder.add_node("validate_query", validate_query)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("refine_query", refine_query)
    
    # Add edges - following research assistant flow
    builder.add_edge(START, "analyze_schema")
    builder.add_edge("analyze_schema", "generate_query")
    builder.add_edge("generate_query", "validate_query")
    
    # Conditional edges for human approval
    builder.add_conditional_edges(
        "validate_query",
        should_get_human_approval,
        ["human_feedback", END]
    )
    
    # Handle refinement loop
    builder.add_conditional_edges(
        "human_feedback",
        should_refine_query,
        ["refine_query", END]
    )
    
    # Refinement loop back to validation
    builder.add_edge("refine_query", "validate_query")
    
    # Compile with memory for persistence (like research assistant)
    memory = MemorySaver()
    graph = builder.compile(
        interrupt_before=['human_feedback'],  # Breakpoint for human review
        checkpointer=memory
    )
    
    return graph

# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example usage following research assistant pattern
    graph = create_sql_agent()
    
    # Sample input state
    initial_state = {
        "user_question": "Show me the top 10 customers by total order value",
        "available_tables": {
            "customers": {
                "columns": ["customer_id", "name", "email", "created_date"],
                "primary_keys": ["customer_id"],
                "foreign_keys": []
            },
            "orders": {
                "columns": ["order_id", "customer_id", "order_date", "total_amount"],
                "primary_keys": ["order_id"],
                "foreign_keys": [["customer_id", "customers.customer_id"]]
            }
        },
        "relationships": [
            {
                "from_table": "orders",
                "from_column": "customer_id",
                "to_table": "customers", 
                "to_column": "customer_id"
            }
        ],
        "messages": []
    }
    
    # Thread configuration for persistence
    thread = {"configurable": {"thread_id": "sql_session_1"}}
    
    # Run the agent
    for event in graph.stream(initial_state, thread, stream_mode="values"):
        print("Current state:", event.get("validation_status", "processing"))
        if event.get("generated_sql"):
            print("Generated SQL:", event["generated_sql"])
            print("Confidence:", event.get("confidence_score", 0))
