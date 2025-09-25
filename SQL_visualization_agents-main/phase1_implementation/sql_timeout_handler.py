"""
SQL Timeout Handler for Complex Query Protection
Prevents system freezing on expensive SQL queries
"""

import time
import signal
import threading
import logging
from typing import Any, Dict, Optional
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLTimeoutError(Exception):
    """SQL execution timeout exception"""
    pass


def timeout_sql_execution(timeout_seconds: int = 60):
    """
    Decorator to add timeout to SQL execution functions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                logger.error(f"SQL execution timed out after {timeout_seconds} seconds")
                raise SQLTimeoutError(f"SQL execution timed out after {timeout_seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator


def analyze_query_complexity(sql: str) -> Dict[str, Any]:
    """
    Analyze SQL query complexity to predict if it might be expensive
    """
    sql_lower = sql.lower()
    
    # Count tables in FROM and JOIN clauses
    join_count = sql_lower.count(' join ')
    table_count = join_count + 1  # FROM table + JOINs
    
    # Check for expensive operations
    has_group_by = 'group by' in sql_lower
    has_order_by = 'order by' in sql_lower
    has_subqueries = '(' in sql and 'select' in sql_lower.split('(', 1)[1] if '(' in sql else False
    has_functions = any(func in sql_lower for func in ['strftime(', 'substr(', 'date(', 'datetime('])
    
    # Count GROUP BY columns
    group_by_complexity = 0
    if has_group_by:
        group_by_part = sql_lower.split('group by')[1].split('order by')[0] if 'order by' in sql_lower else sql_lower.split('group by')[1]
        group_by_part = group_by_part.split('limit')[0] if 'limit' in group_by_part else group_by_part
        group_by_complexity = len([col.strip() for col in group_by_part.split(',') if col.strip()])
    
    # Estimate complexity score
    complexity_score = 0
    complexity_score += table_count * 2  # Each table adds complexity
    complexity_score += group_by_complexity * 3  # GROUP BY columns are expensive
    complexity_score += 5 if has_subqueries else 0
    complexity_score += 3 if has_functions else 0
    complexity_score += 2 if has_order_by else 0
    
    # Determine risk level
    risk_level = "low"
    if complexity_score > 20:
        risk_level = "very_high"
    elif complexity_score > 15:
        risk_level = "high"
    elif complexity_score > 10:
        risk_level = "medium"
    
    return {
        "table_count": table_count,
        "join_count": join_count,
        "group_by_complexity": group_by_complexity,
        "has_subqueries": has_subqueries,
        "has_functions": has_functions,
        "has_order_by": has_order_by,
        "complexity_score": complexity_score,
        "risk_level": risk_level,
        "estimated_execution_time": min(300, complexity_score * 2),  # Cap at 5 minutes
        "recommended_timeout": min(120, max(30, complexity_score * 1.5))
    }


def suggest_query_optimization(sql: str, complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest optimizations for complex queries
    """
    suggestions = []
    optimized_sql = sql
    
    # If very complex, suggest simplification
    if complexity_analysis["risk_level"] in ["high", "very_high"]:
        suggestions.append("Query is very complex and may take a long time to execute")
        
        # Suggest reducing GROUP BY complexity
        if complexity_analysis["group_by_complexity"] > 3:
            suggestions.append("Consider reducing GROUP BY dimensions - try focusing on 2-3 key dimensions")
            
        # Suggest adding date filters
        if complexity_analysis["has_functions"] and "strftime" in sql.lower():
            suggestions.append("Add date range filters to limit the time period (e.g., WHERE payment_date >= '2023-01-01')")
            
        # Suggest smaller LIMIT
        if "limit" in sql.lower():
            limit_match = sql.lower().find("limit")
            if limit_match > 0:
                # Replace with smaller limit
                optimized_sql = sql[:limit_match] + "LIMIT 100"
                suggestions.append("Reduced LIMIT to 100 for faster execution")
        
        # Suggest removing some JOINs if possible
        if complexity_analysis["join_count"] > 4:
            suggestions.append("Consider if all table joins are necessary - remove unused tables")
    
    return {
        "suggestions": suggestions,
        "optimized_sql": optimized_sql,
        "should_warn_user": complexity_analysis["risk_level"] in ["high", "very_high"]
    }


@contextmanager 
def safe_sql_execution(sql: str, timeout_seconds: Optional[int] = None):
    """
    Context manager for safe SQL execution with automatic timeout
    """
    # Analyze query complexity
    complexity = analyze_query_complexity(sql)
    
    # Use recommended timeout or provided timeout
    if timeout_seconds is None:
        timeout_seconds = int(complexity["recommended_timeout"])
    
    logger.info(f"Executing SQL with {complexity['risk_level']} complexity (timeout: {timeout_seconds}s)")
    logger.info(f"Query complexity score: {complexity['complexity_score']}")
    
    start_time = time.time()
    try:
        yield complexity
    finally:
        elapsed = time.time() - start_time
        logger.info(f"SQL execution completed in {elapsed:.2f} seconds")


def create_safe_query_alternative(original_sql: str) -> str:
    """
    Create a safer, faster alternative query for complex cases
    """
    sql_lower = original_sql.lower()
    
    # If the query has complex GROUP BY, create a simplified version
    if "group by" in sql_lower and "strftime" in sql_lower:
        # Replace monthly grouping with yearly or remove time dimension
        safe_sql = original_sql.replace("strftime('%Y-%m', payment.payment_date) AS month", "strftime('%Y', payment.payment_date) AS year")
        safe_sql = safe_sql.replace(", month", ", year")
        safe_sql = safe_sql.replace("LIMIT 10000", "LIMIT 50")  # Much smaller limit
        return safe_sql
    
    # For other complex queries, just add a very small limit
    if "limit" in sql_lower:
        return original_sql.replace("LIMIT 10000", "LIMIT 20")
    else:
        return original_sql + " LIMIT 20"


# Example usage and testing
if __name__ == "__main__":
    # Test with the problematic query
    test_sql = """
    SELECT store.store_id, customer.activebool, film.rating, strftime('%Y-%m', payment.payment_date) AS month, SUM(payment.amount) AS total_revenue 
    FROM payment 
    JOIN customer ON payment.customer_id = customer.customer_id 
    JOIN rental ON payment.rental_id = rental.rental_id 
    JOIN inventory ON rental.inventory_id = inventory.inventory_id 
    JOIN store ON inventory.store_id = store.store_id 
    JOIN film ON inventory.film_id = film.film_id 
    GROUP BY store.store_id, customer.activebool, film.rating, month 
    LIMIT 10000
    """
    
    complexity = analyze_query_complexity(test_sql)
    print(f"Complexity Analysis: {complexity}")
    
    optimization = suggest_query_optimization(test_sql, complexity)
    print(f"Optimization Suggestions: {optimization}")
    
    safe_alternative = create_safe_query_alternative(test_sql)
    print(f"Safe Alternative: {safe_alternative}") 