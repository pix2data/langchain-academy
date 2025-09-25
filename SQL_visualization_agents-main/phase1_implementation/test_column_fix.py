"""
Test Script for Column Name Fixing
Tests the visualization agent's ability to handle column mismatches
"""

import pandas as pd
from agents.visualization_agent import _validate_and_fix_columns, _create_intelligent_figure

def test_column_fixing():
    """Test the column name fixing functionality"""
    
    print("🧪 Testing Column Name Fixing")
    print("=" * 50)
    
    # Create test DataFrame with the actual problematic column names
    test_data = {
        'first_name': ['John', 'Jane', 'Bob'],
        'last_name': ['Doe', 'Smith', 'Johnson'],
        'rating': ['PG', 'R', 'PG-13'],
        'month': ['2023-01', '2023-02', '2023-03'],
        'total_revenue': [25.50, 18.75, 32.00]
    }
    df = pd.DataFrame(test_data)
    
    print("📊 Actual DataFrame columns:")
    print(f"  {list(df.columns)}")
    print()
    
    # Test problematic plot specification (what the LLM expected)
    problematic_spec = {
        "plot_type": "bar",
        "x_column": "customer_segment",  # This doesn't exist!
        "y_column": "total_revenue",     # This exists
        "color_column": "film_rating",   # This doesn't exist!
        "reasoning": "Test specification"
    }
    
    print("❌ Problematic specification:")
    print(f"  x_column: '{problematic_spec['x_column']}'")
    print(f"  y_column: '{problematic_spec['y_column']}'")
    print(f"  color_column: '{problematic_spec['color_column']}'")
    print()
    
    # Test the fixing function
    fixed_spec = _validate_and_fix_columns(df, problematic_spec)
    
    print("✅ Fixed specification:")
    print(f"  x_column: '{fixed_spec['x_column']}'")
    print(f"  y_column: '{fixed_spec['y_column']}'")
    print(f"  color_column: '{fixed_spec['color_column']}'")
    print()
    
    # Test creating a figure with the fixed specification
    try:
        fig = _create_intelligent_figure(df, problematic_spec)
        print("✅ Figure creation: SUCCESS")
        print(f"  Figure has {len(fig.data)} trace(s)")
    except Exception as e:
        print(f"❌ Figure creation failed: {e}")
    
    print()
    print("🔍 Column Mapping Test Results:")
    test_mappings = [
        ("customer_segment", "categorical"),
        ("film_rating", "categorical"),
        ("revenue", "numeric"),
        ("month", "categorical")
    ]
    
    for desired, col_type in test_mappings:
        # Test the internal function
        from agents.visualization_agent import _validate_and_fix_columns
        test_spec = {"x_column": desired}
        fixed = _validate_and_fix_columns(df, test_spec)
        mapped = fixed.get("x_column")
        print(f"  '{desired}' → '{mapped}'")


if __name__ == "__main__":
    test_column_fixing() 