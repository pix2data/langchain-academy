"""
Test Script for Complex Query Timeout Protection
Demonstrates how the system now handles the problematic query that was freezing
"""

import time
import logging
from main_orchestrator import DVDRentalVisualizationSystem

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_complex_query():
    """Test the complex query that was causing freezing"""
    
    print("🧪 Testing Complex Query Timeout Protection")
    print("=" * 60)
    
    # The problematic query that was freezing the system
    complex_query = "Compare revenue by store, customer segment, and film rating with monthly trends"
    
    print(f"📝 Query: {complex_query}")
    print("⏳ Processing with timeout protection...")
    print()
    
    system = DVDRentalVisualizationSystem()
    
    start_time = time.time()
    try:
        result = system.process_request(complex_query)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Completed in {elapsed_time:.2f} seconds")
        print(f"🎯 Success: {result['success']}")
        
        if result['success']:
            print(f"📊 Generated SQL: {result['selected_sql'][:200]}...")
            print(f"📈 Data quality: {result.get('data_quality_score', 0):.2f}")
            print(f"📋 Plot type: {result.get('plot_specification', {}).get('plot_type', 'unknown')}")
            
            # Check for warnings about complexity
            if 'warnings' in result.get('_query_results_dict', {}):
                print(f"⚠️ Warnings: {result['_query_results_dict']['warnings']}")
        else:
            print(f"❌ Errors: {result.get('errors', [])}")
            print(f"🔧 Current stage: {result.get('current_stage', 'unknown')}")
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Exception after {elapsed_time:.2f} seconds: {e}")
    
    print()
    print("🔍 Key Improvements:")
    print("• SQL complexity analysis before execution")
    print("• Automatic query optimization for very complex queries")
    print("• 2-minute timeout protection on SQL execution")
    print("• Fallback to simpler alternatives when timeouts occur")
    print("• Emergency queries when everything else fails")


def test_simple_query():
    """Test a simple query to ensure normal functionality still works"""
    
    print("\n🧪 Testing Simple Query (Control)")
    print("=" * 60)
    
    simple_query = "Show me the top 5 film categories"
    
    print(f"📝 Query: {simple_query}")
    print("⏳ Processing...")
    
    system = DVDRentalVisualizationSystem()
    
    start_time = time.time()
    result = system.process_request(simple_query)
    elapsed_time = time.time() - start_time
    
    print(f"✅ Completed in {elapsed_time:.2f} seconds")
    print(f"🎯 Success: {result['success']}")
    
    if result['success']:
        print(f"📊 Generated SQL: {result['selected_sql']}")
        print(f"📈 Data quality: {result.get('data_quality_score', 0):.2f}")


def demonstrate_complexity_analysis():
    """Show how the complexity analyzer works"""
    
    print("\n🔬 SQL Complexity Analysis Demo")
    print("=" * 60)
    
    from sql_timeout_handler import analyze_query_complexity, suggest_query_optimization
    
    queries = [
        ("Simple", "SELECT name FROM category LIMIT 10"),
        ("Medium", "SELECT c.name, COUNT(*) FROM category c JOIN film_category fc ON c.category_id = fc.category_id GROUP BY c.name"),
        ("Complex", """SELECT store.store_id, customer.activebool, film.rating, strftime('%Y-%m', payment.payment_date) AS month, SUM(payment.amount) AS total_revenue 
                      FROM payment 
                      JOIN customer ON payment.customer_id = customer.customer_id 
                      JOIN rental ON payment.rental_id = rental.rental_id 
                      JOIN inventory ON rental.inventory_id = inventory.inventory_id 
                      JOIN store ON inventory.store_id = store.store_id 
                      JOIN film ON inventory.film_id = film.film_id 
                      GROUP BY store.store_id, customer.activebool, film.rating, month 
                      LIMIT 10000""")
    ]
    
    for name, sql in queries:
        print(f"\n📊 {name} Query Analysis:")
        complexity = analyze_query_complexity(sql)
        print(f"  Risk Level: {complexity['risk_level']}")
        print(f"  Complexity Score: {complexity['complexity_score']}")
        print(f"  Table Count: {complexity['table_count']}")
        print(f"  JOIN Count: {complexity['join_count']}")
        print(f"  GROUP BY Complexity: {complexity['group_by_complexity']}")
        print(f"  Recommended Timeout: {complexity['recommended_timeout']:.1f}s")
        
        if complexity['risk_level'] in ['high', 'very_high']:
            optimization = suggest_query_optimization(sql, complexity)
            print(f"  ⚠️ Optimization Needed:")
            for suggestion in optimization['suggestions'][:3]:  # Show first 3 suggestions
                print(f"    • {suggestion}")


if __name__ == "__main__":
    # Run all tests
    demonstrate_complexity_analysis()
    test_simple_query()
    test_complex_query()
    
    print("\n🎉 Testing Complete!")
    print("The system should now handle complex queries without freezing.") 