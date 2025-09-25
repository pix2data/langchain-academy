"""
Quick test to reproduce and fix the column mismatch issue
"""

from main_orchestrator import DVDRentalVisualizationSystem
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

def quick_test():
    """Test the column mismatch fix"""
    print("🧪 Quick Test: Column Mismatch Fix")
    print("=" * 50)
    
    # The query that was causing column mismatch
    query = "Compare revenue by customer segment, and film rating with monthly trends"
    
    print(f"📝 Query: {query}")
    print("🔄 Processing...")
    
    system = DVDRentalVisualizationSystem()
    result = system.process_request(query)
    
    print(f"\n✅ Success: {result['success']}")
    
    if result['success']:
        print(f"📊 Plot type: {result.get('plot_specification', {}).get('plot_type', 'unknown')}")
        print(f"📈 Data quality: {result.get('data_quality_score', 0):.2f}")
        
        # Check the plot specification details
        plot_spec = result.get('plot_specification', {})
        print(f"\n🎯 Plot Configuration:")
        print(f"  X-axis: {plot_spec.get('x_column', 'unknown')}")
        print(f"  Y-axis: {plot_spec.get('y_column', 'unknown')}")
        print(f"  Color: {plot_spec.get('color_column', 'none')}")
        
        # Check actual data columns
        sql = result.get('selected_sql', '')
        print(f"\n📋 Generated SQL:")
        print(f"  {sql[:200]}...")
    else:
        print(f"❌ Errors: {result.get('errors', [])}")
    
    return result

if __name__ == "__main__":
    quick_test() 