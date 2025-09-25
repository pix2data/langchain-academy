#!/usr/bin/env python3
"""
Example Usage of DVD Rental SQL Visualization System - Phase 1
Demonstrates programmatic usage of the LangGraph implementation
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the main system
from main_orchestrator import DVDRentalVisualizationSystem


def basic_example():
    """Basic example of using the system"""
    print("🎬 Basic Example: DVD Rental SQL Visualization System")
    print("=" * 60)
    
    # Create system instance
    system = DVDRentalVisualizationSystem()
    
    # Show system capabilities
    status = system.get_system_status()
    print(f"System: {status['system_name']}")
    print(f"Version: {status['version']}")
    print(f"Supported visualizations: {', '.join(status['supported_visualizations'])}")
    print()
    
    # Process a simple query
    query = "Show me the top 5 film categories by revenue"
    print(f"Processing query: {query}")
    print("-" * 40)
    
    result = system.process_request(query)
    
    # Display results
    if result['success']:
        print("✅ Query processed successfully!")
        print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
        print(f"📋 Stages completed: {' → '.join(result['stages_completed'])}")
        print()
        
        # SQL Information
        print("📝 Generated SQL Query:")
        print(f"   {result['selected_sql']}")
        print()
        
        # Visualization Information
        plot_spec = result['plot_specification']
        print("📊 Visualization Details:")
        print(f"   Type: {plot_spec.get('plot_type', 'N/A')}")
        print(f"   X-axis: {plot_spec.get('x_column', 'N/A')}")
        print(f"   Y-axis: {plot_spec.get('y_column', 'N/A')}")
        print(f"   Reasoning: {plot_spec.get('reasoning', 'N/A')}")
        print()
        
        # Data Quality
        print(f"📈 Data Quality Score: {result['data_quality_score']:.2f}/1.0")
        
        # Save visualization
        if result['plot_html']:
            output_file = Path(__file__).parent / "example_output.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['plot_html'])
            print(f"💾 Visualization saved to: {output_file}")
        
    else:
        print("❌ Query processing failed!")
        for error in result['errors']:
            print(f"   Error: {error}")


def multiple_queries_example():
    """Example of processing multiple queries"""
    print("\n🎬 Multiple Queries Example")
    print("=" * 60)
    
    system = DVDRentalVisualizationSystem()
    
    # Define multiple test queries
    queries = [
        "What are the most popular film ratings?",
        "Compare rental counts between stores",
        "Show the top 10 actors by number of films"
    ]
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing: {query}")
        print("-" * 40)
        
        result = system.process_request(query)
        results.append(result)
        
        if result['success']:
            plot_type = result['plot_specification'].get('plot_type', 'unknown')
            print(f"✅ Success - Created {plot_type} visualization")
            print(f"⏱️  Time: {result['processing_time']:.2f}s")
        else:
            print(f"❌ Failed: {result['errors'][0] if result['errors'] else 'Unknown error'}")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n📊 Summary:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Successful: {successful}")
    print(f"   Success rate: {successful/len(queries)*100:.1f}%")
    
    return results


def error_handling_example():
    """Example of error handling"""
    print("\n🎬 Error Handling Example")
    print("=" * 60)
    
    system = DVDRentalVisualizationSystem()
    
    # Try an intentionally problematic query
    problematic_queries = [
        "",  # Empty query
        "Delete all records",  # Dangerous query (should be caught)
        "Show me information about unicorns",  # Query about non-existent data
    ]
    
    for query in problematic_queries:
        print(f"\nTesting query: '{query}'")
        print("-" * 30)
        
        result = system.process_request(query)
        
        if result['success']:
            print("✅ Unexpectedly succeeded")
        else:
            print(f"❌ Failed as expected: {result['errors'][0] if result['errors'] else 'Unknown error'}")
            print(f"📍 Failed at stage: {result.get('current_stage', 'unknown')}")


def custom_database_example():
    """Example with custom database URL"""
    print("\n🎬 Custom Database Example")
    print("=" * 60)
    
    system = DVDRentalVisualizationSystem()
    
    # Use custom database URL (in this case, same database but explicitly specified)
    custom_db_url = "sqlite:///SQL_visualization_agents/example-reference/dvdrental.sqlite"
    
    query = "How many films are in the database?"
    print(f"Processing with custom DB: {query}")
    
    result = system.process_request(query, database_url=custom_db_url)
    
    if result['success']:
        print("✅ Custom database query successful")
        print(f"📝 SQL: {result['selected_sql']}")
    else:
        print(f"❌ Custom database query failed: {result['errors']}")


def main():
    """Run all examples"""
    # Check if environment is set up
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return
    
    # Check if database exists in any of the possible locations
    possible_db_paths = [
        Path(__file__).parent.parent / "example-reference" / "dvdrental.sqlite",
        Path(__file__).parent / "../example-reference/dvdrental.sqlite",
        Path("example-reference/dvdrental.sqlite")
    ]
    
    db_found = any(path.exists() for path in possible_db_paths)
    if not db_found:
        print("❌ Database file not found in any expected location")
        print("Expected locations:")
        for path in possible_db_paths:
            print(f"   - {path}")
        print("Please ensure the dvdrental.sqlite file exists or set DATABASE_URL")
        return
    
    try:
        # Run examples
        basic_example()
        multiple_queries_example()
        error_handling_example()
        custom_database_example()
        
        print("\n🎉 All examples completed successfully!")
        print("=" * 60)
        print("Next steps:")
        print("1. Try the CLI demo: python cli_demo.py")
        print("2. Experiment with your own queries")
        print("3. Explore the generated HTML visualizations")
        
    except Exception as e:
        print(f"\n💥 Example execution failed: {str(e)}")
        print("Please check your environment setup and database availability")


if __name__ == "__main__":
    main()
