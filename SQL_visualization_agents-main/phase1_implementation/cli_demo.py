#!/usr/bin/env python3
"""
CLI Demo for DVD Rental SQL Visualization System - Phase 1
Simple command-line interface for testing the LangGraph implementation
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# LangSmith tracing setup
def setup_langsmith_tracing(disable_tracing=False, project_override=None):
    """Setup LangSmith tracing if environment variables are available"""
    if disable_tracing:
        print("🔍 LangSmith tracing explicitly disabled")
        return False
    
    langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
    langsmith_project = project_override or os.environ.get("LANGSMITH_PROJECT", "dvd-rental-sql-viz")
    
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project
        
        print(f"🔍 LangSmith tracing enabled")
        print(f"   Project: {langsmith_project}")
        print(f"   Dashboard: https://smith.langchain.com/")
        return True
    else:
        print("ℹ️  LangSmith tracing disabled (LANGSMITH_API_KEY not set)")
        print("   To enable tracing, add LANGSMITH_API_KEY to your .env file")
        return False

# Note: LangSmith setup will be called after argument parsing

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from main_orchestrator import DVDRentalVisualizationSystem


def setup_environment():
    """Setup environment and check requirements"""
    required_env = ["OPENAI_API_KEY"]
    optional_env = ["LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
    missing_env = []
    
    for env_var in required_env:
        if not os.environ.get(env_var):
            missing_env.append(env_var)
    
    if missing_env:
        print("❌ Missing required environment variables:")
        for var in missing_env:
            print(f"   - {var}")
        print("\nPlease set these in your .env file or environment.")
        return False
    
    # Check optional LangSmith variables
    print("\n🔧 Optional LangSmith Configuration:")
    for env_var in optional_env:
        if os.environ.get(env_var):
            if env_var == "LANGSMITH_API_KEY":
                print(f"   ✅ {env_var} is set")
            else:
                print(f"   ✅ {env_var} = {os.environ.get(env_var)}")
        else:
            print(f"   ⚪ {env_var} not set (optional)")
    
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("\n💡 To enable LangSmith tracing:")
        print("   1. Sign up at https://smith.langchain.com/")
        print("   2. Get your API key from settings")
        print("   3. Add to .env file: LANGSMITH_API_KEY=your_key_here")
        print("   4. Optionally set: LANGSMITH_PROJECT=your_project_name")
    
    # Check if database file exists in any of the possible locations
    possible_db_paths = [
        current_dir.parent / "example-reference" / "dvdrental.sqlite",
        current_dir / "../example-reference/dvdrental.sqlite", 
        current_dir / "example-reference/dvdrental.sqlite",
        Path("example-reference/dvdrental.sqlite")
    ]
    
    db_found = False
    for db_path in possible_db_paths:
        if db_path.exists():
            db_found = True
            print(f"✅ Found database at: {db_path.resolve()}")
            break
    
    if not db_found:
        print("❌ Database file not found in any of these locations:")
        for path in possible_db_paths:
            print(f"   - {path}")
        print("\nPlease ensure the dvdrental.sqlite file exists.")
        print("You can:")
        print("1. Copy it from the example-reference directory")
        print("2. Set DATABASE_URL environment variable to point to your database")
        return False
    
    print("✅ Environment setup complete")
    return True


def run_interactive_mode():
    """Run interactive mode for testing queries"""
    print("\n🎬 DVD Rental SQL Visualization System - Interactive Mode")
    print("=" * 60)
    print("Enter natural language questions about the DVD rental database.")
    print("Type 'quit', 'exit', or 'q' to stop.\n")
    
    system = DVDRentalVisualizationSystem()
    
    # Show system status
    status = system.get_system_status()
    print(f"📊 System: {status['system_name']}")
    print(f"🔧 Version: {status['version']}")
    print(f"🤖 Agents: {len(status['agents'])}")
    print()
    
    while True:
        try:
            user_input = input("🔍 Enter your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print(f"\n⏳ Processing: {user_input}")
            print("-" * 40)
            
            # Process the request
            result = system.process_request(user_input)
            
            # Display results
            print(f"✅ Success: {result['success']}")
            print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f} seconds")
            print(f"📋 Stages: {' → '.join(result.get('stages_completed', []))}")
            
            if result['success']:
                print(f"\n📝 Generated SQL:")
                print(f"   {result.get('selected_sql', 'N/A')}")
                
                plot_spec = result.get('plot_specification', {})
                print(f"\n📊 Visualization:")
                print(f"   Type: {plot_spec.get('plot_type', 'N/A')}")
                print(f"   X-axis: {plot_spec.get('x_column', 'N/A')}")
                print(f"   Y-axis: {plot_spec.get('y_column', 'N/A')}")
                print(f"   Reasoning: {plot_spec.get('reasoning', 'N/A')}")
                
                print(f"\n📈 Data Quality: {result.get('data_quality_score', 0):.2f}")
                
                # Save plot HTML if available
                plot_html = result.get('plot_html', '')
                if plot_html:
                    output_file = current_dir / f"output_plot_{int(result.get('processing_time', 0) * 1000)}.html"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(plot_html)
                    print(f"💾 Plot saved to: {output_file}")
                
                # Display LangSmith trace link
                if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
                    project = os.environ.get("LANGCHAIN_PROJECT", "dvd-rental-sql-viz")
                    print(f"🔍 LangSmith trace: https://smith.langchain.com/projects/p/{project}")
                
            else:
                print(f"\n❌ Errors:")
                for error in result.get('errors', []):
                    print(f"   - {error}")
                
                # Display LangSmith trace link for debugging errors
                if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
                    project = os.environ.get("LANGCHAIN_PROJECT", "dvd-rental-sql-viz")
                    print(f"🔍 Debug trace: https://smith.langchain.com/projects/p/{project}")
            
            print("\n" + "=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Please try again with a different question.\n")


def run_batch_mode(queries_file: str):
    """Run batch mode with queries from a file"""
    queries_path = Path(queries_file)
    if not queries_path.exists():
        print(f"❌ Queries file not found: {queries_file}")
        return
    
    print(f"\n🎬 DVD Rental SQL Visualization System - Batch Mode")
    print(f"📁 Processing queries from: {queries_file}")
    print("=" * 60)
    
    system = DVDRentalVisualizationSystem()
    
    # Read queries from file
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"📋 Found {len(queries)} queries to process\n")
    
    results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Processing: {query}")
        print("-" * 40)
        
        result = system.process_request(query)
        results.append(result)
        
        print(f"✅ Success: {result['success']}")
        if result['success']:
            print(f"📊 Plot: {result.get('plot_specification', {}).get('plot_type', 'N/A')}")
        else:
            print(f"❌ Error: {result.get('errors', ['Unknown'])[0]}")
        
        print()
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print("📊 Batch Processing Summary:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(queries) - successful}")
    print(f"   Success rate: {successful/len(queries)*100:.1f}%")


def run_demo_queries():
    """Run predefined demo queries"""
    demo_queries = [
        "Show me the top 10 grossing film categories",
        "What are the most popular film ratings?", 
        "Compare rental counts by store",
        "Show customer rental frequency distribution",
        "Which actors appear in the most films?",
        "What is the average rental duration by category?"
    ]
    
    print(f"\n🎬 DVD Rental SQL Visualization System - Demo Mode")
    print("🎭 Running predefined demo queries")
    print("=" * 60)
    
    system = DVDRentalVisualizationSystem()
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n[{i}/{len(demo_queries)}] Demo Query: {query}")
        print("-" * 50)
        
        result = system.process_request(query)
        
        print(f"✅ Success: {result['success']}")
        print(f"⏱️  Time: {result.get('processing_time', 0):.2f}s")
        
        if result['success']:
            plot_spec = result.get('plot_specification', {})
            print(f"📊 Plot: {plot_spec.get('plot_type', 'N/A')} - {plot_spec.get('reasoning', 'N/A')}")
            
            # Display LangSmith trace link if available
            if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
                project = os.environ.get("LANGCHAIN_PROJECT", "dvd-rental-sql-viz")
                print(f"🔍 Trace: https://smith.langchain.com/projects/p/{project}")
        else:
            print(f"❌ Error: {result.get('errors', ['Unknown'])[0]}")
            
            # Display LangSmith trace link even for errors (helpful for debugging)
            if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
                project = os.environ.get("LANGCHAIN_PROJECT", "dvd-rental-sql-viz")
                print(f"🔍 Debug trace: https://smith.langchain.com/projects/p/{project}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DVD Rental SQL Visualization System - Phase 1 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_demo.py                          # Interactive mode
  python cli_demo.py --demo                   # Run demo queries
  python cli_demo.py --batch queries.txt     # Process queries from file
  python cli_demo.py --query "Show top films" # Single query
  
LangSmith Tracing Examples:
  python cli_demo.py --demo                   # Use default project name
  python cli_demo.py --demo --langsmith-project my-project  # Custom project
  python cli_demo.py --demo --no-tracing     # Disable tracing
  
To enable LangSmith tracing:
  1. Set LANGSMITH_API_KEY in your .env file
  2. Optionally set LANGSMITH_PROJECT (default: dvd-rental-sql-viz)
  3. View traces at https://smith.langchain.com/
        """
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run predefined demo queries'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        metavar='FILE',
        help='Run queries from a file (one per line)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        metavar='QUESTION',
        help='Run a single query'
    )
    
    parser.add_argument(
        '--check-env',
        action='store_true',
        help='Check environment setup and exit'
    )
    
    parser.add_argument(
        '--no-tracing',
        action='store_true',
        help='Disable LangSmith tracing even if API key is available'
    )
    
    parser.add_argument(
        '--langsmith-project',
        type=str,
        metavar='PROJECT_NAME',
        help='Override LangSmith project name'
    )
    
    args = parser.parse_args()
    
    # Setup LangSmith tracing based on arguments
    setup_langsmith_tracing(
        disable_tracing=args.no_tracing,
        project_override=args.langsmith_project
    )
    
    # Check environment setup
    if not setup_environment():
        sys.exit(1)
    
    if args.check_env:
        print("✅ Environment check passed")
        return
    
    # Route to appropriate mode
    if args.demo:
        run_demo_queries()
    elif args.batch:
        run_batch_mode(args.batch)
    elif args.query:
        print(f"\n🎬 DVD Rental SQL Visualization System - Single Query")
        print("=" * 60)
        system = DVDRentalVisualizationSystem()
        result = system.process_request(args.query)
        
        print(f"Query: {args.query}")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"SQL: {result.get('selected_sql', 'N/A')}")
            print(f"Plot: {result.get('plot_specification', {}).get('plot_type', 'N/A')}")
        else:
            print(f"Errors: {result.get('errors', [])}")
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()
