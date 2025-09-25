#!/usr/bin/env python3
"""
Setup script for DVD Rental SQL Visualization System - Phase 1
Helps users verify environment and database setup
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("⚠️  This system requires Python 3.8 or higher")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Compatible")
        return True


def check_required_packages():
    """Check if required packages are installed"""
    print("\n📦 Checking required packages...")
    required_packages = [
        "langgraph",
        "langchain",
        "langchain_openai", 
        "pandas",
        "plotly",
        "sqlalchemy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\n📋 Or install all requirements:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def check_environment_variables():
    """Check environment variables"""
    print("\n🔑 Checking environment variables...")
    
    # Check for .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"✅ Found .env file: {env_file}")
    else:
        print("⚠️  No .env file found (but environment variables might still be set)")
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OPENAI_API_KEY is set (ends with ...{openai_key[-4:]})")
    else:
        print("❌ OPENAI_API_KEY not found")
        print("💡 Set your OpenAI API key:")
        print("   1. Create a .env file: echo 'OPENAI_API_KEY=your_key_here' > .env")
        print("   2. Or set environment variable: export OPENAI_API_KEY=your_key_here")
        
        # Check if .env exists but key is missing
        if env_file.exists():
            print("   📁 You have a .env file but OPENAI_API_KEY might be missing or incorrectly formatted")
            print("   📝 Make sure your .env file contains: OPENAI_API_KEY=sk-...")
        
        return False
    
    return True


def find_database():
    """Find the database file"""
    print("\n🗄️ Looking for database file...")
    
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir.parent / "example-reference" / "dvdrental.sqlite",
        current_dir / "../example-reference/dvdrental.sqlite",
        current_dir / "example-reference/dvdrental.sqlite",
        Path("example-reference/dvdrental.sqlite"),
        Path("dvdrental.sqlite")
    ]
    
    for path in possible_paths:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ Found database: {path.resolve()} ({size_mb:.1f} MB)")
            return True, str(path.resolve())
    
    print("❌ Database file (dvdrental.sqlite) not found")
    print("📍 Searched in these locations:")
    for path in possible_paths:
        print(f"   - {path}")
    
    print("\n💡 To fix this:")
    print("1. Copy dvdrental.sqlite from the example-reference directory")
    print("2. Or set DATABASE_URL environment variable")
    print("3. Or run the data loading script to create the database")
    
    return False, None


def test_imports():
    """Test importing the main system"""
    print("\n🧪 Testing system imports...")
    
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        from main_orchestrator import DVDRentalVisualizationSystem
        print("✅ Main system imports successfully")
        
        # Try to create system instance
        system = DVDRentalVisualizationSystem()
        status = system.get_system_status()
        print(f"✅ System initialized: {status['system_name']}")
        print(f"📊 Available agents: {len(status['agents'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False


def run_quick_test():
    """Run a quick test of the system"""
    print("\n🚀 Running quick system test...")
    
    try:
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        from main_orchestrator import DVDRentalVisualizationSystem
        
        system = DVDRentalVisualizationSystem()
        
        # Try a simple query
        test_query = "How many tables are in the database?"
        print(f"📝 Test query: {test_query}")
        
        result = system.process_request(test_query)
        
        if result['success']:
            print("✅ Quick test passed!")
            print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
            print(f"📋 Stages completed: {' → '.join(result['stages_completed'])}")
            return True
        else:
            print("❌ Quick test failed")
            if result['errors']:
                print(f"🔍 Error: {result['errors'][0]}")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed with exception: {str(e)}")
        return False


def main():
    """Main setup verification"""
    print("🎬 DVD Rental SQL Visualization System - Setup Verification")
    print("=" * 65)
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    # Check packages
    if not check_required_packages():
        all_good = False
    
    # Check environment
    if not check_environment_variables():
        all_good = False
    
    # Find database
    db_found, db_path = find_database()
    if not db_found:
        all_good = False
    
    # Test imports
    if all_good and not test_imports():
        all_good = False
    
    # Run quick test if everything else passed
    if all_good:
        if not run_quick_test():
            all_good = False
    
    # Final status
    print("\n" + "=" * 65)
    if all_good:
        print("🎉 Setup verification PASSED!")
        print("🚀 You're ready to use the system!")
        print("\n📚 Next steps:")
        print("   python cli_demo.py --demo")
        print("   python cli_demo.py")
        print("   python example_usage.py")
    else:
        print("❌ Setup verification FAILED")
        print("🔧 Please fix the issues above and run this script again")
        print("\n📚 For help:")
        print("   - Check the README.md file")
        print("   - Ensure all requirements are installed")
        print("   - Verify your OpenAI API key is set")
        print("   - Make sure the database file exists")
    
    return all_good


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
