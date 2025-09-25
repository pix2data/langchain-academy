# DVD Rental SQL Visualization System - Phase 1 Implementation

This is the Phase 1 implementation of the DVD Rental SQL Visualization System using LangGraph orchestration. It converts the existing standalone agents into a coordinated multi-agent system with state management, error handling, and human-in-the-loop capabilities.

## 🏗️ Architecture Overview

The system consists of 4 coordinated agents orchestrated by LangGraph:

1. **Database Connection & Schema Agent** - Validates connections and discovers database schema
2. **Natural Language to SQL Agent** - Converts user questions to safe SQL queries  
3. **Data Execution & Analysis Agent** - Executes SQL and analyzes results
4. **Intelligent Visualization Agent** - Creates smart visualizations based on data characteristics

## 📁 Project Structure

```
phase1_implementation/
├── __init__.py                 # Package initialization
├── state_schemas.py            # LangGraph state definitions
├── main_orchestrator.py        # Main system orchestrator
├── cli_demo.py                 # Command-line demo interface
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── test_queries.txt            # Sample queries for testing
└── agents/                     # Individual agent implementations
    ├── __init__.py
    ├── database_agent.py        # Database & schema discovery
    ├── nl2sql_agent.py          # Natural language to SQL
    ├── data_analysis_agent.py   # Data execution & analysis
    └── visualization_agent.py   # Visualization creation
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **OpenAI API Key** - Set in environment or `.env` file
3. **DVD Rental Database** - SQLite file should be available

### Installation

1. **Install dependencies:**
   ```bash
   cd phase1_implementation
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   # Create .env file or set environment variables
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

3. **Verify database access:**
   ```bash
   python cli_demo.py --check-env
   ```

### Running the System

#### Interactive Mode (Recommended)
```bash
python cli_demo.py
```
Enter natural language questions about the DVD rental database.

#### Demo Mode
```bash
python cli_demo.py --demo
```
Runs predefined demo queries to showcase system capabilities.

#### Single Query
```bash
python cli_demo.py --query "Show me the top grossing film categories"
```

#### Batch Processing
```bash
python cli_demo.py --batch test_queries.txt
```

## 📊 Example Queries

The system can handle various types of questions about the DVD rental database:

- **Revenue Analysis**: "Show me the top grossing film categories"
- **Customer Analysis**: "What is the rental frequency by customer?"
- **Inventory Analysis**: "Compare rental counts by store"
- **Film Analysis**: "Which actors appear in the most films?"
- **Trend Analysis**: "Show rental patterns over time"

## 🔧 Technical Details

### LangGraph Integration

- **State Management**: Centralized state with type-safe schemas
- **Sub-graph Isolation**: Each agent runs in its own sub-graph
- **Error Handling**: Comprehensive error recovery at each stage
- **Memory Persistence**: MemorySaver for conversation continuity

### Key Features

1. **Safe SQL Generation**: Only SELECT queries, automatic LIMIT enforcement
2. **FK-Aware Joins**: Uses discovered foreign key relationships
3. **Multi-Candidate SQL**: Generates and ranks multiple query options
4. **LLM-Driven Visualization**: Intelligent chart type selection
5. **Data Quality Assessment**: Automatic scoring of query results

### Processing Pipeline

```
User Request → Database Discovery → SQL Generation → Data Execution → Visualization
     ↓              ↓                    ↓              ↓              ↓
   Validate      Schema &           Safe SQL         Execute       Smart Plot
 Environment   Relationships      Generation        & Analyze      Creation
```

## 🎯 Phase 1 Scope

### ✅ Implemented
- [x] LangGraph orchestration of all 4 agents
- [x] State management with typed schemas
- [x] Basic error handling and recovery
- [x] Sub-graph communication patterns
- [x] CLI demo interface
- [x] Preserved all existing functionality

### 🔄 Phase 1 Limitations
- Human-in-the-loop breakpoints disabled (will be enabled in Phase 2)
- Basic error recovery (advanced strategies in Phase 3)
- Single database support (SQLite only)
- Limited visualization feedback mechanisms

## 🧪 Testing

### Manual Testing
```bash
# Test environment setup
python cli_demo.py --check-env

# Test basic functionality
python cli_demo.py --demo

# Test specific scenarios
python cli_demo.py --query "Your test query here"
```

### Automated Testing
```python
from main_orchestrator import DVDRentalVisualizationSystem

system = DVDRentalVisualizationSystem()
result = system.process_request("Show top film categories")
assert result['success'] == True
```

## 📈 Performance Expectations

- **Typical Response Time**: 10-30 seconds for complex queries
- **Database Schema Discovery**: ~2-5 seconds (cached after first run)
- **SQL Generation**: ~3-8 seconds (depends on query complexity)
- **Data Execution**: ~1-5 seconds (depends on data size)
- **Visualization**: ~2-5 seconds (depends on chart complexity)

## 🔍 Debugging and Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```
   Error: No OpenAI API key found
   Solution: Set OPENAI_API_KEY in environment or .env file
   ```

2. **Database Not Found**
   ```
   Error: Database file not found
   Solution: Ensure dvdrental.sqlite exists in example-reference/
   ```

3. **SQL Generation Fails**
   ```
   Error: No valid SQL generated
   Solution: Try rephrasing query or check schema availability
   ```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔧 Development

### Adding New Queries
Add test queries to `test_queries.txt`:
```
# Customer analysis queries
Show customer rental frequency
Which customers rent the most movies?

# Revenue queries  
What are the top grossing film categories?
```

### Extending Functionality
```python
# Example: Add custom processing step
def custom_processing(state: DVDRentalSystemState) -> Dict[str, Any]:
    # Your custom logic here
    return {"custom_field": "custom_value"}

# Add to main orchestrator
builder.add_node("custom_processing", custom_processing)
```

## 📝 State Schema Reference

### Main System State
```python
class DVDRentalSystemState(TypedDict):
    # Input
    user_request: str
    database_url: str
    
    # Database Agent Output
    schema: Dict[str, Any]
    fk_hints: List[Dict[str, str]]
    roles: Dict[str, str]
    
    # SQL Agent Output
    sql_candidates: List[str]
    selected_sql: str
    
    # Data Agent Output
    query_results: pd.DataFrame
    data_characteristics: Dict[str, Any]
    
    # Visualization Agent Output
    plot_specification: Dict[str, Any]
    generated_plot: go.Figure
    plot_html: str
    
    # Control Flow
    current_stage: str
    errors: List[str]
    stages_completed: List[str]
```

## 🚦 Next Steps (Phase 2+)

### Phase 2: Human-in-the-Loop
- Enable SQL query approval breakpoints
- Add visualization feedback mechanisms
- Implement state editing capabilities

### Phase 3: Advanced Features
- Multi-candidate SQL execution with ranking
- Advanced error recovery strategies
- Performance optimization and caching

### Phase 4: Production Ready
- Web interface development
- Comprehensive testing suite
- Deployment and monitoring

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example queries and expected outputs
3. Test with the demo mode to verify system functionality
4. Check environment setup with `--check-env`

## 📜 License

This implementation builds upon the existing DVD rental database agents and integrates them with LangGraph for orchestration and state management.
