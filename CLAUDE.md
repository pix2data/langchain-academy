# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **LangChain Academy** - a comprehensive learning resource for building agent and multi-agent applications using LangGraph. The repository is structured as educational modules (0-6) that progressively introduce LangGraph concepts, from basic setup to advanced agent workflows.

## Environment Setup

### Python Environment
- **Python 3.11 or later required** for optimal LangGraph compatibility
- Create virtual environment: `python3 -m venv lc-academy-env`
- Activate environment:
  - Mac/Linux/WSL: `source lc-academy-env/bin/activate`
  - Windows PowerShell: `lc-academy-env\scripts\activate`
- Install dependencies: `pip install -r requirements.txt`

### Required API Keys
Set these environment variables or use a `.env` file:
- `OPENAI_API_KEY` - OpenAI API access (required for most examples)
- `LANGSMITH_API_KEY` - LangSmith tracing (optional but recommended)
- `LANGSMITH_TRACING_V2=true` - Enable LangSmith tracing
- `LANGSMITH_PROJECT="langchain-academy"` - Set project name
- `TAVILY_API_KEY` - Web search functionality (required for Module 4)

### Development Commands

#### Running Jupyter Notebooks
```bash
jupyter notebook
```

#### LangGraph Studio Development
Navigate to any `module-x/studio/` directory and run:
```bash
langgraph dev
```
This starts:
- API server: http://127.0.0.1:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API docs: http://127.0.0.1:2024/docs

#### Studio Environment Setup
For modules 1-5, create `.env` files in studio directories:
```bash
for i in {1..5}; do
  cp module-$i/studio/.env.example module-$i/studio/.env
  echo "OPENAI_API_KEY=\"$OPENAI_API_KEY\"" > module-$i/studio/.env
done
echo "TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> module-4/studio/.env
```

## Repository Architecture

### Module Structure
- **module-0**: Basic setup and LangChain/LangGraph fundamentals
- **module-1**: Core LangGraph concepts (chains, agents, routers, memory, deployment)
- **module-2**: State management (schemas, reducers, message handling)
- **module-3**: Human-in-the-loop workflows (breakpoints, interruptions, time travel)
- **module-4**: Advanced patterns (parallelization, sub-graphs, map-reduce)
- **module-5**: Memory systems (stores, schemas, collections, profiles)
- **module-6**: Production deployment and assistant creation

### File Organization
Each module contains:
- **Jupyter notebooks** (`.ipynb`): Interactive tutorials with code and explanations
- **studio/ directory**: LangGraph applications for Studio IDE
  - `langgraph.json`: Studio configuration defining graph entry points
  - `.env.example`: Template for environment variables
  - Python files: Graph implementations referenced in langgraph.json

### Key Dependencies
Core packages from `requirements.txt`:
- `langgraph` - Main graph framework
- `langgraph-prebuilt` - Pre-built components
- `langgraph-sdk` - SDK for graph interaction
- `langgraph-checkpoint-sqlite` - Persistence layer
- `langsmith` - Observability and tracing
- `langchain-*` - LangChain ecosystem packages
- `tavily-python` - Web search integration

## Development Patterns

### LangGraph Applications
LangGraph apps use graph-based architecture with:
- **State**: TypedDict defining shared state structure
- **Nodes**: Functions that process state and return updates
- **Edges**: Connections between nodes (regular or conditional)
- **StateGraph**: Main graph builder class

Example structure:
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list

def node_function(state):
    return {"messages": [...]}

builder = StateGraph(State)
builder.add_node("node_name", node_function)
builder.add_edge(START, "node_name")
graph = builder.compile()
```

### Studio Configuration
`langgraph.json` files define graph entry points for Studio:
```json
{
  "graphs": {
    "graph_name": "./file.py:graph_variable"
  },
  "env": "./.env",
  "python_version": "3.11"
}
```

### DevContainer Support
Repository includes `.devcontainer/devcontainer.json` for containerized development with automatic dependency installation.

## Working with This Repository

When working with notebooks, ensure the virtual environment is activated and all API keys are configured. For Studio development, navigate to the specific module's studio directory before running `langgraph dev`. Each module builds upon previous concepts, so follow the numerical order for learning.