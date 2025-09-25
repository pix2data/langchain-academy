# LangSmith Integration Guide

## Overview
LangSmith integration has been added to the DVD Rental SQL Visualization System to provide comprehensive tracing and observability of the LangGraph agent execution.

## What LangSmith Provides
- **Execution Traces**: Visual representation of agent workflows
- **Performance Monitoring**: Timing and latency metrics for each step
- **Error Debugging**: Detailed error traces and stack traces
- **Agent Interactions**: See how agents communicate and pass data
- **LLM Call Monitoring**: Track OpenAI API calls, tokens, and costs
- **Data Flow Visualization**: Understand how data moves between agents

## Setup Instructions

### 1. Get LangSmith API Key
1. Sign up at [https://smith.langchain.com/](https://smith.langchain.com/)
2. Navigate to Settings → API Keys
3. Create a new API key
4. Copy the key (starts with `ls_`)

### 2. Configure Environment Variables
Add to your `.env` file:
```bash
# Required for LangSmith tracing
LANGSMITH_API_KEY=ls_your_actual_api_key_here

# Optional: Custom project name (default: dvd-rental-sql-viz)
LANGSMITH_PROJECT=my-custom-project-name
```

### 3. Install LangSmith Package
```bash
pip install langsmith
# or
pip install -r requirements.txt  # (langsmith is now included)
```

## Usage Examples

### Basic Usage (Auto-detect LangSmith)
```bash
# LangSmith will be enabled automatically if LANGSMITH_API_KEY is set
python cli_demo.py --demo
```

### Custom Project Name
```bash
python cli_demo.py --demo --langsmith-project "my-analysis-session"
```

### Disable Tracing
```bash
python cli_demo.py --demo --no-tracing
```

### Interactive Mode with Tracing
```bash
python cli_demo.py
# Enter queries and get trace links after each execution
```

## What You'll See

### Console Output
When LangSmith is enabled, you'll see:
```
🔍 LangSmith tracing enabled
   Project: dvd-rental-sql-viz
   Dashboard: https://smith.langchain.com/

[After each query]
🔍 Trace: https://smith.langchain.com/projects/p/dvd-rental-sql-viz
```

### LangSmith Dashboard
Navigate to the provided URL to see:

1. **Trace Timeline**: Visual flow of agent execution
2. **Agent Details**: Each agent's inputs, outputs, and processing time
3. **LLM Calls**: OpenAI API calls with prompts and responses
4. **Error Analysis**: Detailed error information when things go wrong
5. **Performance Metrics**: Latency, token usage, costs

## Trace Structure
You'll see traces for each of the 4 main agents:
1. **Database Agent**: Schema discovery and connection validation
2. **NL2SQL Agent**: Natural language to SQL conversion
3. **Data Analysis Agent**: SQL execution and data analysis
4. **Visualization Agent**: Chart generation and reasoning

## Benefits for Development

### Debugging
- See exactly where errors occur in the agent pipeline
- Understand why certain SQL queries are generated
- Track data flow between agents

### Performance Optimization
- Identify bottlenecks in agent execution
- Monitor OpenAI API usage and costs
- Optimize prompt engineering based on actual LLM interactions

### Understanding Agent Behavior
- See how agents interpret user queries
- Understand the reasoning behind visualization choices
- Track how schema information influences SQL generation

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LANGSMITH_API_KEY` | Yes* | - | Your LangSmith API key |
| `LANGSMITH_PROJECT` | No | `dvd-rental-sql-viz` | Project name in LangSmith |
| `LANGCHAIN_TRACING_V2` | No | `true` | Auto-set when API key present |
| `LANGCHAIN_API_KEY` | No | - | Auto-set to LANGSMITH_API_KEY |
| `LANGCHAIN_PROJECT` | No | - | Auto-set to LANGSMITH_PROJECT |

*Required only if you want tracing enabled

## Command Line Options

| Option | Description |
|--------|-------------|
| `--no-tracing` | Disable LangSmith tracing even if API key is set |
| `--langsmith-project PROJECT` | Override the project name for this session |

## Troubleshooting

### Tracing Not Working
1. Check that `LANGSMITH_API_KEY` is set in your `.env` file
2. Verify the API key is valid (starts with `ls_`)
3. Ensure you have network connectivity to LangSmith servers

### Can't See Traces
1. Make sure you're logged into the correct LangSmith account
2. Check that the project name matches what's shown in the console
3. Traces may take a few seconds to appear in the dashboard

### Performance Impact
- LangSmith adds minimal overhead (~10-50ms per trace)
- Network calls to LangSmith are asynchronous
- You can disable tracing with `--no-tracing` if needed

## Example Session

```bash
$ python cli_demo.py --demo --langsmith-project "demo-session-2024"

🔍 LangSmith tracing enabled
   Project: demo-session-2024
   Dashboard: https://smith.langchain.com/

🎬 DVD Rental SQL Visualization System - Demo Mode
🎭 Running predefined demo queries
============================================================

[1/6] Demo Query: Show me the top 10 grossing film categories
--------------------------------------------------
✅ Success: True
⏱️  Time: 3.45s
📊 Plot: bar - Bar chart best shows category comparison with clear ranking
🔍 Trace: https://smith.langchain.com/projects/p/demo-session-2024
```

## Next Steps
1. Set up your LangSmith account and API key
2. Run a few demo queries to generate traces
3. Explore the LangSmith dashboard to understand agent behavior
4. Use traces to optimize and debug your queries

Happy tracing! 🔍
