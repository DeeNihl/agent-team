# Agent Team

A bare bones Python autonomous agent project that manages a team of agents with different roles.

## Overview

This project provides a framework for autonomous agent teams that collaborate on tasks. The system supports four agent roles:

- **MANAGER**: Translates plans into actionable steps and assigns roles
- **ACTOR**: Executes the tasks defined by the Manager
- **PLANNER**: Creates high-level plans for tasks
- **EVALUATOR**: Reviews task outputs and determines next actions

All agents operate within sessions that work through a queue of high-level tasks.

## Requirements

- Python 3.7+
- [Ollama](https://ollama.ai/) (default model provider)
- Required Python packages (see requirements.txt)

## Installation

```bash
# Clone the repository
git clone https://github.com/DeeNihl/agent-team.git
cd agent-team

# Install required packages
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

1. Ensure Ollama is running and has the required models available (default: llama3)
2. Create a `session_queue.csv` file in the `agent_team/data` directory with session details
3. Run the main script:

```bash
python -m agent_team.main
```

### Session Queue Format

The `session_queue.csv` file should have the following columns:
- `session_name`: A unique name for the session
- `session_prompt`: The high-level task description for the session
- `session_folder_path`: (Optional) A custom folder path for the session

### Agent Model Properties

Agent model properties are defined in `agent_team/data/agent_model_props.csv` with these columns:
- `amp_id`: Unique identifier for the agent model properties
- `AGENT_ROLE`: One of MANAGER, ACTOR, PLANNER, or EVALUATOR
- `model_name`: The name of the LLM model to use
- `model_props_args`: Comma-separated key:value pairs for model properties
- `AGENT_ROLE_PROMPT`: The prompt prefix for the agent role
- `TOOL_LIST`: List of tools available to the agent

## Project Structure

```
agent-team/
├── agent_team/
│   ├── data/          # Configuration and session data
│   ├── logs/          # Log files
│   ├── templates/     # Jinja templates for prompts
│   ├── tools/         # Agent tools
│   └── main.py        # Main entry point
├── logs/              # Global logs directory
├── tests/             # Unit tests
├── requirements.txt   # Project dependencies
├── setup.py           # Package setup
└── README.md          # Project documentation
```

## Testing

The project uses pytest for testing. To run the tests:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage report
pytest --cov=agent_team

# Skip tests that require actual model communication
pytest -m "not optional"
```

The tests include:
- Unit tests for each agent role (PLANNER, MANAGER, ACTOR, EVALUATOR)
- Tests that verify model communication works correctly
- Tests that validate configuration file parsing

## License

MIT