"""
Pytest configuration for agent team tests.
"""
import os
import csv
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def agent_model_props_path():
    """Get the path to the agent model properties file."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                        "agent_team", "data", "agent_model_props.csv")


@pytest.fixture
def agent_roles():
    """Return a list of all agent roles."""
    return ["PLANNER", "MANAGER", "ACTOR", "EVALUATOR"]


@pytest.fixture
def test_props_file():
    """Create a temporary props file for testing with valid entries."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        writer = csv.writer(f)
        writer.writerow([
            "amp_id", "AGENT_ROLE", "model_name", "model_props_args", 
            "AGENT_ROLE_PROMPT", "TOOL_LIST"
        ])
        writer.writerow([
            "1", "PLANNER", "qwq", "model:qwq:latest,temperature:0.8,num_ctx:32000,api_url:10.0.0.16:11434,api_key:",
            "You are a planner", ""
        ])
        writer.writerow([
            "2", "MANAGER", "qwq", "model:qwq:latest,temperature:0.7,num_ctx:32000,api_url:10.0.0.16:11434,api_key:",
            "You are a manager", ""
        ])
        writer.writerow([
            "3", "ACTOR", "gemma", "model:gemma:12b,temperature:0.7,num_ctx:32000,api_url:10.0.0.16:11434,api_key:",
            "You are an actor", ""
        ])
        writer.writerow([
            "4", "EVALUATOR", "gemma", "model:gemma:12b,temperature:0.3,num_ctx:32000,api_url:10.0.0.16:11434,api_key:",
            "You are an evaluator", ""
        ])
    
    yield f.name
    # Clean up the temporary file
    os.unlink(f.name)


@pytest.fixture
def test_agent_team():
    """Create an instance of the AgentTeam class."""
    from agent_team.main import AgentTeam
    return AgentTeam()