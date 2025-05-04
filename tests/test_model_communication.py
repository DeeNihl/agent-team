"""
Tests for model communication in the agent team.

These tests verify that the agent team can communicate with each model
for all agent roles defined in the agent_model_props.csv file.
"""
import os
import csv
import pytest
import requests
from unittest import mock

from agent_team.main import AgentTeam


def test_agent_model_props_exists(agent_model_props_path):
    """Test that the agent model properties file exists."""
    assert os.path.exists(agent_model_props_path), "Agent model properties file does not exist"
    
    # Verify the file has content
    with open(agent_model_props_path, 'r') as f:
        content = f.read()
        assert len(content) > 0, "Agent model properties file is empty"
        
    # Verify the file is valid CSV
    with open(agent_model_props_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) > 1, "Agent model properties file has no data rows"
        assert len(rows[0]) >= 5, "Agent model properties file has insufficient columns"


@pytest.mark.parametrize("agent_role", ["PLANNER", "MANAGER", "ACTOR", "EVALUATOR"])
def test_parse_model_props(test_agent_team, agent_role, agent_model_props_path):
    """Test that the agent team can parse model properties for each agent role."""
    # Load the properties from the file
    agent_props = None
    with open(agent_model_props_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['AGENT_ROLE'] == agent_role:
                agent_props = row
                break
    
    assert agent_props is not None, f"No properties found for {agent_role}"
    
    # Test parsing model props
    model_props_args = agent_props.get('model_props_args', '')
    model_params = test_agent_team._parse_model_props(model_props_args)
    
    assert isinstance(model_params, dict), "Model parameters should be a dictionary"
    
    # Check that we have the expected properties
    if 'api_url' in model_props_args:
        assert 'api_url' in model_params, f"api_url not found in model parameters for {agent_role}"
    
    # Make sure model key exists which is needed for ollama
    if 'model:' in model_props_args:
        model_keys = [k for k in model_params.keys() if k == 'model']
        assert len(model_keys) > 0, f"No model specified in model parameters for {agent_role}"


class MockResponse:
    """Mock response for requests."""
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        
    def json(self):
        """Return the mock JSON data."""
        return self.json_data
        
    def raise_for_status(self):
        """Raise an exception if the status code indicates an error."""
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}")


@pytest.mark.parametrize("agent_role", ["PLANNER", "MANAGER", "ACTOR", "EVALUATOR"])
def test_model_communication_planner(test_agent_team, agent_role, agent_model_props_path):
    """Test that the agent team can communicate with the planner model."""
    # Load the properties from the file
    agent_props = None
    with open(agent_model_props_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['AGENT_ROLE'] == agent_role:
                agent_props = row
                break
    
    assert agent_props is not None, f"No properties found for {agent_role}"
    
    # Mock the requests.post method
    with mock.patch('requests.post') as mock_post:
        # Set up the mock to return a successful response
        mock_post.return_value = MockResponse({
            'response': 'This is a test response.'
        })
        
        # Test the model communication
        response = test_agent_team._call_model(
            agent_role=agent_role,
            prompt="This is a test prompt.",
            agent_props=agent_props
        )
        
        # Verify the response
        assert isinstance(response, str), "Response should be a string"
        assert response == 'This is a test response.', "Response does not match expected value"
        
        # Verify that requests.post was called with the correct arguments
        mock_post.assert_called_once()
        
        # Extract the args and kwargs from the call
        args, kwargs = mock_post.call_args
        
        # Verify that the URL is correct
        url = args[0] if args else kwargs.get('url')
        assert url is not None, "URL not found in requests.post call"
        
        # Verify that the data includes a prompt
        data = kwargs.get('json')
        assert data is not None, "JSON data not found in requests.post call"
        assert 'prompt' in data, "Prompt not found in JSON data"
        assert data['prompt'] == "This is a test prompt.", "Prompt does not match expected value"


# Test with actual model communication if the environment allows it
# This test is marked as 'optional' because it requires access to the actual model
@pytest.mark.optional
@pytest.mark.parametrize("agent_role", ["PLANNER", "MANAGER", "ACTOR", "EVALUATOR"])
def test_actual_model_communication(test_agent_team, agent_role, agent_model_props_path):
    """Test that the agent team can communicate with the actual model."""
    # Skip this test if we're in a CI environment or don't have access to the model
    if os.environ.get('CI') == 'true' or not os.environ.get('TEST_ACTUAL_MODEL'):
        pytest.skip("Skipping actual model communication test in CI or without TEST_ACTUAL_MODEL=true")
    
    # Load the properties from the file
    agent_props = None
    with open(agent_model_props_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['AGENT_ROLE'] == agent_role:
                agent_props = row
                break
    
    assert agent_props is not None, f"No properties found for {agent_role}"
    
    # Test the model communication with a very short prompt
    response = test_agent_team._call_model(
        agent_role=agent_role,
        prompt=f"This is a test prompt for {agent_role}. Please respond with a single word: 'test'.",
        agent_props=agent_props
    )
    
    # Verify that we got some kind of response and not an error
    assert isinstance(response, str), "Response should be a string"
    assert not response.startswith("Error:"), f"Response indicates an error: {response}"
    
    # The response should have at least some length
    assert len(response) > 0, "Response is empty"