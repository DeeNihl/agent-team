#!/usr/bin/env python3
"""
Autonomous Agent Team System.

A bare bones Python autonomous agent project that manages a team of agents with different roles:
- MANAGER: Translates plans into actionable steps and assigns roles
- ACTOR: Executes the tasks defined by the Manager
- PLANNER: Creates high-level plans for tasks
- EVALUATOR: Reviews task outputs and determines next actions

All agents operate within sessions that work through a queue of high-level tasks.
"""

import os
import csv
import json
import time
import random
import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any

import jinja2
import requests

# Constants
STEP_LIMIT = 10
SESSION_TIME_LIMIT = 1800  # 30 minutes in seconds

# Agent roles
AGENT_ROLES = ["MANAGER", "ACTOR", "PLANNER", "EVALUATOR"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentTeam:
    """Manages a team of autonomous agents that collaborate on tasks."""
    
    def __init__(self):
        """Initialize the agent team system."""
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("agent_team/templates"),
            autoescape=jinja2.select_autoescape()
        )
        self.session_queue = []
        self.agent_model_props = {}
        self.last_task_status = ""
        self.last_agent_response = ""
        
        # Load agent model properties
        self._load_agent_model_props()
        
    def _load_agent_model_props(self) -> None:
        """Load agent model properties from CSV file."""
        try:
            with open("agent_team/data/agent_model_props.csv", 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.agent_model_props[row['AGENT_ROLE']] = row
                    
            logger.info("Loaded agent model properties")
        except FileNotFoundError:
            logger.error("agent_model_props.csv not found. Creating default configuration.")
            self._create_default_agent_model_props()
    
    def _create_default_agent_model_props(self) -> None:
        """Create default agent model properties file if not exists."""
        os.makedirs("agent_team/data", exist_ok=True)
        
        with open("agent_team/data/agent_model_props.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "amp_id", "AGENT_ROLE", "model_name", "model_props_args", 
                "AGENT_ROLE_PROMPT", "TOOL_LIST"
            ])
            
            # Write default values for each role
            for i, role in enumerate(AGENT_ROLES):
                writer.writerow([
                    i+1,
                    role,
                    "llama3",
                    "model:llama3:latest,max_tokens:2000,temperature:0.7,num_ctx:32000",
                    self._get_default_role_prompt(role),
                    ""
                ])
                
        logger.info("Created default agent model properties file")
    
    def _get_default_role_prompt(self, role: str) -> str:
        """Return default prompt for each agent role."""
        prompts = {
            "PLANNER": "You are a master planner, you excel at explaining linear workflows, step by step approaches, encouraging and enabling your MANAGER with detailed tasks and suggestions. You break down high level tasks into steps which can be performed by small basic llms. WRITE this detailed plan as markdown and return only it as your output.",
            "MANAGER": "You are a team manager and excel as choosing the right AGENT_ROLE, and AGENT_INSTRUCTIONS (the instructions you have defined, for the agent assigned to this task) and CONTEXT comprised of tools, asset_paths, context_string. You READ the PLANNER's PLAN.md and construct a prompt_queue.csv for the SESSION in the SESSION ROOT_FOLDER by breaking the PLAN down into individual steps written as records to promt_queue.csv. Try to limit yourself to no more steps than the {STEP_LIMIT} You write your AGENT_INSTRUCTIONS to prompt_queue.csv within each SESSION_ROOT_FOLDER based on the outline provided by PLANNER in PLAN.md and the AGENT_ROLE most capable of performing the instructions. Identify the correct asset_paths_csv_list, tools_csv_list, and context_string to use for this task and add these to prompt_queue.csv m to the context_{pq_id).csv in the SESSION_ROOT_FOLDER ",
            "ACTOR": "You are an ACTOR agent responsible for executing specific tasks as defined by the MANAGER. Follow the instructions carefully and use the provided tools and assets to complete your task.",
            "EVALUATOR": "You are an EVALUATOR agent responsible for reviewing task outputs. Determine if the output meets the requirements with a response of GOOD, STOP, or RETRY."
        }
        return prompts.get(role, "You are an agent on the team.")
            
    def load_session_queue(self, queue_path: str = "agent_team/data/session_queue.csv") -> None:
        """Load the session queue from a CSV file."""
        try:
            with open(queue_path, 'r') as f:
                reader = csv.DictReader(f)
                self.session_queue = list(reader)
                
            logger.info(f"Loaded {len(self.session_queue)} sessions from queue")
        except FileNotFoundError:
            logger.error("session_queue.csv not found. Creating example file.")
            self._create_default_session_queue(queue_path)
    
    def _create_default_session_queue(self, path: str) -> None:
        """Create a default session queue file with an example task."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_name", "session_prompt", "session_folder_path"])
            writer.writerow([
                "hello_world",
                "Create a simple Python script that prints 'Hello, Agent Team!' to the console.",
                ""
            ])
            
        logger.info("Created default session queue file")
            
    def run(self) -> None:
        """Run the main loop processing all sessions in the queue."""
        logger.info("Starting agent team system")
        
        # Load session queue if not already loaded
        if not self.session_queue:
            self.load_session_queue()
            
        # Process each session in the queue
        for session in self.session_queue:
            try:
                self._process_session(session)
            except Exception as e:
                logger.error(f"Error processing session {session.get('session_name')}: {str(e)}")
                continue
                
        logger.info("All sessions completed")
    
    def _process_session(self, session: Dict[str, str]) -> None:
        """Process a single session from the queue."""
        session_name = session.get('session_name')
        session_prompt = session.get('session_prompt')
        
        # Create session folder
        today = datetime.datetime.now().strftime("%Y%m%d")
        session_folder = session.get('session_folder_path') or f"{today}-{session_name}"
        session_root = os.path.join("agent_team", session_folder)
        os.makedirs(session_root, exist_ok=True)
        
        logger.info(f"Processing session: {session_name} in {session_root}")
        
        # Step 1: Send prompt to PLANNER and write output to PLAN.md
        plan = self._run_planner(session_prompt)
        plan_path = os.path.join(session_root, "PLAN.md")
        with open(plan_path, 'w') as f:
            f.write(plan)
            
        # Step 2: Send PLAN.md to MANAGER to create prompt_queue.csv
        prompt_queue = self._run_manager(plan_path, session_root)
        prompt_queue_path = os.path.join(session_root, "prompt_queue.csv")
        
        # Process each step in the prompt queue
        for step in prompt_queue:
            try:
                start_time = time.time()
                
                # Check if we're exceeding the session time limit
                if (time.time() - start_time) > SESSION_TIME_LIMIT:
                    logger.warning(f"Session time limit exceeded for {session_name}")
                    break
                    
                # Process the step
                self._process_step(step, prompt_queue, session_root)
                
                # If the last step failed, write to FIN.md and exit
                if self.last_task_status == "failed":
                    self._write_fin_md(session_root)
                    break
                    
            except Exception as e:
                logger.error(f"Error processing step {step.get('pq_id')}: {str(e)}")
                self.last_task_status = "failed"
                self.last_agent_response = str(e)
                self._write_fin_md(session_root)
                break
                
        logger.info(f"Session {session_name} completed with status: {self.last_task_status}")
    
    def _run_planner(self, session_prompt: str) -> str:
        """Run the PLANNER agent to create a plan from the session prompt."""
        logger.info("Running PLANNER agent")
        
        # Get the model properties for PLANNER
        planner_props = self.agent_model_props.get("PLANNER", {})
        planner_prompt = planner_props.get("AGENT_ROLE_PROMPT", "")
        
        # Create the prompt using template
        template = self.env.get_template("planner_prompt.j2")
        prompt_string = template.render(
            role_prompt=planner_prompt,
            session_prompt=session_prompt
        )
        
        # Send prompt to model
        response = self._call_model(
            "PLANNER", 
            prompt_string, 
            planner_props
        )
        
        return response
    
    def _run_manager(self, plan_path: str, session_root: str) -> List[Dict[str, str]]:
        """Run the MANAGER agent to create a prompt queue from the plan."""
        logger.info("Running MANAGER agent")
        
        # Get the model properties for MANAGER
        manager_props = self.agent_model_props.get("MANAGER", {})
        manager_prompt = manager_props.get("AGENT_ROLE_PROMPT", "")
        
        # Read the plan file
        with open(plan_path, 'r') as f:
            plan_content = f.read()
            
        # Create the prompt using template
        template = self.env.get_template("manager_prompt.j2")
        prompt_string = template.render(
            role_prompt=manager_prompt.replace("{STEP_LIMIT}", str(STEP_LIMIT)),
            plan=plan_content,
            session_root=session_root,
            step_limit=STEP_LIMIT
        )
        
        # Send prompt to model
        response = self._call_model(
            "MANAGER", 
            prompt_string, 
            manager_props
        )
        
        # Parse response to create prompt_queue.csv
        prompt_queue = []
        
        # Create CSV header
        header = [
            "pq_id", "AGENT_ROLE", "AGENT_INSTRUCTIONS", 
            "asset_file_paths", "context_string", "output_schema",
            "status", "start", "end"
        ]
        
        # Write response to prompt_queue.csv
        prompt_queue_path = os.path.join(session_root, "prompt_queue.csv")
        with open(prompt_queue_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Parse the response and write to CSV
            # This is a simplified approach - in practice, you'd need to
            # properly parse the manager's response to extract the queue items
            lines = response.strip().split('\n')
            for i, line in enumerate(lines[:STEP_LIMIT]):  # Enforce step limit
                if line and not line.startswith('#') and ',' in line:
                    parts = line.split(',', 2)
                    if len(parts) >= 3:
                        agent_role = parts[0].strip()
                        agent_instructions = parts[1].strip()
                        
                        # Validate agent role
                        if agent_role in AGENT_ROLES:
                            row = [
                                i+1,  # pq_id
                                agent_role,
                                agent_instructions,
                                "",  # asset_file_paths
                                "",  # context_string
                                "",  # output_schema
                                "",  # status
                                "",  # start
                                "",  # end
                            ]
                            writer.writerow(row)
                            
                            # Add to our in-memory queue
                            prompt_queue.append(dict(zip(header, row)))
        
        logger.info(f"Created prompt queue with {len(prompt_queue)} steps")
        return prompt_queue
        
    def _process_step(self, step: Dict[str, str], prompt_queue: List[Dict[str, str]], session_root: str) -> None:
        """Process a single step from the prompt queue."""
        pq_id = step.get('pq_id')
        agent_role = step.get('AGENT_ROLE')
        agent_instructions = step.get('AGENT_INSTRUCTIONS')
        
        logger.info(f"Processing step {pq_id} with {agent_role}")
        
        # Update step start time
        step_start_time = datetime.datetime.now()
        step['start'] = step_start_time.strftime("%Y%m%d%H%M%S")
        step['status'] = "started"
        
        # Get agent model properties
        agent_props = self.agent_model_props.get(agent_role, {})
        agent_role_prompt = agent_props.get("AGENT_ROLE_PROMPT", "")
        
        # Build context
        previous_step = None
        previous_response = ""
        if prompt_queue and int(pq_id) > 1:
            prev_id = int(pq_id) - 1
            for s in prompt_queue:
                if s.get('pq_id') == str(prev_id):
                    previous_step = s
                    break
                    
            if previous_step:
                output_path = os.path.join(session_root, f"output_{prev_id}.json")
                if os.path.exists(output_path):
                    try:
                        with open(output_path, 'r') as f:
                            previous_data = json.load(f)
                            previous_response = previous_data.get('RESPONSE', '')
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode previous output: {output_path}")
        
        # Create formatted context
        context = self._build_context(
            agent_role_prompt=agent_role_prompt,
            previous_step=previous_step,
            previous_response=previous_response,
            current_step=step
        )
        
        # Update step with context
        step['context_string'] = context
        
        # Create actor prompt from context
        template = self.env.get_template("actor_prompt.j2")
        actor_prompt = template.render(
            context=context,
            instructions=agent_instructions
        )
        
        # Send prompt to model
        actor_response = self._call_model(
            agent_role, 
            actor_prompt, 
            agent_props
        )
        
        # Write response to output file
        output_path = os.path.join(session_root, f"output_{pq_id}.json")
        output_data = {
            "pq_id": pq_id,
            "RESPONSE": actor_response
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Check if we're over time limit
        if (datetime.datetime.now() - step_start_time).total_seconds() > SESSION_TIME_LIMIT:
            logger.warning(f"Step time limit exceeded for step {pq_id}")
            self.last_task_status = "failed"
            self.last_agent_response = "Step time limit exceeded"
            return
            
        # Evaluate the response
        evaluator_props = self.agent_model_props.get("EVALUATOR", {})
        evaluator_prompt = evaluator_props.get("AGENT_ROLE_PROMPT", "")
        
        template = self.env.get_template("evaluator_prompt.j2")
        eval_prompt = template.render(
            role_prompt=evaluator_prompt,
            context=context,
            response=actor_response
        )
        
        evaluator_response = self._call_model(
            "EVALUATOR", 
            eval_prompt, 
            evaluator_props
        ).strip().upper()
        
        # Check if we need to retry
        if evaluator_response == "RETRY":
            logger.info(f"Retrying step {pq_id}")
            
            # Modify temperature slightly for retry
            agent_props_args = agent_props.get('model_props_args', '')
            model_params = self._parse_model_props(agent_props_args)
            
            if 'temperature' in model_params:
                current_temp = float(model_params['temperature'])
                new_temp = current_temp + random.uniform(-0.2, 0.2)
                new_temp = max(0.1, min(1.0, new_temp))  # Keep in sensible range
                model_params['temperature'] = str(new_temp)
                
                # Update the model props for this retry
                agent_props['model_props_args'] = ','.join([f"{k}:{v}" for k, v in model_params.items()])
            
            # Retry with adjusted temperature
            actor_response = self._call_model(
                agent_role, 
                actor_prompt, 
                agent_props
            )
            
            # Update the output file with retry response
            output_data = {
                "pq_id": pq_id,
                "RESPONSE": actor_response,
                "RETRY": True
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Re-evaluate after retry
            eval_prompt = template.render(
                role_prompt=evaluator_prompt,
                context=context,
                response=actor_response
            )
            
            evaluator_response = self._call_model(
                "EVALUATOR", 
                eval_prompt, 
                evaluator_props
            ).strip().upper()
            
            # If still not good, mark as failed
            if evaluator_response == "STOP":
                logger.warning(f"Evaluation failed after retry for step {pq_id}")
                self.last_task_status = "failed"
                self.last_agent_response = actor_response
                return
        
        # Update step status based on evaluation
        if evaluator_response == "GOOD":
            step['status'] = "done"
        else:
            step['status'] = "failed"
            
        step['end'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Update the prompt queue CSV file with new status
        self._update_prompt_queue_csv(prompt_queue, session_root)
        
        # Set status for next iteration
        self.last_task_status = step['status']
        self.last_agent_response = actor_response
    
    def _update_prompt_queue_csv(self, prompt_queue: List[Dict[str, str]], session_root: str) -> None:
        """Update the prompt queue CSV file with current status."""
        prompt_queue_path = os.path.join(session_root, "prompt_queue.csv")
        
        # Create CSV header
        header = [
            "pq_id", "AGENT_ROLE", "AGENT_INSTRUCTIONS", 
            "asset_file_paths", "context_string", "output_schema",
            "status", "start", "end"
        ]
        
        with open(prompt_queue_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for step in prompt_queue:
                row = [step.get(h, "") for h in header]
                writer.writerow(row)
                
    def _build_context(
        self,
        agent_role_prompt: str,
        previous_step: Optional[Dict[str, str]],
        previous_response: str,
        current_step: Dict[str, str]
    ) -> str:
        """Build the context string for an agent prompt."""
        context_parts = []
        
        # Add instructions
        context_parts.append(f"<|instructions|>{agent_role_prompt}</|instructions|>")
        
        # Add previous step info if available
        if previous_step:
            context_parts.append("\nPREVIOUSLY:")
            context_parts.append(f"```markdown\nAGENT_ROLE: {previous_step.get('AGENT_ROLE')}")
            context_parts.append(f"INSTRUCTIONS: {previous_step.get('AGENT_INSTRUCTIONS')}\n```")
            
            # Add previous output
            context_parts.append("\nLAST OUTPUT:")
            context_parts.append(f"```\n{previous_response}\n```")
        
        # Add current task info
        context_parts.append("\nCURRENT TASK:")
        context_parts.append(f"```markdown\nAGENT_ROLE: {current_step.get('AGENT_ROLE')}")
        context_parts.append(f"INSTRUCTIONS: {current_step.get('AGENT_INSTRUCTIONS')}\n```")
        
        # Add asset paths if needed
        asset_paths = current_step.get('asset_file_paths', '')
        if asset_paths:
            context_parts.append("\nUSE ASSETS:")
            context_parts.append(f"```\n{asset_paths}\n```")
            
        # Add tool list if needed from agent model props
        agent_props = self.agent_model_props.get(current_step.get('AGENT_ROLE', ''), {})
        tool_list = agent_props.get('TOOL_LIST', '')
        if tool_list:
            context_parts.append("\nUSE TOOLS:")
            context_parts.append(f"```\n{tool_list}\n```")
            
        # Add output schema if specified
        output_schema = current_step.get('output_schema', '')
        if output_schema:
            context_parts.append("\nOUTPUT SCHEMA:")
            context_parts.append(f"```\n{output_schema}\n```")
            
        return "\n".join(context_parts)
    
    def _call_model(self, agent_role: str, prompt: str, agent_props: Dict[str, str]) -> str:
        """Call the LLM with the specified prompt and parameters."""
        model_name = agent_props.get('model_name', 'llama3')
        model_props_args = agent_props.get('model_props_args', '')
        
        # Parse model properties
        model_params = self._parse_model_props(model_props_args)
        
        # Check if we have an API key setting
        api_key = model_params.get('api_key', '')
        if api_key:
            model_provider_name = api_key
            api_key_env = f"{api_key}_api_key"
            api_url_env = f"{api_key}_api_url"
            
            # Get API key and URL from environment variables
            api_key_value = os.environ.get(api_key_env)
            api_url_value = os.environ.get(api_url_env, "http://localhost:11434/api/generate")
            
            # Configure request for external API
            headers = {"Authorization": f"Bearer {api_key_value}"}
            data = {
                "model": model_name,
                "prompt": prompt
            }
            
            # Add other parameters
            for k, v in model_params.items():
                if k != 'api_key':
                    try:
                        # Convert to appropriate types
                        if v.isdigit():
                            data[k] = int(v)
                        elif v.replace('.', '', 1).isdigit():
                            data[k] = float(v)
                        else:
                            data[k] = v
                    except (ValueError, AttributeError):
                        data[k] = v
                        
            # Make API request
            try:
                response = requests.post(api_url_value, headers=headers, json=data)
                response.raise_for_status()
                return response.json().get('response', '')
            except requests.RequestException as e:
                logger.error(f"API request failed: {str(e)}")
                return f"Error: {str(e)}"
                
        else:
            # Default to ollama
            # Use api_url from parameters if provided, otherwise use default
            url = model_params.get('api_url', "http://localhost:11434/api/generate")
            
            # Make sure url has the full API path
            if not url.endswith('/api/generate'):
                url = f"{url}/api/generate"
                
            # Format the data according to Ollama API requirements
            data = {
                "model": model_name,
                "prompt": prompt,
                "options": {}
            }
            
            # Add other parameters to the options object, excluding api_url and api_key
            for k, v in model_params.items():
                if k not in ['api_url', 'api_key', 'model']:
                    try:
                        # Convert to appropriate types
                        if v.isdigit():
                            data["options"][k] = int(v)
                        elif v.replace('.', '', 1).isdigit():
                            data["options"][k] = float(v)
                        else:
                            data["options"][k] = v
                    except (ValueError, AttributeError):
                        data["options"][k] = v
            
            # Make ollama API request
            try:
                logger.info(f"Making Ollama API request to {url}")
                response = requests.post(url, json=data)
                response.raise_for_status()
                return response.json().get('response', '')
            except requests.RequestException as e:
                logger.error(f"Ollama API request failed: {str(e)}")
                return f"Error: {str(e)}"
    
    def _parse_model_props(self, model_props_str: str) -> Dict[str, str]:
        """Parse model properties string into a dictionary."""
        props = {}
        if model_props_str:
            parts = model_props_str.split(',')
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    props[key.strip()] = value.strip()
        return props
    
    def _write_fin_md(self, session_root: str) -> None:
        """Write final status to FIN.md when a session ends with failure."""
        fin_path = os.path.join(session_root, "FIN.md")
        with open(fin_path, 'w') as f:
            f.write(f"# Session Ended\n\n")
            f.write(f"Status: {self.last_task_status}\n\n")
            f.write("## Last Agent Response\n\n")
            f.write(f"```\n{self.last_agent_response}\n```\n")


if __name__ == "__main__":
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Run the agent team
        agent_team = AgentTeam()
        agent_team.run()
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}", exc_info=True)