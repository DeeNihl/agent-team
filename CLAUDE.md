Lets design a bare bones python autonomous agent project.
Please write a straight forward, well documented python script that uses these components as a part of the solution:
  our default model provider is ollama. use ollama best practices. 
  use jinja templates for constructing strings like prompt_string and context_string. prompt_string should have placeholders for the asset_paths_csv_list, tools_csv_list, and context_string parameters somewhere within the prompt.
  STEP_LIMIT = 10,   SESSION_TIME_LIMIT = 1800  # SESSION TIME LIMIT is in seconds, STEP_LIMIT is the the maximum number of steps the PLANNER can break the PLAN into and rows limit for prompt_queue.csv 
  There are 4 agent roles on our team [MANAGER,ACTOR,PLANNER,EVALUATOR]
  all prompts sent to PLANNER should begin with "You are a master planner, you excel at explaining linear workflows, step by step approaches, encouraging and enabling your MANAGER with detailed tasks and suggestions. You break down high level tasks into steps which can be performed by small basic llms. WRITE this detailed plan as markdown and return only it as your output."
  all prompts sent to MANAGER should begin with "You are a team manager and excel as choosing the right AGENT_ROLE, and AGENT_INSTRUCTIONS (the instructions you have defined, for the agent assigned to this task) and CONTEXT comprised of tools, asset_paths, context_string. You READ the PLANNER's PLAN.md and construct a prompt_queue.csv for the SESSION in the SESSION ROOT_FOLDER by breaking the PLAN down into individual steps written as records to promt_queue.csv. Try to limit yourself to no more steps than the {STEP_LIMIT}  You write your AGENT_INSTRUCTIONS to prompt_queue.csv within each SESSION_ROOT_FOLDER based on the outline provided by PLANNER in PLAN.md and the AGENT_ROLE most capable of performing the instructions. Identify the correct asset_paths_csv_list, tools_csv_list, and context_string to use for this task and add these to prompt_queue.csv m to the context_{pq_id).csv in the SESSION_ROOT_FOLDER "
  

OUTER LOOP iterates over session_queue.csv which contains a row for each high level task our agent team will tackle.
 
session_queue.csv columns: session_name, session_prompt, session_folder_path

create a SESSION_ROOT_FOLDER using the format {yyyymmdd}-{session_name} 
  all objects created during this session will be placed in that folder, and that folders is treated as the SESSION_ROOT_FOLDER

 agent_model_props.csv
  contains columns (amp_id,AGENT_ROLE,model_name,model_props_args, AGENT_ROLE_PROMPT, TOOL_LIST)
    * model_props_args is a key:value comma separated list. if there is a key called api_key then use the value of that as both a variable called  model_provider_name ,as the prefix for an environment variable ending in _api_key,and as the prefix for an environment variable ending in _api_url. 
	   e.g (api_key:openai -> model_provider_name="openai" & environment variable name "openai_api_key" ). set variables within the loops scope for"model_provider_name", "api_key", and api_url.
	   if there is no "api_key" in model_props_args then use the model_props_args to set properties of the requests post's data. e.g.(data = {"model": "qwq:latest","max_tokens": 1000,"temperature": 0.3, "num_ctx": 32000})
  The possible values for AGENT_ROLE are [MANAGER,ACTOR,PLANNER,EVALUATOR]
  TOOLS is a list of tools available to the agent
   
SESSION LOOP (for each session in session_queue.csv)

	SESSION_PROMPT is sent to the PLANNER and the PLANNER's output is written to PLAN.md in the SESSION_ROOT_FOLDER
	PLAN.md is sent to MANAGER who returns the contents of a new prompt_queue.csv file  with AGENT_ROLE, and AGENT_INSTRUCTIONS populated but not tools_csvlist, asset_file_paths, and context_string. 

	prompt_queue.csv
	  contains columns: pq_id,AGENT_ROLE, AGENT_INSTRUCTIONS,asset_file_paths="",context_string="",output_schema="",status="",start="",end="")
		* possible status values ["","started","evaluate","evaluating","failed","done","retry"]
	  WRITTEN to by MANAGER

			STEP LOOP (for each STEP in prompt_queue)
			SET step_start_time
			check the LAST_TASK_STATUS by looking at the last STEP  processed from prompt_queue.csv if there is one
			if the LAST TASK_STATUS is "failed" write LAST_AGENT_RESPONSE and any exceptions or error detailes to FIN.md and exit this loop to start the next session. 
			Get the  agent_model_props[] for this AGENT_ROLE from  agent_model_props.csv

			CLEAR CONTEXT
			CONTEXT
					ADD to CONTEXT - "<|instructions|>{AGENT_ROLE_PROMPT}</|instructions|>
					ADD to CONTEXT - PREVIOUSLY:  {a niceley formatted markdown  version of previous processed STEP's AGENT_INSTRUCTIONS from prompt_queue.csv , if there is one, otherwise use ""} 
					ADD to CONTEXT - LAST OUTPUT: TOOLS used & the complete out from the last ACTOR response
					ADD TO CONTEXT - CURRENT TASK: {get a niceley formatted markdown version of the content of this row  from prompt_queue.csv} 
					ADD to CONTEXT - USE ASSETS: IF MANAGER THINKS THEYE ARE NEEDED , any {ASSET_PATHS} that contains files created by the last STEP, or recommended by the MANAGER
					ADD TO CONTEXT - USE TOOLS:  IF MANAGER THINKS THEYE ARE NEEDED, a list of tools for the ACTOR chosen from the AGENT_ROLE TOOL_LIST
					ADD TO CONTEXT - OUTPUT SCHEMA: only if included in the STEP 

			SET the value of  asset_file_paths=ASSET_PATHS, context_string=CONTEXT, and status="started", and start=current date and time as " yyyyMMddHHmmss" format
			Create an ACTOR_PROMPT from CONTEXT
			SEND ACTOR_PROMPT and other data in a request to ollama for the model and parameters from  agent_model_props.csv based on the STEP's AGENT_ROLE
			WRITE the output_{pq_id}.json is a json object in a file named for the associated pq_id of the STEP. If the output_schema is JSON write the json exactly as it was received, otherwise put the response in a single property "RESPONSE". it is the output for. properties: pq_id,
			CHECK THAT WE HAVE NOT EXCEDED THE STEP_TIME_LIMIT. IF WE HAVE, SET the LAST TASK_STATUS and LAST_AGENT_RESPONSE AND GO TO NEXT STEP
			PROMPT the EVALUATOR to review the CONTEXT and REPONSE to determine what to set the STEP's status to. EVALUATOR_RESPONSE  should be [GOOD, STOP, or RETRY] 
			CHECK THAT WE HAVE NOT EXCEDED THE STEP_TIME_LIMIT. IF WE HAVE, SET the LAST TASK_STATUS and LAST_AGENT_RESPONSE AND GO TO NEXT STEP
			  if the EVALUATOR_RESPONSE is RETRY
			    update the  temperature parameter of this AGENT_ROLE's model_props_args["temperature"] with a random change of plus or minus 0.2 from it's current value, and save the file.
                SEND ACTOR_PROMPT and other data in a request to ollama for the model and parameters from  agent_model_props.csv based on the STEP's AGENT_ROLE
			    WRITE the output_{pq_id}.json is a json object in a file named for the associated pq_id of the STEP. If the output_schema is JSON write the json exactly as it was received, otherwise put the response in a single property "RESPONSE". it is the output for. properties: pq_id,
			    CHECK THAT WE HAVE NOT EXCEDED THE STEP_TIME_LIMIT. IF WE HAVE, SET the LAST TASK_STATUS and LAST_AGENT_RESPONSE AND GO TO NEXT STEP
                PROMPT the EVALUATOR to review the CONTEXT and  ACTOR_RETRY_REPONSE to determine what to set the STEP's status to. EVALUATOR_RESPONSE  should be [GOOD or STOP] 
                  If the latest EVALUATOR_REPONSE is "STOP" then clear the SESSION level and STEP level variables, exit the STEP loop, and start the next session.
 			SET the LAST TASK_STATUS and LAST_AGENT_RESPONSE
		continue the step loop
	continue the Session Loop