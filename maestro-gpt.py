import os
import re
from rich.console import Console
from rich.panel import Panel
from datetime import datetime
import json
from config import ANTHROPIC_API_KEY, OPENAI_API_KEY
from utils import read_file, create_folder_structure, create_folders_and_files
from anthropic import Anthropic
from openai import OpenAI

# Set up the Anthropic API client
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Set up the OpenAI API client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Set the Claude model to use for the sub-agent
claude_model = "claude-3-opus-20240229"

# Initialize the Rich Console
console = Console()

def opus_orchestrator(objective: str, file_content: str = None, previous_results: list = None) -> tuple:
    """
    Calls the Orchestrator to break down the objective into sub-tasks.

    Args:
        objective (str): The main objective to be broken down.
        file_content (str, optional): Content of the file if provided. Defaults to None.
        previous_results (list, optional): Results of previous sub-tasks. Defaults to None.

    Returns:
        tuple: Response text from the orchestrator and file content.
    """
    console.print(f"\n[bold]Calling Orchestrator for your objective[/bold]")
    previous_results_text = "\n".join(previous_results) if previous_results else "None"
    if file_content:
        console.print(Panel(f"File content:\n{file_content}", title="[bold blue]File Content[/bold blue]", title_align="left", border_style="blue"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Based on the following objective{' and file content' if file_content else ''}, and the previous sub-task results (if any), please break down the objective into the next sub-task, and create a concise and detailed prompt for a subagent so it can execute that task. IMPORTANT!!! when dealing with code tasks make sure you check the code for errors and provide fixes and support as part of the next sub-task. If you find any bugs or have suggestions for better code, please include them in the next sub-task prompt. Please assess if the objective has been fully achieved. If the previous sub-task results comprehensively address all aspects of the objective, include the phrase 'The task is complete:' at the beginning of your response. If the objective is not yet fully achieved, break it down into the next sub-task and create a concise and detailed prompt for a subagent to execute that task.:\n\nObjective: {objective}" + ('\\nFile content:\\n' + file_content if file_content else '') + f"\n\nPrevious sub-task results:\n{previous_results_text}"}
            ]
        }
    ]

    if orchestrator_model == "Claude Opus":
        opus_response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            messages=messages
        )
        response_text = opus_response.content[0].text
    else:  # GPT-4
        gpt4_response = openai_client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=messages
        )
        response_text = gpt4_response.choices[0].message.content

    console.print(Panel(response_text, title=f"[bold green]{orchestrator_model} Orchestrator[/bold green]", title_align="left", border_style="green", subtitle="Sending task to subagent 👇"))
    return response_text, file_content

def subagent(prompt: str, previous_subagent_tasks: list = None) -> str:
    """
    Calls the subagent to execute the given prompt.

    Args:
        prompt (str): The prompt for the subagent to execute.
        previous_subagent_tasks (list, optional): Previous tasks executed by the subagent. Defaults to None.

    Returns:
        str: Response text from the subagent.
    """
    if previous_subagent_tasks is None:
        previous_subagent_tasks = []

    system_message = "Previous subagent tasks:\n" + "\n".join(previous_subagent_tasks)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    subagent_response = anthropic_client.messages.create(
        model=claude_model,
        max_tokens=4096,
        messages=messages,
        system=system_message
    )

    response_text = subagent_response.content[0].text
    console.print(Panel(response_text, title="[bold blue]Subagent Result[/bold blue]", title_align="left", border_style="blue", subtitle="Task completed, sending result to Maestro 👇"))
    return response_text

def opus_refine(objective: str, sub_task_results: list, filename: str, projectname: str) -> str:
    """
    Calls the Orchestrator to refine the sub-task results into a cohesive final output.

    Args:
        objective (str): The main objective.
        sub_task_results (list): Results of the sub-tasks.
        filename (str): Name of the file.
        projectname (str): Name of the project.

    Returns:
        str: Refined final output.
    """
    print("\nCalling Orchestrator to provide the refined final output for your objective:")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Objective: " + objective + "\n\nSub-task results:\n" + "\n".join(sub_task_results) + "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. When working on code projects, ONLY AND ONLY IF THE PROJECT IS CLEARLY A CODING ONE please provide the following:\n1. Project Name: Create a concise and appropriate project name that fits the project based on what it's creating. The project name should be no more than 20 characters long.\n2. Folder Structure: Provide the folder structure as a valid JSON object, where each key represents a folder or file, and nested keys represent subfolders. Use null values for files. Ensure the JSON is properly formatted without any syntax errors. Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, separating items with commas as necessary.\nWrap the JSON object in <folder_structure> tags.\n3. Code Files: For each code file, include ONLY the file name NEVER EVER USE THE FILE PATH OR ANY OTHER FORMATTING YOU ONLY USE THE FOLLOWING format 'Filename: <filename>' followed by the code block enclosed in triple backticks, with the language identifier after the opening backticks, like this:\n\n​python\n<code>\n​"}
            ]
        }
    ]

    opus_response = anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        messages=messages
    )

    response_text = opus_response.content[0].text
    console.print(Panel(response_text, title="[bold green]Final Output[/bold green]", title_align="left", border_style="green"))
    return response_text

# Ask the user for the orchestrator model choice
orchestrator_model = input("Please choose the orchestrator model (Claude Opus or GPT-4): ")
while orchestrator_model not in ["Claude Opus", "GPT-4"]:
    orchestrator_model = input("Invalid choice. Please enter 'Claude Opus' or 'GPT-4': ")

# Get the objective from user input
objective = input("Please enter your objective with or without a text file path: ")

# Check if the input contains a file path
if "./" in objective or "/" in objective:
    # Extract the file path from the objective
    file_path = re.findall(r'[./\w]+\.[\w]+', objective)[0]
    # Read the file content
    file_content = read_file(file_path)
    # Update the objective string to remove the file path
    objective = objective.split(file_path)[0].strip()
else:
    file_content = None

task_exchanges = []
subagent_tasks = []

while True:
    # Call Orchestrator to break down the objective into the next sub-task or provide the final output
    previous_results = [result for _, result in task_exchanges]
    if not task_exchanges:
        # Pass the file content only in the first iteration if available
        opus_result, file_content_for_subagent = opus_orchestrator(objective, file_content, previous_results)
    else:
        opus_result, _ = opus_orchestrator(objective, previous_results=previous_results)

    if "The task is complete:" in opus_result:
        # If Opus indicates the task is complete, exit the loop
        final_output = opus_result.replace("The task is complete:", "").strip()
        break
    else:
        sub_task_prompt = opus_result
        # Include file content in the first subagent call if available
        if file_content_for_subagent and not subagent_tasks:
            sub_task_prompt += "\n\nFile content:\n" + file_content_for_subagent
        sub_task_result = subagent(sub_task_prompt, subagent_tasks)
        subagent_tasks.append(f"Task: {sub_task_prompt}\nResult: {sub_task_result}")
        task_exchanges.append((sub_task_prompt, sub_task_result))
        # Ensure file content is not passed in subsequent calls
        file_content_for_subagent = None

# Create the .md filename
sanitized_objective = re.sub(r'\W+', '_', objective)
timestamp = datetime.now().strftime("%H-%M-%S")

# Call Orchestrator to review and refine the sub-task results
refined_output = opus_refine(objective, [result for _, result in task_exchanges], timestamp, sanitized_objective)

# Extract the project name from the refined output
project_name_match = re.search(r'Project Name: (.*)', refined_output)
project_name = project_name_match.group(1).strip() if project_name_match else sanitized_objective

# Extract the folder structure from the refined output
folder_structure_match = re.search(r'<folder_structure>(.*?)</folder_structure>', refined_output, re.DOTALL)
folder_structure = {}
if folder_structure_match:
    json_string = folder_structure_match.group(1).strip()
    try:
        folder_structure = json.loads(json_string)
    except json.JSONDecodeError as e:
        console.print(Panel(f"Error parsing JSON: {e}", title="[bold red]JSON Parsing Error[/bold red]", title_align="left", border_style="red"))
        console.print(Panel(f"Invalid JSON string: [bold]{json_string}[/bold]", title="[bold red]Invalid JSON String[/bold red]", title_align="left", border_style="red"))

# Extract code files from the refined output
code_blocks = re.findall(r'Filename: (\S+)\s*```[\w]*\n(.*?)\n```', refined_output, re.DOTALL)

# Create the folder structure and code files
create_folder_structure(project_name, folder_structure, code_blocks)

# Truncate the sanitized_objective to a maximum of 50 characters
max_length = 40
truncated_objective = sanitized_objective[:max_length] if len(sanitized_objective) > max_length else sanitized_objective

# Update the filename to include the project name
filename = f"{timestamp}_{truncated_objective}.md"

# Prepare the full exchange log
exchange_log = f"Objective: {objective}\n\n"
exchange_log += "=" * 40 + " Task Breakdown " + "=" * 40 + "\n\n"
for i, (prompt, result) in enumerate(task_exchanges, start=1):
    exchange_log += f"Task {i}:\n"
    exchange_log += f"Prompt: {prompt}\n"
    exchange_log += f"Result: {result}\n\n"

exchange_log += "=" * 40 + " Refined Final Output " + "=" * 40 + "\n\n"
exchange_log += refined_output

console.print(f"\n[bold]Refined Final output:[/bold]\n{refined_output}")

with open(filename, 'w') as file:
    file.write(exchange_log)
print(f"\nFull exchange log saved to {filename}")
