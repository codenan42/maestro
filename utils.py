import os
import json
from rich.console import Console
from rich.panel import Panel

console = Console()

def read_file(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The content of the file.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def create_folder_structure(project_name: str, folder_structure: dict, code_blocks: list) -> None:
    """
    Creates the folder structure and code files based on the provided structure and code blocks.

    Args:
        project_name (str): The name of the project.
        folder_structure (dict): The folder structure as a dictionary.
        code_blocks (list): A list of tuples containing file names and their respective code content.
    """
    try:
        os.makedirs(project_name, exist_ok=True)
        console.print(Panel(f"Created project folder: [bold]{project_name}[/bold]", title="[bold green]Project Folder[/bold green]", title_align="left", border_style="green"))
    except OSError as e:
        console.print(Panel(f"Error creating project folder: [bold]{project_name}[/bold]\nError: {e}", title="[bold red]Project Folder Creation Error[/bold red]", title_align="left", border_style="red"))
        return

    create_folders_and_files(project_name, folder_structure, code_blocks)

def create_folders_and_files(current_path: str, structure: dict, code_blocks: list) -> None:
    """
    Recursively creates folders and files based on the provided structure and code blocks.

    Args:
        current_path (str): The current path to create folders and files in.
        structure (dict): The folder structure as a dictionary.
        code_blocks (list): A list of tuples containing file names and their respective code content.
    """
    for key, value in structure.items():
        path = os.path.join(current_path, key)
        if isinstance(value, dict):
            try:
                os.makedirs(path, exist_ok=True)
                console.print(Panel(f"Created folder: [bold]{path}[/bold]", title="[bold blue]Folder Creation[/bold blue]", title_align="left", border_style="blue"))
                create_folders_and_files(path, value, code_blocks)
            except OSError as e:
                console.print(Panel(f"Error creating folder: [bold]{path}[/bold]\nError: {e}", title="[bold red]Folder Creation Error[/bold red]", title_align="left", border_style="red"))
        else:
            code_content = next((code for file, code in code_blocks if file == key), None)
            if code_content:
                try:
                    with open(path, 'w') as file:
                        file.write(code_content)
                    console.print(Panel(f"Created file: [bold]{path}[/bold]", title="[bold green]File Creation[/bold green]", title_align="left", border_style="green"))
                except IOError as e:
                    console.print(Panel(f"Error creating file: [bold]{path}[/bold]\nError: {e}", title="[bold red]File Creation Error[/bold red]", title_align="left", border_style="red"))
            else:
                console.print(Panel(f"Code content not found for file: [bold]{key}[/bold]", title="[bold yellow]Missing Code Content[/bold yellow]", title_align="left", border_style="yellow"))
