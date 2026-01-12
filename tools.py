from typing import List, Optional
from llama_index.core.tools import BaseTool, FunctionTool
import subprocess
import shlex
import os

def run_terminal_command(command: str) -> str:
    """
    Run a terminal command safely.
    """

    try:
        # Parse command
        parts = shlex.split(command)
        if not parts:
            return "Empty command"
        # Run the command
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"

        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"

        return output

    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {str(e)}"

terminal_tool = FunctionTool.from_defaults(
    fn=run_terminal_command,
    name="run_terminal_command",
    description="Run all terminal commands with this tool."
)