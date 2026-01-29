from typing import List, Optional
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from embedings import ollama_embedding
from llm import ollama_embeding_llm
import subprocess
import os
from llama_index.core.readers.base import BaseReader
from logger import logger
from kanban import kanban_instance

def run_terminal_command(command: str) -> str:
    """
    Run a terminal command safely.
    """
    logger.info(f"Agent executing command: {command}")

    try:
        if not command.strip():
            logger.warning("Agent attempted to run empty command")
            return "Empty command"
            
        # Run the command with shell=True to support redirection, pipes, etc.
        result = subprocess.run(
            command, shell=True, executable='/bin/bash', capture_output=True, text=True, timeout=300, cwd=os.getcwd()
        )

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
            logger.warning(f"Command '{command}' produced stderr: {result.stderr}")

        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
            logger.error(f"Command '{command}' failed with exit code: {result.returncode}")
        else:
            logger.debug(f"Command '{command}' execution successful.")

        return output

    except subprocess.TimeoutExpired:
        logger.error(f"Command '{command}' timed out")
        return "Command timed out after 30 seconds"
    except Exception as e:
        logger.exception(f"Exception during command execution: {command}")
        return f"Error running command: {str(e)}"


terminal_tool = FunctionTool.from_defaults(
    fn=run_terminal_command,
    name="run_terminal_command",
    description="Run all terminal commands with this tool.",
)

def search_doc_with_citiation(query: str) -> str:
    logger.info(f"Searching documents with query: {query}")
    try:
        my_file_extractors = {
            ".txt": LineNumberedReader(),
        }
        documents = SimpleDirectoryReader(
            ".", file_extractor=my_file_extractors, recursive=True
        ).load_data()
        index = VectorStoreIndex.from_documents(documents, embed_model=ollama_embedding)
        query_engine = index.as_query_engine(similarity_top_k=5, llm=ollama_embeding_llm)
        response = query_engine.query(query)

        final_output = f"{str(response)}\n\nSOURCES FOUND:"

        for node in response.source_nodes:
            score = f"{node.score:.2f}" if node.score else "N/A"
            final_output += f"\n- Metadata: {node.metadata} | Score: {score}"
        
        logger.debug(f"Search completed for query: {query}")
        return final_output
    except Exception as e:
        logger.exception(f"Error searching documents for query: {query}")
        return f"Error searching documents: {str(e)}"


doc_vector_tool = FunctionTool.from_defaults(
    fn=search_doc_with_citiation,
    name="search_doc_with_citiation",
    description="Search documents and returns the answer WITH page numbers.",
)

def read_file_context(file_path: str,start:int,end:int) -> str:
    logger.info(f"Reading file context: {file_path} lines {start}-{end}")
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return f"error file not dound -> {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        selected_lines = []
        for i in range(start, end):
            selected_lines.append(f"[L:{i+1}] {lines[i]}")
        
        content = "".join(selected_lines)
        logger.debug(f"Read {len(content)} characters from {file_path}")
        return content

    except Exception as e:
        logger.exception(f"Error reading file {file_path}")
        return f"Error while reading file: {str(e)}"
    
read_file_context_tool = FunctionTool.from_defaults(
    fn=read_file_context,
    name="read_file_context",
    description="Read specific lines from a file given the start and end line numbers.",
)    


class LineNumberedReader(BaseReader):
    def load_data(self, file, extra_info=None):
        tagged_text = ""
        file_path = str(file)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip():
                    tagged_text += f"[L:{i+1}] {line}"

        return [Document(text=tagged_text, metadata=extra_info or {})]

def create_file(file_path: str, content: str) -> str:
    """Create or overwrite a file with the given content."""
    logger.info(f"Creating file: {file_path}")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Successfully wrote {len(content)} chars to {file_path}")
        return f"File created successfully: {file_path}"
    except Exception as e:
        logger.exception(f"Error creating file {file_path}")
        return f"Error creating file: {str(e)}"

create_file_tool = FunctionTool.from_defaults(
    fn=create_file,
    name="create_file",
    description="Create or overwrite a file with the given content. Use this tool instead of shell commands to write files.",
)



add_task_tool = FunctionTool.from_defaults(
    fn=kanban_instance.add_task,
    name="add_task",
    description="Add a new task to the project's todo list. Use this to plan future work or break down large tasks."
)

complete_task_tool = FunctionTool.from_defaults(
    fn=kanban_instance.complete_current_task,
    name="complete_task",
    description="Mark the currently active task as done. You MUST call this with a summary result when you finish a task."
)

def ask_to_user(question: str) -> str:
    """Ask a question to the user and return their response."""
    logger.info(f"Asking user: {question}")
    try:
        response = input(f"{question}\nYour answer: ")
        return response
    except Exception as e:
        logger.exception("Error asking user.")
        return f"Error asking user: {str(e)}"

ask_to_user_tool = FunctionTool.from_defaults(
    fn=ask_to_user,
    name="ask_to_user",
    description="Ask a question to the user and get their input. Use this tool when you need clarification or additional information from the user."
)    
    

