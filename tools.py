from asyncio.log import logger
from typing import List, Optional
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from embedings import ollama_embedding
from llm import ollama_embeding_llm
import subprocess
import shlex
import os
import random
from llama_index.core.readers.base import BaseReader


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
            parts, capture_output=True, text=True, timeout=30, cwd=os.getcwd()
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
    description="Run all terminal commands with this tool.",
)


def get_rand_number() -> str:
    rand = str(random.randint(1, 1000))
    print(f"Generated random number: {rand}")
    return rand


rand_number_tool = FunctionTool.from_defaults(
    fn=get_rand_number, name="get_random_number", description="Generate a random number"
)


def search_doc_with_citiation(query: str) -> str:
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

    return final_output


doc_vector_tool = FunctionTool.from_defaults(
    fn=search_doc_with_citiation,
    name="search_doc_with_citiation",
    description="Search documents and returns the answer WITH page numbers.",
)

def read_file_context(file_path: str,start:int,end:int) -> str:
    try:
        if not os.path.exists(file_path):
            return f"error file not dound -> {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        selected_lines = []
        for i in range(start, end):
            selected_lines.append(f"[L:{i+1}] {lines[i]}")

        return "".join(selected_lines)

    except Exception as e:
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

def list_files(path: str = ".") -> str:
    """List all files in a directory."""
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

list_files_tool = FunctionTool.from_defaults(
    fn=list_files,
    name="list_files",
    description="List files in the current directory or a specific path.",
)

