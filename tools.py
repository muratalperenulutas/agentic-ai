from typing import List, Optional
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from embedings import ollama_embedding
from llm import ollama_llm
import subprocess
import shlex
import os
import random


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
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(
        documents, embed_model=ollama_embedding
    )
    query_engine = index.as_query_engine(similarity_top_k=3,llm=ollama_llm)
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
