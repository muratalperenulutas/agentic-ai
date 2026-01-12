from typing import List, Optional, Any, Dict
from llama_index.llms.gemini import Gemini
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
import asyncio
import os
from config import GEMINI_API_KEY

class BasicAgent:
    def __init__(self, role: str, rules: str, tools: Optional[List[BaseTool]] = None):
        self.role = role
        self.rules = rules
        self.tools = tools or []

        # Initialize LLM
        self.llm = Gemini(api_key=GEMINI_API_KEY, model_name="gemini-2.5-flash")

        # Initialize memory with local in-memory vector store
        try:
            self.memory_index = VectorStoreIndex([], embed_model=GeminiEmbedding(api_key=GEMINI_API_KEY))
        except Exception as e:
            print(f"Warning: Could not initialize vector memory: {e}")
            self.memory_index = None

        self.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

        self.agent = ReActAgent(
            tools=self.tools,
            llm=self.llm,
            memory=self.chat_memory,
            verbose=True
        )

    async def run_prompt(self, task: str) -> str:
        prompt = f"""

        Task: {task}
        Role: {self.role}
        Rules: {self.rules}

        Please complete this task using your available tools and knowledge.
        Be thorough and provide a clear result.
        """
        try:
            response = await self.agent.run(prompt)
            return str(response)
        except Exception as e:
            return f"Error executing task: {str(e)}"