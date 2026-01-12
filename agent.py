import time
import asyncio
from typing import List, Optional
from llama_index.core.agent import ReActAgent,AgentWorkflow,FunctionAgent
from llama_index.core.tools import BaseTool
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from settings import OLLAMA_URL
from llama_index.core import set_global_handler

set_global_handler("simple")


class Agent:
    def __init__(
        self,
        role: str,
        rules: str,
        tools: Optional[List[BaseTool]] = None,
        model_name: str = "qwen3:1.7b",
    ):
        self.role = role
        self.rules = rules
        self.tools = tools or []

        self.llm = Ollama(
            base_url=OLLAMA_URL,
            model=model_name,
            request_timeout=300.0,
            additional_kwargs={"num_ctx": 8192, "num_predict": 512},
        )

        self.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

        self.agent = FunctionAgent(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            memory=self.chat_memory,
            system_prompt=self._build_system_prompt(),
            streaming=True
        )

    def _build_system_prompt(self) -> str:
        tools_info = ""
        if self.tools:
            tools_info = "\n".join(
                [f"- {tool.metadata.name}: {tool.metadata.description}" for tool in self.tools]
            )

        return f"""
        Role: {self.role}
        Rules: {self.rules}
        
        You are an AI assistant equipped with specific tools.

        Your Tools: 
        {tools_info}

        If the user asks "what tools do you have?", list the tools defined above.
        If the user asks a question that requires a tool, use the tool directly via function calling.
        """
        
    async def work_until_done(self, task: str):
        try:
            return await self.agent.run(user_msg=task,memory=self.chat_memory)
        except Exception as e:
            return f"Error executing task: {str(e)}"