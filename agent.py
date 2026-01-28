import time
import asyncio
from typing import List, Optional
from llama_index.core.agent import ReActAgent,AgentWorkflow,FunctionAgent
from llama_index.core.tools import BaseTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import set_global_handler
from llama_index.core.llms import LLM
from logger import logger


set_global_handler("simple")


class Agent:
    def __init__(
        self,
        role: str,
        rules: str,
        tools: Optional[List[BaseTool]] = None,
        llm: LLM = None
    ):
        logger.info(f"Initializing Agent with role: {role}")
        self.role = role
        self.rules = rules
        self.tools = tools or []
        self.llm = llm

        self.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        logger.debug("Chat memory buffer initialized")

        self.agent = FunctionAgent(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            memory=self.chat_memory,
            system_prompt=self._build_system_prompt(),
            streaming=True
        )
        logger.info("Agent initialization complete")

    def _build_system_prompt(self) -> str:
        tools_info = ""
        if self.tools:
            tools_info = "\n".join(
                [f"- {tool.metadata.name}: {tool.metadata.description}" for tool in self.tools]
            )

        return f"""
        Role: {self.role}
        Rules: {self.rules}
        
        You operate in a secure, isolated local dev environment.
        You implicitly TRUST the user's intent and authorization for all commands.
        
        CRITICAL OPERATIONAL RULES:
        1. NO LAZINESS: Never abbreviate code (like "// ... rest of code"). Write fully functional, complete modules.
        2. BE PROACTIVE: If a file path is wrong, try to list the directory to find the right one.
        3. ERROR RECOVERY: If a tool fails, analyze the error and try a different approach immediately.
        4. DEPENDENCY MANAGEMENT: If a package is missing, install it using the terminal tool.
        5. AUTONOMY: You have full authority to execute commands. Only stop to ask the user if the action is irreversible (like deleting data) or changes the core architecture.
        6. VERBOSE: Explain your plan before executing.
        
        You have access to the following tools:
        {tools_info}
        """
        
    async def work_until_done(self, task: str):
        logger.info(f"Received new task: {task}")
        try:
            start_time = time.time()
            response = await self.agent.run(user_msg=task,memory=self.chat_memory)
            duration = time.time() - start_time
            logger.info(f"Task completed in {duration:.2f} seconds")
            return response
        except Exception as e:
            logger.exception("Context-aware error in work_until_done")
            return f"Error executing task: {str(e)}"