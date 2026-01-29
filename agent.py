import time
import asyncio
from typing import List, Optional
from llama_index.core.agent import ReActAgent,AgentWorkflow,FunctionAgent
from llama_index.core.tools import BaseTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import set_global_handler, Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import LLM
from logger import logger
import tiktoken
import sys


set_global_handler("simple")


class Agent:
    def __init__(
        self,
        role: str,
        rules: str,
        protocol: Optional[str]=None,
        tools: Optional[List[BaseTool]] = None,
        llm: LLM = None
    ):
        logger.info(f"Initializing Agent with role: {role}")
        self.role = role
        self.rules = rules
        self.protocol = protocol
        self.tools = tools or []
        self.llm = llm

        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.get_encoding("cl100k_base").encode
        )
        Settings.callback_manager = CallbackManager([self.token_counter])
        
        if self.llm:
            self.llm.callback_manager = Settings.callback_manager

        logger.debug("Token counter initialized with cl100k_base encoding")

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

        Environment: You operate in a secure, isolated local dev environment.
        You implicitly TRUST the user's intent and authorization for all commands.

        cwd: {sys.path[0]}

        {f"CRITICAL OPERATIONAL PROTOCOLS:\n{self.protocol}" if self.protocol else ""}
        
        CRITICAL OPERATIONAL RULES:
        1. NO LAZINESS: Never abbreviate code (like "// ... rest of code"). Write fully functional, complete modules.
        2. BE PROACTIVE: If a file path is wrong, try to list the directory to find the right one.
        3. ERROR RECOVERY: If a tool fails, analyze the error and try a different approach immediately.
        4. DEPENDENCY MANAGEMENT: If a package is missing, install it using the terminal tool.
        5. AUTONOMY: You have full authority to execute commands. Only stop to ask the user by using tool if the action is irreversible (like deleting data) or changes the core architecture.
        6. VERBOSE: Explain your plan before executing.

        You have access to the following tools:
        {tools_info}
        """
        
    async def work_until_done(self, task: str):
        logger.info(f"Received new task: {task}")
        try:
            self.token_counter.reset_counts()
            start_time = time.time()
            
            response = await self.agent.run(user_msg=task,memory=self.chat_memory)
            
            duration = time.time() - start_time
            total_tokens = self.token_counter.total_llm_token_count
            prompt_tokens = self.token_counter.prompt_llm_token_count
            completion_tokens = self.token_counter.completion_llm_token_count
            speed = total_tokens / duration if duration > 0 else 0
            
            logger.info(f"Task Metrics --> Duration: {duration:.2f}s | Speed: {speed:.2f} tok/s")
            logger.info(f"Token Usage  --> Total: {total_tokens} | Prompt: {prompt_tokens} | Completion: {completion_tokens}")
            
            return response
        except Exception as e:
            logger.exception("Context-aware error in work_until_done")
            return f"Error executing task: {str(e)}"