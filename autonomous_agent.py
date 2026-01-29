#!/usr/bin/env python3

import asyncio
import time
from agent import Agent
from tools import doc_vector_tool, read_file_context_tool, terminal_tool, create_file_tool, add_task_tool, complete_task_tool,ask_to_user_tool
from llm import ollama_llm
from logger import logger, setup_logger
from kanban import kanban_instance

protocol="""
        1. **DECISION MAKING & AUTONOMY**:
           - If requirements are vague, ASK the user for clarification (e.g., "Which tech stack?").
           - Make architectural decisions based on best practices.
           - Once you have a plan, EXECUTE it step-by-step.

        2. **SELF-CORRECTION & DEBUGGING (MOST IMPORTANT)**:
           - You WILL encounter errors. This is normal.
           - If a command fails or code throws an error:
             a) READ the error message carefully.
             b) READ the file context to see the actual code.
             c) ANALYZE why it failed.
             d) FIX the code using `create_file` or run a command to install dependencies.
             e) RETRY.
           - DO NOT give up after one error. Try to fix it at least 3 times before asking for help.

        3. **CONTEXT MANAGEMENT**:
           - Do not rely solely on chat history. It may be truncated.
           - Always READ relevant files to understand the current state of the project before managing it.
           - Keep your internal reasoning concise to save context window tokens.

        4. **TESTING & FINALIZATION**:
           - Never assume code works. Run it! Write tests (e.g., `python main.py`, `npm test`).
           - When a feature is working and tested, use `git` to commit your changes:
             `git add .`
             `git commit -m "feat: implemented user login"`
             """

async def main():
    setup_logger()
    logger.info("Starting Agentic AI System")

    try:
        my_agent = Agent(
            role="Autonomous Lead Developer",
            rules="You are an autonomous agent. Manage your own tasks using the provided Context and Project State.",
            protocol=protocol,
            llm=ollama_llm,
            tools=[doc_vector_tool, read_file_context_tool, terminal_tool, create_file_tool, add_task_tool, complete_task_tool, ask_to_user_tool],
        )
    except Exception as e:
        logger.critical(f"Failed to initialize Agent: {e}")
        return

    try:
        while True:
            task_msg = kanban_instance.get_next_task()
            logger.info(f"Task: {task_msg}")
            
            context_str = f"""
            - Task: {task_msg}
            - Last State: {kanban_instance.load_state()}
            
            Perform the task. If you complete a sub-task, update the project state using tools.
            """
            
            result = await my_agent.work_until_done(context_str)
            logger.info(f"Result:\n{result}")

            
    except KeyboardInterrupt:
        logger.info("User initiated shutdown (KeyboardInterrupt)")

    exit(0)


if __name__ == "__main__":
    asyncio.run(main())
