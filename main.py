#!/usr/bin/env python3

import asyncio
import sys
from agent import Agent
from tools import rand_number_tool, doc_vector_tool, read_file_context_tool, terminal_tool, list_files_tool
from llm import ollama_llm
from logger import logger, setup_logger

async def main():
    setup_logger()
    logger.info("Starting Agentic AI System")

    try:
        my_agent = Agent(
            role="System Assistant",
            rules="You are helpful.",
            llm=ollama_llm,
            tools=[rand_number_tool, doc_vector_tool, read_file_context_tool, terminal_tool, list_files_tool],
        )
    except Exception as e:
        logger.critical(f"Failed to initialize Agent: {e}")
        return

    try:
        while True:
            command: str = input("Enter your command: ")
            if command:
                logger.debug(f"User input command: {command}")
                result = await my_agent.work_until_done(command)
                logger.info(f"Result:\n{result}")
            else:
                logger.warning("Empty command received from user")
    except KeyboardInterrupt:
        logger.info("User initiated shutdown (KeyboardInterrupt)")

    exit(0)


if __name__ == "__main__":
    asyncio.run(main())
