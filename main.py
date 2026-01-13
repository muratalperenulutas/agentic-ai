#!/usr/bin/env python3

import asyncio
import sys
from agent import Agent
from tools import rand_number_tool, doc_vector_tool, read_file_context_tool,terminal_tool
from llm import ollama_llm


async def main():
    print("Welcome to agentic ai")

    my_agent = Agent(
        role="System Assistant",
        rules="You are helpful.",
        llm=ollama_llm,
        tools=[rand_number_tool,doc_vector_tool,read_file_context_tool,terminal_tool],
    )

    try:
        while True:
            command: str = input("Enter your command: ")
            if command:
                result = await my_agent.work_until_done(command)
                print("\n")
                print(result)
                print("\n")
            else:
                print("Please enter an valid command!")
    except KeyboardInterrupt:
        print("\nBye")

    exit(0)


if __name__ == "__main__":
    asyncio.run(main())
