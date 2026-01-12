#!/usr/bin/env python3

import asyncio
import sys
from agent import BasicAgent
import tools

async def main():
    print("Welcome to agentic ai")

    my_agent = BasicAgent(role="System Assistant", rules="You are helpful.", tools=[tools.terminal_tool])

    try:
        while True:
            command:str = input("Enter your command: ")
            if command:
                result = await my_agent.run_prompt(command)
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