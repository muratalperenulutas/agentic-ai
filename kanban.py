import json
import os

PROJECT_STATE_FILE = "project_state.json"

KANBAN_STATE = {
    "todo": [],
    "in_progress": [],
    "done": [],
    "last_error": None,
    "current_goal": ""
}

class Kanban:
    def __init__(self):
        self.state = self.load_state()
        if not self.state["todo"] and not self.state["in_progress"]:
            self.state["todo"].append("Analyze current workspace and create a plan")
            self.save_state(self.state)

    def load_state(self):
        if os.path.exists(PROJECT_STATE_FILE):
                with open(PROJECT_STATE_FILE, "r") as f:
                    return json.load(f)
        return KANBAN_STATE

    def save_state(self, state):
        with open(PROJECT_STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    
    def get_next_task(self):
        if self.state["in_progress"]:
            return f"Continue working on: {self.state['in_progress'][0]}"
        if self.state["todo"]:
            task = self.state["todo"].pop(0)
            self.state["in_progress"].append(task)
            self.save_state(self.state)
            return f"Start working on: {task}"
        return "Check project status and suggest new tasks based on architecture and project goals."
    
    def add_task(self, task: str):
        """Add a new task to the todo list."""
        if task not in self.state["todo"] and task not in self.state["in_progress"] and task not in self.state["done"]:
            self.state["todo"].append(task)
            self.save_state(self.state)
            return f"Task added to todo: {task}"
        return f"Task already exists: {task}"
    
    def complete_current_task(self, result_summary: str):
        """Mark the current in-progress task as done."""
        if self.state["in_progress"]:
            task = self.state["in_progress"].pop(0)
            self.state["done"].append({"task": task, "result": result_summary})
            self.save_state(self.state)
            return f"Task completed: {task}"
        return "No task in progress to complete."

kanban_instance = Kanban()

