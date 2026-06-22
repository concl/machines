from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
import subprocess
import re
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
    

class HumanEvalProblem:
    def __init__(self, problem, test_cases_code, entry_point):
        self.problem = problem
        self.test_cases_code = test_cases_code
        self.entry_point = entry_point

    def execute(self, code):
        try:
            # Create a temporary file to hold the code
            temp_file = BASE_DIR / "sandbox" / "temp_code.py"
            with open(temp_file, "w") as f:
                f.write(code)
            
            # Execute the code in a subprocess
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10,  # Set a timeout for execution
            )

            # Clean up the temporary file
            os.remove(temp_file)

            if result.returncode != 0:
                return {"success": False, "error": result.stderr}
            else:
                return {"success": True, "output": result.stdout}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def evaluate_solution(self, solution_code):
        # Combine the solution code with the test cases
        combined_code = f"{solution_code}\n\n{self.test_cases_code}\n\nif __name__ == '__main__':\n    check({self.entry_point})"

        # Execute the combined code
        execution_result = self.execute(combined_code)

        if not execution_result["success"]:
            return {"success": False, "error": execution_result["error"]}

        # Check if the output contains any assertion errors
        if "AssertionError" in execution_result["output"]:
            return {"success": False, "error": "Test cases failed."}

        return {"success": True, "output": execution_result["output"]}


class HumanEvalSolver:
    def __init__(self):
        pass

    def solve_problem(self, problem):
        return "def solution():\n    pass  # Replace with your implementation"
