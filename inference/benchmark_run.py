from human_eval import HumanEvalProblem, HumanEvalSolver
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from argparse import ArgumentParser
import subprocess
import re
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class DeepSeekHumanEvalSolver(HumanEvalSolver):
    def __init__(self, model_name="deepseek-v4-pro"):
        super().__init__()

        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
        self.model_name = model_name
        self.prompt = (
            "Please solve the following Python programming problem with exactly the given signature. "
            "Provide a complete and correct implementation. "
            "Write your final solution in a single code block without any additional explanations or comments. "
            "If the problem includes any extra helper functions, include them in your final solution. "
        )

    def solve_problem(self, problem, reasoning_effort="high"):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": f"{self.prompt}\n\nProblem:\n{problem}"},
            ],
            logprobs=True,
            stream=False,
            reasoning_effort=reasoning_effort,
            extra_body={"thinking": {"type": "enabled"}},  # enabled/disabled/adaptive
        )

        raw_output = response.choices[0].message.content
        solution_code = self.extract_code(raw_output)
        return solution_code

    def extract_code(self, text):
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        else:
            return text.strip()


def main():
    load_dotenv()

    solver = DeepSeekHumanEvalSolver()

    dataset = load_dataset("openai_humaneval", split="test")

    total_problems = len(dataset)
    correct_solutions = 0

    for i, example in enumerate(dataset):

        problem = example["prompt"]
        test_cases_code = example["test"]
        entry_point = example["entry_point"]

        print(f"Problem {i + 1}:")
        print(problem)

        solution_code = solver.solve_problem(problem)
        print("Generated Solution:")
        print(solution_code)

        problem_instance = HumanEvalProblem(problem, test_cases_code, entry_point)
        evaluation_result = problem_instance.evaluate_solution(solution_code)
        print("Evaluation Result:")
        print(evaluation_result)
        print("-" * 80)
        if evaluation_result["success"]:
            correct_solutions += 1

    print(f"Total Problems: {total_problems}")
    print(f"Correct Solutions: {correct_solutions}")


if __name__ == "__main__":
    main()
