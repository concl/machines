
from templates.template import *

class MathTester:
    """
    A class that evaluates an LLM on a math problem dataset with 
    single numerical answers.

    Assumes a solver API that includes a "solve_problem" method which takes a problem statement and returns an integer answer.
    """

    def __init__(self, solver):
        self.solver = solver

    def evaluate(self, math_dataset: pd.DataFrame, return_answers=False):
        """
        Evaluates the model on the math dataset and returns a DataFrame with the results.
        Assumes the math_dataset has a "problem" column with the problem statements and an "answer" column with the correct answers.

        Args:
            math_dataset (pd.DataFrame): A DataFrame containing the math problems and their correct answers.
            return_answers (bool): Whether to include the model's answers in the returned DataFrame. Defaults to False.
        Returns:
            dict: A dictionary containing the evaluation results, including accuracy and optionally the model's answers.
        """

        correct_answers = 0
        total_problems = len(math_dataset)
        model_answers = []

        for _, row in math_dataset.iterrows():
            problem = row["problem"]
            correct_answer = row["answer"]

            # Get the model's answer using the solve_problem method
            model_answer = self.solver.solve_problem(problem)
            model_answers.append(model_answer)

            if model_answer == correct_answer:
                correct_answers += 1

        accuracy = correct_answers / total_problems if total_problems > 0 else 0
        results = {"accuracy": accuracy, "correct_answers": correct_answers, "total_problems": total_problems}

        if return_answers:
            results["model_answers"] = model_answers

        return results

class PythonSandbox:
    """
    A simple Python sandbox for executing code safely.
    """
    def __init__(self):
        pass

class MathProblemSolver:
    """
    A math problem solving pipeline that uses an LLM to solve math problems.
    """
    def __init__(self):
        pass
