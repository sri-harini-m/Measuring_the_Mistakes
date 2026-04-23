import fire
import sys
from evaluation_java import evaluate_functional_correctness_java

def entry_point(
    sample_file: str,
    timeout: float = 30.0,
    problem_file: str = "edit_eval_java.jsonl",
):
    evaluate_functional_correctness_java(
        sample_file=sample_file,
        timeout=timeout,
        problem_file=problem_file,
    )

if __name__ == "__main__":
    fire.Fire(entry_point)
