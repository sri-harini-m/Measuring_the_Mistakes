import fire
import sys
from evaluation_cpp import evaluate_functional_correctness_cpp

def entry_point(
    sample_file: str,
    timeout: float = 30.0,
    problem_file: str = "edit_eval_cpp.jsonl",
):
    results = evaluate_functional_correctness_cpp(sample_file, timeout, problem_file)
    print(results)

def main():
    fire.Fire(entry_point)

if __name__ == "__main__":
    sys.exit(main())
