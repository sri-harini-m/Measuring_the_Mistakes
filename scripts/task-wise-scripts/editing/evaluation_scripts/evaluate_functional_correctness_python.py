import fire
import sys

from evaluation_python import evaluate_functional_correctness_python


def entry_point(
    sample_file: str,
    timeout: float = 30.0,
    problem_file: str = "edit_eval_python.jsonl",
):
    results = evaluate_functional_correctness_python(sample_file, timeout, problem_file)
    print(results)


def main():
    fire.Fire(entry_point)


if __name__ == '__main__':
    sys.exit(main())
