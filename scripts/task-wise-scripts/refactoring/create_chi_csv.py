#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CODEEDITORBENCH_DIR = ROOT / "output" / "codeeditorbench"
PROBLEMS_JSONL = ROOT / "refactoring" / "input" / "codeeditorbench" / "problems.jsonl"
OUT_CSV = ROOT / "output" / "chi_input_codeeditorbench_all_models.csv"

LANG_MAP = {
    "python": "python",
    "python3": "python",
    "java": "java",
    "cpp": "cpp",
    "c++": "cpp",
    "c": "cpp",
}


def normalize_language(language: str) -> str:
    if not language:
        return ""
    return LANG_MAP.get(language.strip().lower(), "")


def build_test_cases(problem: Dict) -> List[Dict[str, str]]:
    test_cases: List[Dict[str, str]] = []

    pub_in = problem.get("public_tests_input", "")
    pub_out = problem.get("public_tests_output", "")
    if pub_in is not None and pub_out is not None and str(pub_in) != "" and str(pub_out) != "":
        test_cases.append({"input": str(pub_in), "output": str(pub_out)})

    priv_ins = problem.get("private_tests_input", [])
    priv_outs = problem.get("private_tests_output", [])
    if isinstance(priv_ins, list) and isinstance(priv_outs, list):
        for pi, po in zip(priv_ins, priv_outs):
            test_cases.append({"input": str(pi), "output": str(po)})

    return test_cases


def load_problem_tests() -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    if not PROBLEMS_JSONL.exists():
        raise FileNotFoundError(f"Missing dataset file: {PROBLEMS_JSONL}")

    problem_tests: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    with PROBLEMS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = str(row.get("idx", "")).strip()
            language = normalize_language(str(row.get("code_language", "")))
            if not idx or not language:
                continue
            tests = build_test_cases(row)
            if tests:
                problem_tests[(idx, language)] = tests
    return problem_tests


def iter_solution_files() -> List[Path]:
    if not OUTPUT_CODEEDITORBENCH_DIR.exists():
        raise FileNotFoundError(f"Missing output directory: {OUTPUT_CODEEDITORBENCH_DIR}")
    return sorted(OUTPUT_CODEEDITORBENCH_DIR.glob("*/refactored_solutions_*.jsonl"))


def language_from_solution_filename(path: Path) -> str:
    name = path.stem
    lang = name.replace("refactored_solutions_", "", 1)
    return normalize_language(lang)


def main() -> None:
    problem_tests = load_problem_tests()
    solution_files = iter_solution_files()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_idx = 0
    skipped_no_code = 0
    skipped_no_tests = 0

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "language", "code", "test_cases", "problem_idx"],
        )
        writer.writeheader()

        for solution_file in solution_files:
            model_name = solution_file.parent.name
            file_language = language_from_solution_filename(solution_file)

            with solution_file.open("r", encoding="utf-8") as sf:
                for line in sf:
                    if not line.strip():
                        continue
                    row = json.loads(line)

                    idx = str(row.get("idx", "")).strip()
                    if not idx:
                        skipped_no_idx += 1
                        continue

                    language = normalize_language(str(row.get("code_language", ""))) or file_language
                    code = row.get("refactored_solution", "")
                    if not code:
                        skipped_no_code += 1
                        continue

                    tests = problem_tests.get((idx, language), [])
                    if not tests:
                        skipped_no_tests += 1
                        continue

                    writer.writerow(
                        {
                            "model_name": model_name,
                            "language": language,
                            "code": code,
                            "test_cases": json.dumps(tests, ensure_ascii=False),
                            "problem_idx": idx,
                        }
                    )
                    written += 1

    print(f"Wrote {written} rows to {OUT_CSV}")
    print(
        "Skipped rows: "
        f"no_idx={skipped_no_idx}, "
        f"no_code={skipped_no_code}, "
        f"no_tests={skipped_no_tests}"
    )


if __name__ == "__main__":
    main()
