import sys
sys.path.append('.')

from collections import defaultdict
from typing import List
import json
import tqdm

from data import read_problems, stream_jsonl
from execution_java import check_correctness_java

CODE_MARKER = r"{{Code}}"

def _load_baseline_assert_totals(path: str) -> dict:
    """Load assert_total per task from a prior results JSONL file if available."""
    totals = {}
    try:
        for row in stream_jsonl(path):
            tid = row.get("task_id")
            if tid is None:
                continue
            total = row.get("assert_total")
            if isinstance(total, int) and total > 0:
                totals[tid] = total
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return totals


def strip_code_block(code: str) -> str:
    """
    Strips markdown code block formatting from code.
    Handles cases with and without language identifiers.
    """
    if not isinstance(code, str) or not code:
        return code
    
    code = code.strip()
    
    if code.startswith('```'):
        lines = code.split('\n')
        lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        code = '\n'.join(lines)
    
    return code.strip()


def inject_edited_code_java(code: str) -> str:
    escaped_code = code.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    
    last_brace = code.rfind("}")
    if last_brace == -1:
        return code
        
    injection = f'\n    public static String edited_code = "{escaped_code}";\n'
    
    return code[:last_brace] + injection + code[last_brace:]

def evaluate_functional_correctness_java(
    sample_file: str,
    timeout: float = 3.0,
    problem_file: str = "edit_eval_java.jsonl",
):
    """Evaluate Java samples against EditEval tests."""

    # Always run sequentially to avoid thread overhead / nondeterminism.
    problems = read_problems(problem_file)
    samples = list(stream_jsonl(sample_file))

    baseline_totals = _load_baseline_assert_totals("edit_eval_java.jsonl_results.jsonl")

    n_samples = len(samples)
    results = defaultdict(list)

    print(f"Evaluating {n_samples} samples...")

    run_results = defaultdict(list)

    # Sequential execution
    for sample in tqdm.tqdm(samples):
        task_id = sample["task_id"]
        code = sample.get("output") or sample.get("completion", "")
        code = strip_code_block(code)
        problem = problems[task_id]

        if "context" in problem and CODE_MARKER in problem["context"]:
            context = problem["context"]
            if "edited_code =" not in context:
                code = context.replace(CODE_MARKER, code)

        code = inject_edited_code_java(code)

        if not code or not str(code).strip():
            continue

        try:
            res = check_correctness_java(problem, code, timeout, None)
        except Exception as exc:
            res = {
                "passed": False,
                "result": "",
                "error": str(exc),
                "assert_total": 0,
                "assert_passed": 0,
            }
        run_results[(task_id, code)].append(res)

    for (task_id, code), res_list in run_results.items():
        res0 = res_list[0]
        passed = res0.get("passed", False)
        assert_total = res0.get("assert_total", 0)
        assert_passed = res0.get("assert_passed", 0)
        observed_assert_total = assert_total

        if not passed and assert_total == 0:
            baseline_total = baseline_totals.get(task_id)
            if isinstance(baseline_total, int) and baseline_total > 0:
                assert_total = baseline_total
            elif task_id in problems:
                problem = problems[task_id]
                problem_assert_total = problem.get("assert_total", 0)
                if problem_assert_total > 0:
                    assert_total = problem_assert_total
        
        if assert_total == 0 and passed:
            assert_total = 1
            assert_passed = 1

        results[task_id].append({
            "task_id": task_id,
            "passed": passed,
            "code": code,
            "assert_passed": assert_passed,
            "observed_assert_total": observed_assert_total,
            "assert_total": assert_total,
            "result": res0.get("result", ""),
            "error": res0.get("error", ""),
            "exec_time_s": res0.get("exec_time_s", 0.0),
            "peak_mem_mb": res0.get("peak_mem_mb", 0.0),
        })
    total_runs = sum(len(task_res) for task_res in results.values())
    correct_runs = sum(1 for task_res in results.values() for r in task_res if r["passed"])

    pass_rate = correct_runs / total_runs if total_runs > 0 else 0.0
    hallucination_rate = 1.0 - pass_rate

    tcpr_sum = 0.0
    count = 0
    for task_id in results:
        for r in results[task_id]:
            at = int(r.get("assert_total", 0))
            ap = int(r.get("assert_passed", 0))
            if at > 0:
                tcpr_sum += ap / at
                count += 1
    tcpr = tcpr_sum / count if count > 0 else 0.0

    metrics = {
        "status": "success",
        "pass_rate": pass_rate,
        "hallucination_rate": hallucination_rate,
        "tcpr": tcpr,
    }

    metrics_file = sample_file + "_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {metrics_file}")

    results_file = sample_file + "_results.jsonl"
    with open(results_file, "w", encoding="utf-8") as f:
        for task_id, task_res in results.items():
            for res in task_res:
                f.write(json.dumps(res) + "\n")
    print(f"Results written to {results_file}")

    return metrics


