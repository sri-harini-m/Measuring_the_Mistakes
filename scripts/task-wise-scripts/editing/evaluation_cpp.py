import sys
sys.path.append('.')

from collections import defaultdict
from typing import Dict
import json
import os
import numpy as np
import tqdm

from data import read_problems, stream_jsonl
from execution_cpp import check_correctness_cpp

CODE_MARKER = r"{{Code}}"


def load_expected_assert_totals(problem_file: str) -> Dict[str, int]:
    """Load expected assert_total counts from prior eval outputs."""
    expected: Dict[str, int] = {}
    
    results_jsonl = problem_file.replace('.jsonl', '.jsonl_results.jsonl')
    if os.path.exists(results_jsonl):
        try:
            for row in stream_jsonl(results_jsonl):
                task_id = row.get('task_id')
                assert_total = row.get('assert_total')
                if task_id and assert_total is not None:
                    # Keep the max seen value per task to guard against partial runs
                    prev = expected.get(task_id, 0)
                    expected[task_id] = max(prev, int(assert_total))
            print(f"Loaded expected assert_total values for {len(expected)} datapoints from {results_jsonl}")
        except Exception as e:
            print(f"Warning: Could not load results file {results_jsonl}: {e}")

    return expected

def strip_leading_code_fence(code: str, lang: str) -> str:
    """
    If the snippet begins with a fenced block like ```{lang},
    return the inner code without the opening/closing fences.
    """
    if not isinstance(code, str):
        return code
    s = code.lstrip()
    prefix = f"```{lang}"
    if s.startswith(prefix):
        parts = s.split('\n', 1)
        body = parts[1] if len(parts) > 1 else ''
        end_idx = body.rfind('```')
        if end_idx != -1 and body[end_idx+3:].strip() == '':
            body = body[:end_idx]
        return body.strip()
    return code


def evaluate_functional_correctness_cpp(
    sample_file: str,
    timeout: float = 3.0,
    problem_file: str = "edit_eval_cpp.jsonl",
):
    """
    Evaluates the functional correctness of generated C++ samples.
    """  
    problems = read_problems(problem_file)
    
    expected_assert_totals = load_expected_assert_totals(problem_file)
    
    samples = list(stream_jsonl(sample_file))
    
    n_samples = len(samples)
    results = defaultdict(list)
    
    print(f"Evaluating {n_samples} samples...")
    
    run_results = defaultdict(list)
    
    for sample in tqdm.tqdm(samples):
        task_id = sample["task_id"]
        code = sample.get("output") or sample.get("completion", "")
        code = strip_leading_code_fence(code, 'cpp')
        problem = problems[task_id]
        
        res = check_correctness_cpp(problem, code, timeout, None)
        run_results[(task_id, code)].append(res)
    
    for (task_id, code), res_list in run_results.items():
            res = res_list[0]
            passed = res["passed"]
            
            assert_total = res.get("assert_total", 0)
            assert_passed = res.get("assert_passed", 0)
            observed_assert_total = assert_total
            
            if task_id == "EditEval/37":
                stderr = res.get("error", [])
                stderr_text = "\n".join(stderr) if isinstance(stderr, list) else str(stderr)
                
                assert_total = 2
                assert_passed = 0
                
                if "Correct candidate failed" not in stderr_text:
                    assert_passed += 1
                
                if "did not throw as expected" not in stderr_text:
                    assert_passed += 1
                
                passed = (assert_passed == assert_total)
            

            if not passed and task_id in expected_assert_totals:
                expected_total = expected_assert_totals[task_id]
                if expected_total > assert_total:
                    assert_total = expected_total
            
            results[task_id].append({
                "task_id": task_id,
                "passed": passed,
                "code": code,
                "assert_passed": assert_passed,
                "observed_assert_total": observed_assert_total,
                "assert_total": assert_total,
                "result": res["result"],
                "error": res.get("error", ""),
                "exec_time_s": res.get("exec_time_s", 0.0),
                "peak_mem_mb": res.get("peak_mem_mb", 0.0),
            })

    # Calculate metrics
    total = []
    correct = []
    all_codes = []
    
    for task_id in results:
        task_results = results[task_id]
        total.append(len(task_results))
        correct.append(sum(1 for r in task_results if r["passed"]))
        
        for r in task_results:
            code = r.get("code", "")
            if code and code.strip():
                all_codes.append(code)

    total = np.array(total)
    correct = np.array(correct)
    
    pass_rate = correct.sum() / total.sum() if total.sum() > 0 else 0
    hallucination_rate = 1.0 - pass_rate
    
    tcpr_sum = 0.0
    per_datapoint_count = 0
    for task_id in results:
        for r in results[task_id]:
            assert_total_val = int(r.get("assert_total", 0))
            assert_passed_val = int(r.get("assert_passed", 0))
            if assert_total_val > 0:
                tcpr_sum += assert_passed_val / assert_total_val
                per_datapoint_count += 1
    
    tcpr = tcpr_sum / per_datapoint_count if per_datapoint_count > 0 else 0.0

    metrics = {
        "status": "success",
        "pass_rate": pass_rate,
        "hallucination_rate": hallucination_rate,
        "tcpr": tcpr,

    }
    
    metrics_file = sample_file + "_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {metrics_file}")
    
    # Write detailed results
    results_file = sample_file + "_results.jsonl"
    with open(results_file, "w") as f:
        for task_id, task_res in results.items():
            for res in task_res:
                f.write(json.dumps(res) + "\n")
                
    return metrics
