import sys
sys.path.append('.')

from collections import defaultdict, Counter
from typing import Dict
import itertools
import json
import os
import multiprocessing
import psutil
import time
import traceback

import numpy as np
import tqdm

from data import read_problems, stream_jsonl

CODE_MARKER = r"{{Code}}"

def load_expected_assert_totals(problem_file: str) -> Dict[str, int]:
    """Load expected assert_total counts from prior eval outputs."""
    expected: Dict[str, int] = {}
    
    results_jsonl = problem_file + "_results.jsonl"
    if os.path.exists(results_jsonl):
        try:
            for row in stream_jsonl(results_jsonl):
                task_id = row.get('task_id')
                assert_total = row.get('assert_total')
                if task_id and assert_total is not None:
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

def build_program_and_tests(problem, code):
    if "context" in problem.keys() and CODE_MARKER in problem["context"]:
        code = problem["context"].replace(CODE_MARKER, code)

    return (
        code + "\n\n" +
        problem["test"] + "\n\n" +
        f"check()"
    ).strip()


def _detect_indent_unit(lines: list) -> str:
    """Detect the indentation unit used in the code (2 spaces, 4 spaces, or tabs)."""
    for line in lines:
        stripped = line.lstrip()
        if stripped and line[0] in ' \t':
            indent = line[: len(line) - len(stripped)]
            if indent.startswith("  ") and not indent.startswith("    "):
                return "  "  # 2 spaces
            elif indent.startswith("\t"):
                return "\t"  # tab
    return "    "  # default 4 spaces


def _find_enclosing_try(lines: list, line_idx: int, current_indent: str) -> tuple:
    """Find the nearest enclosing try block at or above line_idx with same or less indentation."""
    for j in range(line_idx - 1, -1, -1):
        s = lines[j].lstrip()
        ind = lines[j][: len(lines[j]) - len(s)]
        if s.startswith("try:") and len(ind) <= len(current_indent):
            return j, ind
    return None, None

def _update_orig_to_out_idx(orig_to_out_idx: dict, start_idx: int, end_idx: int, delta: int) -> None:
    """Update all indices in orig_to_out_idx for lines after insertion point."""
    for k in range(start_idx, end_idx):
        if k in orig_to_out_idx:
            orig_to_out_idx[k] += delta


def _apply_pending_insertions(out_lines: list, orig_to_out_idx: dict, lines: list, 
                              current_idx: int, pending_flags: dict, pending_counters: dict) -> None:
    """Apply any pending flag or counter insertions scheduled for this line index."""
    if current_idx in pending_flags:
        for flag_line in pending_flags[current_idx]:
            out_lines.append(flag_line)
        _update_orig_to_out_idx(orig_to_out_idx, current_idx + 1, len(lines), len(pending_flags[current_idx]))
        del pending_flags[current_idx]

    if current_idx in pending_counters:
        for cl in pending_counters[current_idx]:
            out_lines.append(cl)
        _update_orig_to_out_idx(orig_to_out_idx, current_idx + 1, len(lines), len(pending_counters[current_idx]))
        del pending_counters[current_idx]


def wrap_asserts_in_code(test_code: str) -> str:
    """
    Replace top-level `assert` lines with try/except blocks that
    increment __assert_total__ and __assert_passed__ counters in the
    execution globals. Preserves indentation.
    Only catches AssertionError to let other exceptions propagate.
    
    Handles nested functions by tracking indentation levels.
    All functions that contain asserts or nested functions need global declarations.
    """
    lines = test_code.splitlines()
    out_lines = []
    out_lines.append("__assert_total__ = 0")
    out_lines.append("__assert_passed__ = 0")
    out_lines.append("__assert_errors__ = []")
    
    indent_unit = _detect_indent_unit(lines)
    
    function_stack = []
    orig_to_out_idx = {}

    pending_flags = {}
    pending_counters = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        current_indent = line[: len(line) - len(stripped)]

        orig_to_out_idx[i] = len(out_lines)

        if stripped.startswith("def "):
            function_stack = [f for f in function_stack if len(f['indent']) < len(current_indent)]
            function_stack.append({
                'indent': current_indent,
                'added_global': False,
                'line_index': len(out_lines) + 1,
            })
            out_lines.append(line)
            i += 1
            continue

        if function_stack and stripped and not stripped.startswith("#"):
            for func in function_stack:
                if not func['added_global']:
                    insert_pos = func['line_index']
                    global_decl = f"{func['indent']}{indent_unit}global __assert_total__, __assert_passed__"
                    out_lines.insert(insert_pos, global_decl)
                    for f in function_stack:
                        if f['line_index'] >= insert_pos:
                            f['line_index'] += 1
                    func['added_global'] = True

        if stripped.startswith("assert"):
            indent = current_indent
            out_lines.append(f"{indent}__assert_total__ += 1")
            out_lines.append(f"{indent}try:")
            out_lines.append(f"{indent}{indent_unit}{stripped}")
            out_lines.append(f"{indent}{indent_unit}__assert_passed__ += 1")
            out_lines.append(
                f"{indent}except AssertionError as e:\n"
                f"{indent}{indent_unit}__assert_errors__.append(str(e) if str(e) else 'AssertionError')\n"
                f"{indent}{indent_unit}pass"
            )
            i += 1
            continue

        if stripped.startswith("raise AssertionError"):
            try_idx, try_indent = _find_enclosing_try(lines, i, current_indent)

            indent = current_indent
            out_lines.append(f"{indent}__assert_total__ += 1")
            # Capture raised AssertionError message similarly
            out_lines.append(f"{indent}try:")
            out_lines.append(f"{indent}{indent_unit}{stripped}")
            out_lines.append(
                f"{indent}except AssertionError as e:\n"
                f"{indent}{indent_unit}__assert_errors__.append(str(e) if str(e) else 'AssertionError')\n"
                f"{indent}{indent_unit}pass"
            )
            i += 1
            continue

        out_lines.append(line)

        _apply_pending_insertions(out_lines, orig_to_out_idx, lines, i, pending_flags, pending_counters)

        i += 1

    for key in sorted(pending_flags.keys()):
        for flag_line in pending_flags[key]:
            out_lines.append(flag_line)
    for key in sorted(pending_counters.keys()):
        for cl in pending_counters[key]:
            out_lines.append(cl)

    return "\n".join(out_lines)


def run_candidate(candidate_code, timeout=3.0, extract_counters=True):
    """Execute candidate code and optionally extract instrumentation counters.

    Mirrors the time/memory monitoring used in chi_updated_script.py by sampling
    the child process RSS via psutil while it runs.
    """
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    def target(res):
        try:
            exec_globals = {}
            start = time.monotonic()
            exec(candidate_code, exec_globals)
            elapsed = time.monotonic() - start
            res["suite_passed"] = True
            res["exec_time_s"] = elapsed
            # Collect assertion errors captured by instrumentation
            if extract_counters:
                try:
                    errs = exec_globals.get("__assert_errors__", [])
                    if isinstance(errs, list):
                        res["error"] = list(errs)
                    else:
                        res["error"] = []
                except Exception:
                    res["error"] = []
            else:
                res["error"] = []
        except Exception:
            elapsed = time.monotonic() - start if "start" in locals() else 0.0
            res["exec_time_s"] = elapsed
            res["suite_passed"] = False
            # Capture traceback for runtime errors
            try:
                tb = traceback.format_exc()
                res["error"] = [tb]
            except Exception:
                res["error"] = ["ExecutionError"]
        
        if extract_counters:
            try:
                res["assert_passed"] = int(exec_globals.get("__assert_passed__", 0))
                res["assert_total"] = int(exec_globals.get("__assert_total__", 0))
            except Exception:
                res["assert_passed"] = 0
                res["assert_total"] = 0

    p = multiprocessing.Process(target=target, args=(result_dict,))
    p.start()

    t_start = time.monotonic()
    peak_mem = 0.0
    try:
        proc = psutil.Process(p.pid)
    except Exception:
        proc = None

    timed_out = False
    while True:
        if not p.is_alive():
            break

        if proc is not None:
            try:
                mem = proc.memory_info().rss
                peak_mem = max(peak_mem, mem)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                proc = None

        elapsed = time.monotonic() - t_start
        if elapsed >= timeout:
            timed_out = True
            break
        time.sleep(0.001)

    if p.is_alive():
        p.terminate()
        p.join(timeout=0.1)

    res = dict(result_dict) if result_dict else {}

    # Ensure timeout reported as error and counters defaulted
    if timed_out:
        res.setdefault("suite_passed", False)
        res.setdefault("assert_passed", 0)
        res.setdefault("assert_total", 0)
        errs = res.get("error") or []
        if not isinstance(errs, list):
            errs = [str(errs)]
        errs.append("TimeError")
        res["error"] = errs

    # Attach timing and peak memory (MB) mirroring chi_updated_script.py sampling
    res["exec_time_s"] = time.monotonic() - t_start if "exec_time_s" not in res else res["exec_time_s"]
    res["peak_mem_mb"] = max(res.get("peak_mem_mb", 0.0), peak_mem / (1024 * 1024))

    return res


def evaluate_functional_correctness_python(
    sample_file: str,
    timeout: float = 3.0,
    problem_file: str = "edit_eval_python.jsonl",
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)
    
    expected_assert_totals = load_expected_assert_totals(problem_file)
    samples = list(stream_jsonl(sample_file))
    code_id = Counter()
    n_samples = 0
    results = defaultdict(list)
    num_passed_testcases = 0
    
    UNINSTRUMENTED_TASKS = {"EditEval/37", "EditEval/56", "EditEval/188"}
    
    print("Evaluating samples...")
    for sample in tqdm.tqdm(samples):
        task_id = sample["task_id"]
        edited_code = sample.get("output") or sample.get("completion", "")
        edited_code = strip_leading_code_fence(edited_code, 'python')
        problem = problems[task_id]
        
        if "context" in problem and CODE_MARKER in problem["context"]:
            edited_code = problem["context"].replace(CODE_MARKER, edited_code)
        
        test_code = problem.get("test", "")
        wrapped_test = wrap_asserts_in_code(test_code)

        _escaped = edited_code.replace("'''", "\\'\\'\\'")
        candidate_and_test_wrapped = (
            "edited_code = '''" + _escaped + "\n'''\n"
            + edited_code + "\n\n" + wrapped_test + "\ncheck()"
        )
        
        if task_id in UNINSTRUMENTED_TASKS:
            candidate_and_test_clean = edited_code + "\n\n" + test_code + "\ncheck()"
            run_res = run_candidate(candidate_and_test_clean, timeout, extract_counters=False)
            suite_passed = run_res.get("suite_passed", False)
            
            assert_total = 1
            observed_assert_total = assert_total
            assert_passed = 1 if suite_passed else 0
            error_msgs = run_res.get("error", [])
            exec_time_s = run_res.get("exec_time_s", 0.0)
            peak_mem_mb = run_res.get("peak_mem_mb", 0.0)
        else:
            run_res_wrapped = run_candidate(candidate_and_test_wrapped, timeout, extract_counters=True)
            
            assert_passed = run_res_wrapped.get("assert_passed", 0)
            assert_total = run_res_wrapped.get("assert_total", 0)
            observed_assert_total = assert_total
            error_msgs = run_res_wrapped.get("error", [])
            exec_time_s = run_res_wrapped.get("exec_time_s", 0.0)
            peak_mem_mb = run_res_wrapped.get("peak_mem_mb", 0.0)
            
            if not run_res_wrapped.get("suite_passed", False) and task_id in expected_assert_totals:
                expected_total = expected_assert_totals[task_id]
                if expected_total > assert_total:
                    assert_total = expected_total
            
            suite_passed = (assert_total > 0 and assert_passed == assert_total)
        
        run_res = {
            "suite_passed": suite_passed,
            "assert_passed": assert_passed,
            "observed_assert_total": observed_assert_total,
            "assert_total": assert_total,
            "error": error_msgs,
            "exec_time_s": exec_time_s,
            "peak_mem_mb": peak_mem_mb,
        }
        results[task_id].append({
            "run_id": code_id[task_id],
            "passed": suite_passed,
            "code": edited_code,
            "task_id": task_id,
            "assert_passed": assert_passed,
            "observed_assert_total": observed_assert_total,
            "assert_total": assert_total,
            "error": error_msgs,
            "exec_time_s": exec_time_s,
            "peak_mem_mb": peak_mem_mb,
        })
        num_passed_testcases += assert_passed
        code_id[task_id] += 1
        n_samples += 1
    assert len(code_id) == len(problems), "Some problems are not attempted."

    status = "success"

    total = []
    correct = []
    for task_id in results:
        total.append(len(results[task_id]))
        correct.append(sum(1 for rr in results[task_id] if rr.get("passed")))
    total = np.array(total)
    correct = np.array(correct)


    pass_rate = correct.sum() / total.sum() if total.sum() > 0 else 0
    hallucination_rate = 1.0 - pass_rate

    tcpr_sum = 0.0 
    count_for_tcpr = 0
    for task_id in results:
        for r in results[task_id]:
            assert_total_val = int(r.get("assert_total", 0))
            assert_passed_val = int(r.get("assert_passed", 0))
            if assert_total_val > 0:
                tcpr_sum += assert_passed_val / assert_total_val
                count_for_tcpr += 1
    
    tcpr = tcpr_sum / count_for_tcpr if count_for_tcpr > 0 else 0.0
    

    metrics = {
        "status": status,
        "pass_rate": pass_rate,
        "hallucination_rate": hallucination_rate,
        "tcpr": tcpr,
    }

    metrics_file = sample_file + "_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {metrics_file}")

    results_file = sample_file + "_results.jsonl"
    with open(results_file, "w") as f:
        for task_id, task_res in results.items():
            for res in task_res:
                f.write(json.dumps(res) + "\n")
    print(f"Results written to {results_file}")

    return metrics

