import os
import re
import subprocess
import tempfile
import time
from typing import Dict, Optional

import psutil

def _extract_class_name(java_code: str, default: str = "Solution") -> str:
    match = re.search(r"public\s+class\s+(\w+)", java_code)
    if match:
        return match.group(1)
    return default


def build_test_class(method_code: str, test_code: str) -> str:
    """Insert a main method containing the instrumented test statements.
    The method_code is expected to define the public class.
    """
    method_code = method_code.rstrip()
    class_name = _extract_class_name(method_code)
    
    if not method_code.endswith("}"):
        return (
            f"public class {class_name} {{\n"
            f"    {method_code}\n"
            f"    public static void main(String[] args) {{\n        int __ASSERT_TOTAL = 0;\n        int __ASSERT_PASSED = 0;\n{instrument_java_tests(test_code)}\n        System.out.println(\"ASSERT_TOTAL=\" + __ASSERT_TOTAL);\n        System.out.println(\"ASSERT_PASSED=\" + __ASSERT_PASSED);\n    }}\n}}"
        )

    insert_pos = method_code.rfind("}")
    main_method = (
        "\n\n    public static void main(String[] args) {\n"
        "        int __ASSERT_TOTAL = 0;\n"
        "        int __ASSERT_PASSED = 0;\n"
        "        java.util.List<String> __ASSERT_ERRORS = new java.util.ArrayList<>();\n"
        f"{instrument_java_tests(test_code)}\n"
        "        System.out.println(\"ASSERT_TOTAL=\" + __ASSERT_TOTAL);\n"
        "        System.out.println(\"ASSERT_PASSED=\" + __ASSERT_PASSED);\n"
        "        for (String err : __ASSERT_ERRORS) {\n"
        "            System.err.println(\"ASSERT_ERROR: \" + err);\n"
        "        }\n"
        "    }\n"
    )
    return method_code[:insert_pos] + main_method + "\n" + method_code[insert_pos:]





def instrument_java_tests(test_code: str) -> str:
    """Transform Java test lines to count passes without aborting on first failure.
    Lines like 'if (!condition) throw new AssertionError(...)' are converted to
    'if (condition) __ASSERT_PASSED++;' to directly count passing assertions.
    Lines like 'if (condition) throw new AssertionError(...)' are converted to
    'if (!(condition)) __ASSERT_PASSED++;' to negate and count passes.
    Special handling for try-catch blocks and for-loops.
    Other lines are kept as-is. Indentation is set for inclusion inside the main method.
    
    IMPORTANT: Now also increments __ASSERT_TOTAL for every assertion check to get accurate totals.
    Also captures error messages in __ASSERT_ERRORS list for granular failure tracking.
    """
    out_lines = []
    in_try_block = False
    in_catch_block = False
    in_catch_block_depth = 0
    in_for_loop = 0  # Track nesting level of for loops
    for_loop_has_assertion = False
    in_lambda = 0  # Track lambda expression depth
    
    for raw_line in test_code.splitlines():
        line = raw_line.strip()
        if not line:
            out_lines.append("")
            continue
        
        if "->" in line and "{" in line:
            in_lambda += line.count("{") - line.count("}")
            out_lines.append(f"        {line}")
            continue
        
        if in_lambda > 0:
            in_lambda += line.count("{") - line.count("}")
            out_lines.append(f"        {line}")
            continue
        
        if line.startswith("for ") or line.startswith("for("):
            in_for_loop += 1
            out_lines.append(f"        {line}")
            continue
        
        if in_for_loop > 0 and line == "}":
            in_for_loop -= 1
            out_lines.append("        }")
            if in_for_loop == 0:
                for_loop_has_assertion = False
            continue
        
        if line.startswith("try {"):
            in_try_block = True
            out_lines.append("        try {")
            continue
        
        if in_try_block:
            if line.startswith("} catch"):
                if "{" in line:
                    out_lines.append("        " + line)
                    out_lines.append("            __ASSERT_TOTAL++;")
                    out_lines.append("            __ASSERT_PASSED++;")
                    in_try_block = False
                    in_catch_block = True
                    in_catch_block_depth = 1
                else:
                    out_lines.append("        " + line)
                    in_try_block = False
                    in_catch_block = True
            elif "throw new AssertionError" in line:
                out_lines.append("            __ASSERT_TOTAL++;")
                out_lines.append("            " + line)
            else:
                out_lines.append("        " + line)
            continue
        
        if in_catch_block:
            if "{" in line and not line.strip().startswith("//") and in_catch_block_depth == 0:
                out_lines.append("        " + line)
                out_lines.append("            __ASSERT_TOTAL++;")
                out_lines.append("            __ASSERT_PASSED++;")
                in_catch_block_depth = 1
            elif "throw new AssertionError" in line:
                out_lines.append("            __ASSERT_TOTAL++;")
                match_negated = re.match(r'if\s*\(\s*!\s*(.+?)\)\s*throw\s+new\s+AssertionError.*', line)
                if match_negated:
                    condition = match_negated.group(1).strip()
                    out_lines.append(f'            if ({condition}) __ASSERT_PASSED++; else __ASSERT_ERRORS.add("AssertionError");')
                else:
                    match_positive = re.match(r'if\s*\((.+?)\)\s*throw\s+new\s+AssertionError.*', line)
                    if match_positive:
                        condition = match_positive.group(1).strip()
                        out_lines.append(f'            if (!({condition})) __ASSERT_PASSED++; else __ASSERT_ERRORS.add("AssertionError");')
                    else:
                        out_lines.append("            " + line)
            elif line == "}":
                in_catch_block_depth -= 1
                if in_catch_block_depth <= 0:
                    in_catch_block = False
                out_lines.append("        }")
            else:
                out_lines.append("        " + line)
            continue
        
        if "throw new AssertionError" in line:
            if in_for_loop > 0:
                out_lines.append(f"            __ASSERT_TOTAL++;")
                match_negated = re.match(r'if\s*\(\s*!\s*(.+?)\)\s*throw\s+new\s+AssertionError.*', line)
                if match_negated:
                    condition = match_negated.group(1).strip()
                    out_lines.append(f'            if ({condition}) {{ __ASSERT_PASSED++; }} else {{ __ASSERT_ERRORS.add("AssertionError"); }}')
                    for_loop_has_assertion = True
                else:
                    match_positive = re.match(r'if\s*\((.+?)\)\s*throw\s+new\s+AssertionError.*', line)
                    if match_positive:
                        condition = match_positive.group(1).strip()
                        out_lines.append(f'            if (!({condition})) {{ __ASSERT_PASSED++; }} else {{ __ASSERT_ERRORS.add("AssertionError"); }}')
                        for_loop_has_assertion = True
                    else:
                        out_lines.append(f"            {line}")
            else:
                out_lines.append(f"        __ASSERT_TOTAL++;")
                match_negated = re.match(r'if\s*\(\s*!\s*(.+?)\)\s*throw\s+new\s+AssertionError.*', line)
                if match_negated:
                    condition = match_negated.group(1).strip()
                    out_lines.append(f'        if ({condition}) __ASSERT_PASSED++; else __ASSERT_ERRORS.add("AssertionError");')
                else:
                    match_positive = re.match(r'if\s*\((.+?)\)\s*throw\s+new\s+AssertionError.*', line)
                    if match_positive:
                        condition = match_positive.group(1).strip()
                        out_lines.append(f'        if (!({condition})) __ASSERT_PASSED++; else __ASSERT_ERRORS.add("AssertionError");')
                    else:
                        out_lines.append(f"        {line}")
        else:
            indent = "        " + ("    " * in_for_loop) if in_for_loop > 0 else "        "
            out_lines.append(f"{indent}{line}")
    return "\n".join(out_lines)


def check_correctness_java(problem: Dict, code: str, timeout: float, run_id: Optional[int] = None) -> Dict:
    """Compile and run Java code against the provided tests with instrumentation."""
    test_code = problem["test"]
    class_name = _extract_class_name(code)
    complete_code = build_test_class(code, test_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        java_file = os.path.join(tmpdir, f"{class_name}.java")
        with open(java_file, "w", encoding="utf-8") as f:
            f.write(complete_code)

        try:
            comp = subprocess.run(
                ["javac", java_file],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout,
                cwd=tmpdir,
            )
            if comp.returncode != 0:
                return dict(
                    task_id=problem["task_id"],
                    passed=False,
                    result="failed: compilation error",
                    error=[comp.stderr] if comp.stderr else [],
                    run_id=run_id,
                    code=code,
                    assert_total=0,
                    assert_passed=0,
                    exec_time_s=0.0,
                    peak_mem_mb=0.0,
                )
        except Exception as e:
            return dict(
                task_id=problem["task_id"],
                passed=False,
                result="failed: compilation error",
                error=[str(e)],
                run_id=run_id,
                code=code,
                assert_total=0,
                assert_passed=0,
                exec_time_s=0.0,
                peak_mem_mb=0.0,
            )

        # Run
        start = time.monotonic()
        proc = subprocess.Popen(
            ["java", class_name],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
        )

        peak_mem = 0.0
        ps_proc = None
        try:
            ps_proc = psutil.Process(proc.pid)
        except Exception:
            ps_proc = None

        timed_out = False
        try:
            while proc.poll() is None:
                if ps_proc is not None:
                    try:
                        mem = ps_proc.memory_info().rss
                        peak_mem = max(peak_mem, mem)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        ps_proc = None

                if time.monotonic() - start > timeout:
                    proc.kill()
                    timed_out = True
                    break
                time.sleep(0.001)
            stdout_data, stderr_data = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_data, stderr_data = proc.communicate()
            timed_out = True

        elapsed = time.monotonic() - start
        peak_mem_mb = peak_mem / (1024 * 1024)

        if timed_out:
            error_msg = ["TimeError"]
            return dict(
                task_id=problem["task_id"],
                passed=False,
                result="timed out",
                error=error_msg,
                run_id=run_id,
                code=code,
                assert_total=0,
                assert_passed=0,
                exec_time_s=elapsed,
                peak_mem_mb=peak_mem_mb,
            )

        assert_passed_count = 0
        assert_total_count = 0
        error_messages = []
        try:
            m_passed = re.search(r"ASSERT_PASSED=(\d+)", stdout_data)
            m_total = re.search(r"ASSERT_TOTAL=(\d+)", stdout_data)
            if m_passed:
                assert_passed_count = int(m_passed.group(1))
            if m_total:
                assert_total_count = int(m_total.group(1))
            
            # Extract individual error messages from stderr
            for line in stderr_data.splitlines():
                if line.startswith("ASSERT_ERROR: "):
                    error_messages.append(line[14:])  # Remove "ASSERT_ERROR: " prefix
        except Exception:
            pass

        if proc.returncode != 0:
            # If we have granular errors, use them; otherwise use stderr
            error_output = error_messages if error_messages else [stderr_data] if stderr_data else []
            return dict(
                task_id=problem["task_id"],
                passed=False,
                result=f"failed: {stderr_data.strip()}",
                error=error_output,
                run_id=run_id,
                code=code,
                assert_total=assert_total_count,
                assert_passed=assert_passed_count,
                stdout=stdout_data,
                stderr=stderr_data,
                exec_time_s=elapsed,
                peak_mem_mb=peak_mem_mb,
            )

        # Success case or partial failure - use granular errors if available
        error_output = error_messages if error_messages else []
        return dict(
            task_id=problem["task_id"],
            passed=(assert_passed_count == assert_total_count and assert_total_count > 0),
            result="passed" if (assert_passed_count == assert_total_count and assert_total_count > 0) else "failed",
            error=error_output,
            run_id=run_id,
            code=code,
            assert_total=assert_total_count,
            assert_passed=assert_passed_count,
            stdout=stdout_data,
            stderr=stderr_data,
            exec_time_s=elapsed,
            peak_mem_mb=peak_mem_mb,
        )
