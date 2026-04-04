import os
import subprocess
import tempfile
import time
from typing import Dict, Optional

import psutil


def check_correctness_cpp(problem: Dict, code: str, timeout: float, run_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a generated C++ code by running the test
    suite provided in the problem.
    Handles context field with {{Code}} placeholder replacement.
    """
    
    CODE_MARKER = "{{Code}}"
    if "context" in problem and CODE_MARKER in problem["context"]:
        code = problem["context"].replace(CODE_MARKER, code)
    
    delimiter = "CPP_EVAL_DELIM"
    while delimiter in code:
        delimiter += "_"

    edited_code_def = f'#include <string>\nstd::string edited_code = R"{delimiter}({code}){delimiter}";'

    instrumentation = r"""
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <cmath>

long long __assert_total__ = 0;
long long __assert_passed__ = 0;

void __print_metrics__() {
    std::cout << "\n__METRICS__:" << __assert_total__ << ":" << __assert_passed__ << std::endl;
}

struct __MetricsPrinter__ {
    __MetricsPrinter__() {
        std::atexit(__print_metrics__);
    }
};
__MetricsPrinter__ __printer__;

#ifdef assert
#undef assert
#endif

#define assert(condition) do { \
    __assert_total__++; \
    try { \
        if (condition) { \
            __assert_passed__++; \
        } else { \
            std::cerr << "Assertion failed at line " << __LINE__ << ": " << #condition << std::endl; \
        } \
    } catch (const std::exception& e) { \
        std::cerr << "Exception during assertion at line " << __LINE__ << ": " << e.what() << std::endl; \
    } catch (...) { \
        std::cerr << "Unknown exception during assertion at line " << __LINE__ << std::endl; \
    } \
} while (0)
"""
    
    clean_code = code.replace("#include <cassert>", "// #include <cassert>").replace("#include <assert.h>", "// #include <assert.h>")
    clean_test = problem["test"].replace("#include <cassert>", "// #include <cassert>").replace("#include <assert.h>", "// #include <assert.h>")
    
    full_program = instrumentation + "\n" + edited_code_def + "\n" + clean_code + "\n\n" + clean_test
    
    result_status = "failed"
    error_message = ""
    assert_total = 0
    assert_passed = 0
    compilation_error = ""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cpp_file = os.path.join(temp_dir, "solution.cpp")
        exe_file = os.path.join(temp_dir, "solution")
        
        with open(cpp_file, "w") as f:
            f.write(full_program)
        
        compile_cmd = ["g++", cpp_file, "-o", exe_file, "-std=c++17"]

        try:
            subprocess.check_output(compile_cmd, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.CalledProcessError as e:
            compilation_error = e.output.decode('utf-8', errors='ignore')
            return dict(
                task_id=problem["task_id"],
                passed=False,
                result="failed: compilation error",
                error=[compilation_error] if compilation_error else [],
                run_id=run_id,
                code=code,
                assert_total=0,
                assert_passed=0,
                exec_time_s=0.0,
                peak_mem_mb=0.0,
            )
        try:
            start = time.monotonic()
            proc = subprocess.Popen(
                [exe_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            peak_mem = 0.0
            ps_proc = None
            try:
                ps_proc = psutil.Process(proc.pid)
            except Exception:
                ps_proc = None

            timed_out = False
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

            try:
                stdout_data, stderr_data = proc.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                stdout_data, stderr_data = proc.communicate()

            elapsed = time.monotonic() - start
            peak_mem_mb = peak_mem / (1024 * 1024)

            stdout_str = stdout_data.decode('utf-8', errors='ignore')
            stderr_str = stderr_data.decode('utf-8', errors='ignore')

            if timed_out:
                result_status = "timed out"
                error_lines = ["TimeError"]
                return dict(
                    task_id=problem["task_id"],
                    passed=False,
                    result=result_status,
                    error=error_lines,
                    run_id=run_id,
                    code=code,
                    assert_total=0,
                    assert_passed=0,
                    exec_time_s=elapsed,
                    peak_mem_mb=peak_mem_mb,
                )

            import re
            metrics_match = re.search(r"__METRICS__:(\d+):(\d+)", stdout_str)
            if metrics_match:
                assert_total = int(metrics_match.group(1))
                assert_passed = int(metrics_match.group(2))
            else:
                assert_total = 0
                assert_passed = 0

            # Parse error messages as a list of individual lines
            error_lines = [line.strip() for line in stderr_str.splitlines() if line.strip()]

            if proc.returncode == 0:
                if assert_total > 0 and assert_passed < assert_total:
                    result_status = "failed: assertions failed"
                else:
                    result_status = "passed"
            else:
                result_status = f"failed: runtime error (exit code {proc.returncode})"

        except Exception as e:
            result_status = f"failed: {str(e)}"
            error_lines = [str(e)]
            assert_total = 0
            assert_passed = 0
            elapsed = 0.0
            peak_mem_mb = 0.0

    # Ensure error is always populated as a list
    if not error_lines and result_status != "passed":
        error_lines = [result_status]
    
    return dict(
        task_id=problem["task_id"],
        passed=(result_status == "passed"),
        result=result_status,
        error=error_lines,
        run_id=run_id,
        code=code,
        assert_total=assert_total,
        assert_passed=assert_passed,
        exec_time_s=elapsed,
        peak_mem_mb=peak_mem_mb,
    )
