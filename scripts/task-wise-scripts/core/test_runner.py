import subprocess
import time
from typing import List, Tuple, Dict

from core.models import TestCase


def run_python_tests(code: str, test_cases: List[TestCase]) -> Tuple[int, int, float]:
    passed_count: int = 0
    total_execution_time: float = 0.0
    
    for test_case in test_cases:
        try:
            start_time: float = time.time()
            process: subprocess.CompletedProcess = subprocess.run(
                ["python3", "-c", code],
                input=test_case['input'],
                capture_output=True,
                text=True,
                timeout=10
            )
            end_time: float = time.time()
            
            actual_output: str = process.stdout.strip()
            expected_output: str = test_case['output'].strip()
            
            if process.returncode == 0 and actual_output == expected_output:
                passed_count += 1
                total_execution_time += (end_time - start_time)
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
    
    return passed_count, len(test_cases), total_execution_time


def run_cpp_tests(code: str, test_cases: List[TestCase]) -> Tuple[int, int, float]:
    import tempfile
    import os
    
    passed_count: int = 0
    total_execution_time: float = 0.0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path: str = os.path.join(temp_dir, "solution.cpp")
        binary_path: str = os.path.join(temp_dir, "solution")
        
        with open(source_path, 'w') as f:
            f.write(code)
        
        compile_result: subprocess.CompletedProcess = subprocess.run(
            ["g++", "-std=c++17", source_path, "-o", binary_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            return 0, len(test_cases), 0.0
        
        for test_case in test_cases:
            try:
                start_time: float = time.time()
                process: subprocess.CompletedProcess = subprocess.run(
                    [binary_path],
                    input=test_case['input'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                end_time: float = time.time()
                
                actual_output: str = process.stdout.strip()
                expected_output: str = test_case['output'].strip()
                
                if process.returncode == 0 and actual_output == expected_output:
                    passed_count += 1
                    total_execution_time += (end_time - start_time)
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
    
    return passed_count, len(test_cases), total_execution_time


def run_java_tests(code: str, test_cases: List[TestCase]) -> Tuple[int, int, float]:
    import tempfile
    import os
    import re
    
    passed_count: int = 0
    total_execution_time: float = 0.0
    
    class_match = re.search(r'^public\s+(?:final\s+)?(?:abstract\s+)?class\s+(\w+)\s*(?:<[^>]*>)?\s*(?:extends\s+\w+(?:<[^>]*>)?)?\s*(?:implements\s+[\w,\s<>]+)?\s*\{', code, re.MULTILINE)
    if not class_match:
        class_match = re.search(r'^(?:final\s+)?(?:abstract\s+)?class\s+(\w+)\s*(?:<[^>]*>)?\s*(?:extends\s+\w+(?:<[^>]*>)?)?\s*(?:implements\s+[\w,\s<>]+)?\s*\{', code, re.MULTILINE)
    if not class_match:
        return 0, len(test_cases), 0.0
    
    class_name: str = class_match.group(1)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        source_file: str = os.path.join(temp_dir, f"{class_name}.java")
        
        with open(source_file, 'w') as f:
            f.write(code)
        
        compile_result: subprocess.CompletedProcess = subprocess.run(
            ["javac", source_file],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=temp_dir
        )
        
        if compile_result.returncode != 0:
            return 0, len(test_cases), 0.0
        
        for test_case in test_cases:
            try:
                start_time: float = time.time()
                process: subprocess.CompletedProcess = subprocess.run(
                    ["java", class_name],
                    input=test_case['input'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=temp_dir
                )
                end_time: float = time.time()
                
                actual_output: str = process.stdout.strip()
                expected_output: str = test_case['output'].strip()
                
                if process.returncode == 0 and actual_output == expected_output:
                    passed_count += 1
                    total_execution_time += (end_time - start_time)
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
    
    return passed_count, len(test_cases), total_execution_time


def run_tests(code: str, language: str, test_cases: List[TestCase]) -> Dict[str, any]:
    if not test_cases:
        return {
            'passed': 0,
            'total': 0,
            'pass_rate': 0.0,
            'execution_time': 0.0
        }
    
    if language == 'python':
        passed, total, exec_time = run_python_tests(code, test_cases)
    elif language == 'cpp':
        passed, total, exec_time = run_cpp_tests(code, test_cases)
    elif language == 'java':
        passed, total, exec_time = run_java_tests(code, test_cases)
    else:
        return {
            'passed': 0,
            'total': len(test_cases),
            'pass_rate': 0.0,
            'execution_time': 0.0
        }
    
    return {
        'passed': passed,
        'total': total,
        'pass_rate': passed / total if total > 0 else 0.0,
        'execution_time': exec_time
    }

