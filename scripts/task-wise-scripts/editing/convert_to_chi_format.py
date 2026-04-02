#!/usr/bin/env python3
"""
Convert evaluation results to CHI script CSV format.

Usage:
    python convert_to_chi_format.py <results_file.jsonl> <problem_file.jsonl> <output.csv> [language]

Example:
    python convert_to_chi_format.py llama_python_samples.jsonl_results.jsonl edit_eval_python.jsonl llama_chi_input.csv python
"""

import json
import csv
import sys
import re
from pathlib import Path

CODE_MARKER = r"{{Code}}"


def inject_edited_code_java(code: str) -> str:
    """
    Injects the 'edited_code' variable into the Java class as a static string field.
    This is required because the Java tests expect 'edited_code' to be available,
    similar to how it is in Python and C++.
    """
    # Escape for Java string literal
    escaped_code = code.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    
    # Find the last closing brace (assuming it closes the class)
    last_brace = code.rfind("}")
    if last_brace == -1:
        return code
        
    injection = f'\n    public static String edited_code = "{escaped_code}";\n'
    
    return code[:last_brace] + injection + code[last_brace:]

def convert_to_chi_format(results_file, problem_file, output_file, language=None):
    """
    Convert evaluation results to CHI CSV format.
    
    Args:
        results_file: Path to *_results.jsonl file
        problem_file: Path to problem definition file (edit_eval_*.jsonl)
        output_file: Path to output CSV file
        language: Language override (python/cpp/java), or auto-detect from filenames
    """
    
    # Auto-detect language if not provided
    if not language:
        if 'python' in str(results_file).lower():
            language = 'python'
        elif 'cpp' in str(results_file).lower():
            language = 'cpp'
        elif 'java' in str(results_file).lower():
            language = 'java'
        else:
            language = 'python'  # Default
    
    print(f"Converting {results_file} to CHI format...")
    print(f"Language: {language}")
    print(f"Problem file: {problem_file}")
    
    # Load problems (for test code)
    problems = {}
    try:
        with open(problem_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problem = json.loads(line)
                    problems[problem['task_id']] = problem
        print(f"Loaded {len(problems)} problems")
    except FileNotFoundError:
        print(f"Warning: Problem file {problem_file} not found. Using placeholder test cases.")
        problems = {}
    
    # Derive sample file path from results file (remove _results.jsonl suffix)
    sample_file = str(results_file).replace('_results.jsonl', '')
    print(f"Sample file: {sample_file}")
    
    # Load samples (for generated code)
    samples = {}
    with open(sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # Code can be in 'code', 'output', or 'completion' field
                code = sample.get('code') or sample.get('output') or sample.get('completion', '')
                samples[sample['task_id']] = code
    
    print(f"Loaded {len(samples)} samples")
    
    # Load results (for task_id list)
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                results.append(result)
    
    print(f"Loaded {len(results)} results")
    
    # Convert to CHI format
    rows = []
    for result in results:
        task_id = result['task_id']
        
        # Get code from samples file
        code = samples.get(task_id, '')
        
        # Get test code and context
        if task_id in problems:
            problem = problems[task_id]
            test_code = problem.get('test', '')
            
            # Apply context if present (e.g., class wrapper for method snippets)
            # Then, for Java, always inject edited_code to mirror evaluation_java behavior
            if 'context' in problem and CODE_MARKER in problem['context']:
                if "edited_code =" not in problem['context'] or str(language).lower() != 'java':
                        code = problem['context'].replace(CODE_MARKER, code)

                if str(language).lower() == 'java':
                    code = inject_edited_code_java(code)
            else:
                if str(language).lower() == 'java':
                    code = inject_edited_code_java(code)
        else:
            # Empty test if problem not found
            test_code = ""

        
        rows.append({
            'language': language,
            'code': code,
            'test_code': test_code,
            # 'assert_passed': result.get('assert_passed', 0),
            # 'assert_total': result.get('assert_total', 0)
        })
    
    # Write CSV
    fieldnames = ['language', 'code', 'test_code']

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nConversion complete!")
    print(f"Output written to: {output_file}")
    print(f"Total samples: {len(rows)}")
    print(f"\nConversion complete! The CHI script now handles EditEval-style assertion tests.")


def main():
    if len(sys.argv) < 4:
        print("Usage: python convert_to_chi_format.py <results_file.jsonl> <problem_file.jsonl> <output.csv> [language]")
        print()
        print("Arguments:")
        print("  results_file.jsonl - Results from evaluation (e.g., llama_python_samples.jsonl_results.jsonl)")
        print("  problem_file.jsonl - Problem definitions with tests (e.g., edit_eval_python.jsonl)")
        print("  output.csv         - Output CSV file for CHI script")
        print("  language           - Optional: python, cpp, or java (auto-detected if not provided)")
        print()
        print("Example:")
        print("  python convert_to_chi_format.py llama_python_samples.jsonl_results.jsonl edit_eval_python.jsonl llama_chi.csv python")
        sys.exit(1)
    
    results_file = sys.argv[1]
    problem_file = sys.argv[2]
    output_file = sys.argv[3]
    language = sys.argv[4] if len(sys.argv) > 4 else None
    
    try:
        convert_to_chi_format(results_file, problem_file, output_file, language)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
