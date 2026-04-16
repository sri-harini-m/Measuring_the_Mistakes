import json
import subprocess
import argparse
import os
import re
import time
import tempfile
import ast
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.raw import analyze
import codebleu
from cognitive_complexity.api import get_cognitive_complexity
import lizard
import google.generativeai as genai


def run_tests(code, test_cases):
    passed_count = 0
    total_execution_time = 0
    python_cmd = "python" if os.name == "nt" else "python3"
    for test_input, expected_output in test_cases:
        try:
            start_time = time.time()
            process = subprocess.run(
                [python_cmd, "-c", code],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=10
            )
            end_time = time.time()

            actual_output = process.stdout.strip()
            expected_output_stripped = expected_output.strip()

            if process.returncode == 0 and actual_output == expected_output_stripped:
                passed_count += 1
                total_execution_time += (end_time - start_time)
        except subprocess.TimeoutExpired:
            print(f"Execution timed out for one of the test cases.")
            pass
        except Exception as e:
            print(f"An error occurred during test execution: {e}")
    return passed_count, total_execution_time


def main():
    parser = argparse.ArgumentParser(description="Debug Python code using Gemini 2.5 Pro.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return

    genai.configure(api_key=api_key)
    generation_config = genai.GenerationConfig(
        temperature=0.0,
        top_p=None,
        top_k=None
    )
    model = genai.GenerativeModel('gemini-2.5-pro', generation_config=generation_config)

    dataset_path = "processed_dataset/verified_python3_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                    continue

    if not dataset:
        print("Error: No valid data found in dataset")
        return

    print(f"Using Gemini 2.5 Pro model")

    checkpoint_file = "Gemini-2.5-Pro_python3_checkpoint.json"
    output_file = "Gemini-2.5-Pro_python3_results.json"
    
    results = []
    total_public_tests = 0
    total_passed_public_tests = 0
    total_private_tests = 0
    total_passed_private_tests = 0
    start_idx = 0
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                start_idx = checkpoint_data.get("last_processed_index", 0)
                print(f"Resuming from checkpoint at index {start_idx}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_data = json.load(f)
                results = existing_data.get("results", [])
                print(f"Loaded {len(results)} existing results")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    print(f"Processing {len(dataset)} items from dataset (starting from index {start_idx})...")

    item_counter = 0
    for idx, item in enumerate(dataset, 1):
        if idx - 1 < start_idx:
            continue
            
        if item.get("code_language") == "python3":
            item_counter += 1
            item_idx = item.get('idx')
            bug_type = item.get("type", "unknown issue")
            
            print(f"\n{'='*60}")
            print(f"Processing item {item_counter}/{len(dataset)} (ID: {item_idx})")
            print(f"Bug Type: {bug_type}")

            incorrect_solution = item.get("incorrect_solutions", "")
            correct_solution = item.get("solutions", "")

            if not incorrect_solution or not correct_solution:
                print(f"Warning: Missing solution data for item {item_idx}. Skipping.")
                continue

            if incorrect_solution.startswith("```python"):
                incorrect_solution = incorrect_solution[9:]
            elif incorrect_solution.startswith("python\n"):
                incorrect_solution = incorrect_solution[7:]
            if incorrect_solution.endswith("```"):
                incorrect_solution = incorrect_solution[:-3]

            if correct_solution.startswith("```python"):
                correct_solution = correct_solution[9:]
            elif correct_solution.startswith("python\n"):
                correct_solution = correct_solution[7:]
            if correct_solution.endswith("```"):
                correct_solution = correct_solution[:-3]

            incorrect_solution = incorrect_solution.strip()
            correct_solution = correct_solution.strip()

            public_inputs = item.get("public_tests_input", [])
            if isinstance(public_inputs, str): public_inputs = [public_inputs]
            public_outputs = item.get("public_tests_output", [])
            if isinstance(public_outputs, str): public_outputs = [public_outputs]
            public_test_cases = list(zip(public_inputs, public_outputs))

            private_inputs = item.get("private_tests_input", [])
            if isinstance(private_inputs, str): private_inputs = [private_inputs]
            private_outputs = item.get("private_tests_output", [])
            if isinstance(private_outputs, str): private_outputs = [private_outputs]
            private_test_cases = list(zip(private_inputs, private_outputs))

            if not public_test_cases and not private_test_cases:
                print(f"Warning: No test cases found for item {item_idx}. Skipping.")
                continue

            prompt = f"""Fix the bugs in this Python code. Return ONLY the corrected code in a ```python code block, with NO explanation.

Incorrect Code:
```python
{incorrect_solution}
```

Bug type: {bug_type}

Corrected code:"""

            response_text = None
            model_error = None
            try:
                response = model.generate_content(prompt)
                response_text = response.text if hasattr(response, 'text') else None
            except Exception as e:
                error_message = str(e).lower()
                model_error = str(e)
                if "quota" in error_message or "rate" in error_message or "429" in error_message or "resource" in error_message:
                    print(f"\n{'='*60}")
                    print(f"RATE LIMIT ERROR detected at item {idx}/{len(dataset)} (ID: {item_idx})")
                    print(f"Error: {e}")
                    print(f"{'='*60}")
                    
                    checkpoint_data = {
                        "last_processed_index": idx - 1,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "error_message": str(e)
                    }
                    with open(checkpoint_file, "w") as f:
                        json.dump(checkpoint_data, f, indent=4)
                    print(f"Checkpoint saved to {checkpoint_file}")
                    
                    if results:
                        with open(output_file, "w") as f:
                            json.dump({"results": results, "partial": True}, f, indent=4)
                        print(f"Partial results saved to {output_file}")
                    
                    print(f"\nTo resume, use a new API key and run the script again.")
                    print(f"It will automatically continue from item {idx}.")
                    return
                else:
                    print(f"Error calling Gemini API for item {item_idx}: {e}")
                    response_text = None

            if response_text is None or not response_text.strip():
                print(f"Warning: No response from model for item {item_idx}.")
                results.append({
                    "id": item.get("idx"),
                    "status": "no response",
                    "pass_rate": 0,
                    "public_tests_passed": 0,
                    "public_tests_total": len(public_test_cases),
                    "private_tests_passed": 0,
                    "private_tests_total": len(private_test_cases),
                    "efficiency": 0,
                    "hallucination_rate": 1.0,
                    "pass_at_1": 0,
                    "codebleu": -1,
                    "code_smells": False,
                    "cyclomatic_complexity": -1,
                    "loc": 0,
                    "cognitive_complexity": -1,
                    "halstead_metrics": None,
                    "model_response": None,
                    "model_error": model_error,
                })
                
                checkpoint_data = {
                    "last_processed_index": idx,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f, indent=4)
                continue

            debugged_code = ""
            code_match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```(.*?)```", response_text, re.DOTALL)

            if code_match:
                debugged_code = code_match.group(1).strip()

            if not debugged_code:
                print(f"Warning: No Python code block found in response for item {item_idx}.")
                results.append({
                    "id": item.get("idx"),
                    "status": "no response",
                    "pass_rate": 0,
                    "public_tests_passed": 0,
                    "public_tests_total": len(public_test_cases),
                    "private_tests_passed": 0,
                    "private_tests_total": len(private_test_cases),
                    "efficiency": 0,
                    "hallucination_rate": 1.0,
                    "pass_at_1": 0,
                    "codebleu": -1,
                    "code_smells": False,
                    "cyclomatic_complexity": -1,
                    "loc": 0,
                    "cognitive_complexity": -1,
                    "halstead_metrics": None,
                    "model_response": response_text,
                    "model_error": None,
                })
                
                checkpoint_data = {
                    "last_processed_index": idx,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f, indent=4)
                continue

            print(f"Running tests on debugged code...")
            passed_public_count, public_exec_time = run_tests(debugged_code, public_test_cases)
            passed_private_count, private_exec_time = run_tests(debugged_code, private_test_cases)

            total_passed_count = passed_public_count + passed_private_count
            total_tests_count = len(public_test_cases) + len(private_test_cases)
            total_execution_time = public_exec_time + private_exec_time

            total_public_tests += len(public_test_cases)
            total_passed_public_tests += passed_public_count
            total_private_tests += len(private_test_cases)
            total_passed_private_tests += passed_private_count

            loc = len(debugged_code.splitlines())
            
            cc = -1
            try:
                cc = sum(c.complexity for c in cc_visit(debugged_code)) if debugged_code else -1
            except Exception as e:
                print(f"  Warning: Could not calculate cyclomatic complexity: {e}")
                cc = -1
            
            halstead = -1
            try:
                halstead_result = h_visit(debugged_code).total if debugged_code else None
                halstead = halstead_result if halstead_result is not None else -1
            except Exception as e:
                print(f"  Warning: Could not calculate Halstead metrics: {e}")
                halstead = -1

            cognitive_complexity = -1
            try:
                tree = ast.parse(debugged_code)
                cognitive_complexity = 0
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        cognitive_complexity += get_cognitive_complexity(node)
            except Exception as e:
                print(f"  Warning: Could not calculate cognitive complexity: {e}")
                cognitive_complexity = -1

            codebleu_value = -1
            try:
                codebleu_score = codebleu.calc_codebleu([correct_solution], [debugged_code], lang="python",
                                                        weights=(0.25, 0.25, 0.25, 0.25))
                codebleu_value = codebleu_score['codebleu']
            except Exception as e:
                print(f"  Warning: Could not calculate CodeBLEU: {e}")
                codebleu_value = -1

            code_smells = False
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False, encoding='utf-8') as temp_f:
                temp_filename = temp_f.name
                temp_f.write(debugged_code)
                temp_f.flush()
            
            try:
                lizard_analysis = lizard.analyze_file(temp_filename)
                for func in lizard_analysis.function_list:
                    if func.cyclomatic_complexity > 10 or func.nloc > 50:
                        code_smells = True
                        break
            except Exception as e:
                print(f"Lizard analysis failed for item {item.get('idx')}: {e}")
            finally:
                try:
                    os.unlink(temp_filename)
                except Exception:
                    pass

            pass_rate = total_passed_count / total_tests_count if total_tests_count > 0 else 0
            status = "no fixes"
            if pass_rate == 1.0:
                status = "fully correct"
            elif pass_rate > 0:
                status = "partially correct"

            print(f"Status: {status}, Pass rate: {pass_rate:.2%} ({total_passed_count}/{total_tests_count})")

            results.append({
                "id": item.get("idx"),
                "status": status,
                "pass_rate": pass_rate,
                "public_tests_passed": passed_public_count,
                "public_tests_total": len(public_test_cases),
                "private_tests_passed": passed_private_count,
                "private_tests_total": len(private_test_cases),
                "efficiency": total_execution_time / total_passed_count if total_passed_count > 0 else 0,
                "hallucination_rate": 1.0 - pass_rate,
                "pass_at_1": 1 if pass_rate == 1.0 else 0,
                "codebleu": codebleu_value,
                "code_smells": code_smells,
                "cyclomatic_complexity": cc,
                "loc": loc,
                "cognitive_complexity": cognitive_complexity,
                "halstead_metrics": halstead._asdict() if (halstead != -1 and halstead is not None) else None,
                "model_response": response_text,
                "debugged_code": debugged_code,
            })
            
            checkpoint_data = {
                "last_processed_index": idx,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=4)
            
            with open(output_file, "w") as f:
                json.dump({"results": results}, f, indent=4)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\nCheckpoint file removed (processing completed successfully)")
    
    if results:
        total_overall_tests = total_public_tests + total_private_tests
        total_overall_passed = total_passed_public_tests + total_passed_private_tests

        aggregated_results = {
            "total_problems_processed": len(results),
            "overall_pass_rate": total_overall_passed / total_overall_tests if total_overall_tests > 0 else 0,
            "public_tests_pass_rate": total_passed_public_tests / total_public_tests if total_public_tests > 0 else 0,
            "private_tests_pass_rate": total_passed_private_tests / total_private_tests if total_private_tests > 0 else 0,
            "average_codebleu": sum(r["codebleu"] for r in results if r["codebleu"] != -1) / len([r for r in results if r["codebleu"] != -1]) if any(r["codebleu"] != -1 for r in results) else 0,
            "total_fully_correct": sum(1 for r in results if r["status"] == "fully correct"),
            "total_partially_correct": sum(1 for r in results if r["status"] == "partially correct"),
            "total_no_fixes": sum(1 for r in results if r["status"] == "no fixes"),
            "total_no_response": sum(1 for r in results if r["status"] == "no response"),
        }

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total problems processed: {aggregated_results['total_problems_processed']}")
        print(f"Overall pass rate: {aggregated_results['overall_pass_rate']:.2%}")
        print(f"Public tests pass rate: {aggregated_results['public_tests_pass_rate']:.2%}")
        print(f"Private tests pass rate: {aggregated_results['private_tests_pass_rate']:.2%}")
        print(f"Average CodeBLEU: {aggregated_results['average_codebleu']:.4f}")
        print(f"Fully correct: {aggregated_results['total_fully_correct']}")
        print(f"Partially correct: {aggregated_results['total_partially_correct']}")
        print(f"No fixes: {aggregated_results['total_no_fixes']}")
        print(f"No response: {aggregated_results['total_no_response']}")
        print("=" * 60)
    else:
        aggregated_results = {"message": "No Python items were processed."}
        print("\nWarning: No Python items were processed.")

    output_file = "gemini_python_results.json"
    with open(output_file, "w") as f:
        json.dump({"results": results, "aggregated": aggregated_results}, f, indent=4)

    print(f"\nDebugging complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
