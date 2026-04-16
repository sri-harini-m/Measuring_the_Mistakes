import json
import subprocess
import argparse
import os
import re
import time
import tempfile
import codebleu
import lizard
from anthropic import Anthropic


def run_tests(code, test_cases):
    passed_count = 0
    total_execution_time = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cpp_file = os.path.join(tmpdir, "solution.cpp")
        exe_file = os.path.join(tmpdir, "solution")
        
        with open(cpp_file, "w") as f:
            f.write(code)
        
        try:
            compile_process = subprocess.run(
                ["g++", "-std=c++17", "-O2", cpp_file, "-o", exe_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if compile_process.returncode != 0:
                return 0, 0
        except subprocess.TimeoutExpired:
            return 0, 0
        except Exception:
            return 0, 0
        
        for test_input, expected_output in test_cases:
            try:
                start_time = time.time()
                process = subprocess.run(
                    [exe_file],
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
                pass
            except Exception:
                pass
    
    return passed_count, total_execution_time


def main():
    parser = argparse.ArgumentParser(description="Debug C++ code using Claude Sonnet 4.5.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    client = Anthropic(api_key=api_key)

    dataset_path = "processed_dataset/verified_cpp_dataset.jsonl"
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

    print(f"Using Claude Sonnet 4.5 model")

    checkpoint_file = "Claude-Sonnet-4.5_cpp_checkpoint.json"
    output_file = "Claude-Sonnet-4.5_cpp_results.json"
    
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
    if args.limit:
        print(f"LIMITING to {args.limit} item(s) for testing")

    item_counter = 0
    for idx, item in enumerate(dataset, 1):
        if idx - 1 < start_idx:
            continue
            
        if item.get("code_language") == "cpp":
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

            for prefix in ["```cpp\n", "```cpp", "```c++\n", "```c++", "cpp\n", "c++\n"]:
                if incorrect_solution.startswith(prefix):
                    incorrect_solution = incorrect_solution[len(prefix):]
                    break
            for prefix in ["```cpp\n", "```cpp", "```c++\n", "```c++", "cpp\n", "c++\n"]:
                if correct_solution.startswith(prefix):
                    correct_solution = correct_solution[len(prefix):]
                    break
            if incorrect_solution.endswith("```"):
                incorrect_solution = incorrect_solution[:-3]
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

            prompt = f"""Fix the bugs in this C++ code. Return ONLY the corrected code in a ```cpp code block, with NO explanation.

Incorrect Code:
```cpp
{incorrect_solution}
```

Bug type: {bug_type}

Corrected code:"""

            response_text = None
            model_error = None
            try:
                message = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=4096,
                    temperature=0.0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = message.content[0].text if message.content else None
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
                    print(f"Error calling Claude API for item {item_idx}: {e}")
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
                
                with open(output_file, "w") as f:
                    json.dump({"results": results}, f, indent=4)
                continue

            debugged_code = ""
            code_match = re.search(r"```cpp\n(.*?)```", response_text, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```c\+\+\n(.*?)```", response_text, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```(.*?)```", response_text, re.DOTALL)

            if code_match:
                debugged_code = code_match.group(1).strip()

            if not debugged_code:
                print(f"Warning: No C++ code block found in response for item {item_idx}.")
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
                
                with open(output_file, "w") as f:
                    json.dump({"results": results}, f, indent=4)
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
            cognitive_complexity = -1
            code_smells = False
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.cpp', delete=False, encoding='utf-8') as temp_f:
                temp_filename = temp_f.name
                temp_f.write(debugged_code)
                temp_f.flush()
            
            try:
                lizard_analysis = lizard.analyze_file(temp_filename)
                cc = sum(func.cyclomatic_complexity for func in lizard_analysis.function_list)
                cognitive_complexity = cc
                
                for func in lizard_analysis.function_list:
                    if func.cyclomatic_complexity > 10 or func.nloc > 50:
                        code_smells = True
                        break
            except Exception as e:
                print(f"  Warning: Lizard analysis failed: {e}")
                cc = -1
                cognitive_complexity = -1
            finally:
                try:
                    os.unlink(temp_filename)
                except Exception:
                    pass

            halstead = None

            codebleu_value = -1
            try:
                codebleu_score = codebleu.calc_codebleu([correct_solution], [debugged_code], lang="cpp",
                                                        weights=(0.25, 0.25, 0.25, 0.25))
                codebleu_value = codebleu_score['codebleu']
            except Exception as e:
                print(f"  Warning: Could not calculate CodeBLEU: {e}")
                codebleu_value = -1

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
                "halstead_metrics": halstead,
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
            
            if args.limit and item_counter >= args.limit:
                print(f"\nReached limit of {args.limit} item(s). Stopping.")
                break

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
        aggregated_results = {"message": "No C++ items were processed."}
        print("\nWarning: No C++ items were processed.")

    with open(output_file, "w") as f:
        json.dump({"results": results, "aggregated": aggregated_results}, f, indent=4)

    print(f"\nDebugging complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
