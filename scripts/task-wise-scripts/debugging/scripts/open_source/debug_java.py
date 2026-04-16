import json
import subprocess
import argparse
import os
import re
import time
import tempfile
import ast
import torch
import transformers
from tqdm import tqdm
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.raw import analyze
import codebleu
from cognitive_complexity.api import get_cognitive_complexity
import lizard

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def run_tests(code, test_cases):
    passed_count = 0
    total_execution_time = 0
    
    class_name_match = re.search(r'public\s+class\s+(\w+)', code)
    if not class_name_match:
        class_name_match = re.search(r'class\s+(\w+)', code)
    
    if not class_name_match:
        return passed_count, total_execution_time
    
    class_name = class_name_match.group(1)
    
    temp_dir = tempfile.mkdtemp()
    source_file_path = os.path.join(temp_dir, f"{class_name}.java")
    
    try:
        with open(source_file_path, 'w') as source_file:
            source_file.write(code)
        
        compile_process = subprocess.run(
            ["javac", source_file_path],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=temp_dir
        )
        
        if compile_process.returncode != 0:
            return passed_count, total_execution_time
        
        for test_input, expected_output in test_cases:
            try:
                start_time = time.time()
                process = subprocess.run(
                    ["java", class_name],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=temp_dir
                )
                end_time = time.time()

                actual_output = process.stdout.strip()
                expected_output_stripped = expected_output.strip()

                if process.returncode == 0 and actual_output == expected_output_stripped:
                    passed_count += 1
                    total_execution_time += (end_time - start_time)
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                pass
    finally:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return passed_count, total_execution_time


def main():
    parser = argparse.ArgumentParser(description="Debug Java code using a specified model.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="The model ID from Hugging Face to use for debugging.")
    parser.add_argument("--hf_token", type=str, required=True, help="Your Hugging Face access token.")
    args = parser.parse_args()

    dataset_path = "processed_dataset/verified_java_dataset.jsonl"
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

    print(f"Loading model: {args.model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="sequential",
        token=args.hf_token,
    )
    
    if pipeline.tokenizer.pad_token is None:
        pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    if pipeline.tokenizer.pad_token_id is None:
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

    output_dir = "open_source"
    model_name = args.model_id.replace('/', '_')
    model_output_dir = os.path.join(output_dir, model_name, "java")
    os.makedirs(model_output_dir, exist_ok=True)
    output_file = os.path.join(model_output_dir, f"{model_name}_java_results.json")
    checkpoint_file = os.path.join(model_output_dir, f"{model_name}_java_checkpoint.json")

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
    print(f"Results will be saved to: {output_file}")

    item_counter = 0
    pbar = tqdm(total=len(dataset), desc="Debugging code", position=0, leave=True, ncols=100, mininterval=1.0)
    for idx, item in enumerate(dataset, 1):
        if idx - 1 < start_idx:
            pbar.update(1)
            continue
            
        if item.get("code_language") == "java":
            item_counter += 1
            item_idx = item.get('idx')
            bug_type = item.get("type", "unknown issue")
            
            print(f"\n{'='*60}", flush=True)
            print(f"Processing item {item_counter}/{len(dataset)} (ID: {item_idx})", flush=True)
            print(f"Bug Type: {bug_type}", flush=True)
            
            incorrect_solution = item.get("incorrect_solutions", "")
            correct_solution = item.get("solutions", "")

            if not incorrect_solution or not correct_solution:
                print(f"Warning: Missing solution data for item {item_idx}. Skipping.", flush=True)
                continue
            
            for prefix in ["```java\n", "```java", "java\n"]:
                if incorrect_solution.startswith(prefix):
                    incorrect_solution = incorrect_solution[len(prefix):]
                    break
            for prefix in ["```java\n", "```java", "java\n"]:
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
                print(f"Warning: No test cases found for item {item_idx}. Skipping.", flush=True)
                continue
            
            system_prompt = "You are an expert at debugging Java code."
            user_prompt = f"""The following code is incorrect and contains the following type(s) of error(s): {bug_type}. Analyze it, fix all the bugs, and provide the corrected version.

Incorrect Code:
```java
{incorrect_solution}
```
Provide only the corrected and complete Java code inside a markdown block. Do not provide any explanation."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            terminators = [t for t in terminators if t is not None]

            outputs = pipeline(
                messages,
                max_new_tokens=1024,
                eos_token_id=terminators if terminators else None,
                do_sample=False,
                pad_token_id=pipeline.tokenizer.pad_token_id
            )
            
            response_text = outputs[0]["generated_text"][-1]['content']

            debugged_code = ""
            try:
                code_match = re.search(r"```java\n(.*?)```", response_text, re.DOTALL)
                if not code_match:
                    code_match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)
                if not code_match:
                    code_match = re.search(r"```(.*?)```", response_text, re.DOTALL)

            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            if code_match:
                debugged_code = code_match.group(1).strip()

            if not debugged_code:
                print(f"Warning: No Java code block found for item {item_idx}. Recording as 'no response'.", flush=True)
                results.append({
                    "id": item.get("idx"),
                    "status": "no response",
                    "model_response": response_text,
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
                    "loc": -1,
                    "cognitive_complexity": -1,
                    "halstead_metrics": None,
                })
                
                checkpoint_data = {
                    "last_processed_index": idx,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f, indent=4)
                
                with open(output_file, "w") as f:
                    json.dump({"results": results}, f, indent=4)
                
                pbar.update(1)
                continue

            print("Running tests on debugged code...", flush=True)
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
            halstead = -1
            code_smells = False

            with tempfile.NamedTemporaryFile(mode='w+', suffix='.java', delete=True) as temp_f:
                temp_f.write(debugged_code)
                temp_f.flush()
                try:
                    lizard_analysis = lizard.analyze_file(temp_f.name)
                    if lizard_analysis.function_list:
                        cc = sum(func.cyclomatic_complexity for func in lizard_analysis.function_list)
                        
                        for func in lizard_analysis.function_list:
                            if func.cyclomatic_complexity > 10 or func.nloc > 50:
                                code_smells = True
                                break
                    else:
                        cc = -1
                except Exception as e:
                    print(f"  Warning: Lizard analysis failed: {e}", flush=True)

            if cc != -1:
                cognitive_complexity = cc

            codebleu_value = -1
            try:
                codebleu_score = codebleu.calc_codebleu([correct_solution], [debugged_code], lang="java",
                                                    weights=(0.25, 0.25, 0.25, 0.25))
                codebleu_value = codebleu_score['codebleu']
            except Exception as e:
                print(f"  Warning: Could not calculate CodeBLEU: {e}", flush=True)
                codebleu_value = -1

            pass_rate = total_passed_count / total_tests_count if total_tests_count > 0 else 0
            status = "no fixes"
            if pass_rate == 1.0:
                status = "fully correct"
            elif pass_rate > 0:
                status = "partially correct"
            
            print(f"Status: {status}, Pass rate: {pass_rate:.2%} ({total_passed_count}/{total_tests_count})", flush=True)

            results.append({
                "id": item.get("idx"),
                "status": status,
                "model_response": response_text,
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
            })
            
            checkpoint_data = {
                "last_processed_index": idx,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=4)
            
            with open(output_file, "w") as f:
                json.dump({"results": results}, f, indent=4)

        pbar.update(1)
    
    pbar.close()

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\nCheckpoint file removed (processing completed successfully)")

    if results:
        total_overall_tests = total_public_tests + total_private_tests
        total_overall_passed = total_passed_public_tests + total_passed_private_tests

        valid_codebleu_scores = [r["codebleu"] for r in results if r['codebleu'] != -1]
        average_codebleu = sum(valid_codebleu_scores) / len(valid_codebleu_scores) if valid_codebleu_scores else 0
        
        aggregated_results = {
            "total_problems_processed": len(results),
            "overall_pass_rate": total_overall_passed / total_overall_tests if total_overall_tests > 0 else 0,
            "public_tests_pass_rate": total_passed_public_tests / total_public_tests if total_public_tests > 0 else 0,
            "private_tests_pass_rate": total_passed_private_tests / total_private_tests if total_private_tests > 0 else 0,
            "average_codebleu": average_codebleu,
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
        aggregated_results = {"message": "No Java items were processed."}
        print("\nWarning: No Java items were processed.")

    with open(output_file, "w") as f:
        json.dump({"results": results, "aggregated": aggregated_results}, f, indent=4)

    print(f"\nDebugging complete. Results saved to {output_file}")

if __name__ == "__main__": 
    main()
