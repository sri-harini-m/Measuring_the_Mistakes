import json
import subprocess
import argparse
import os
import re
import time
import tempfile
import sys
import traceback
from typing import Optional

import lizard
import codebleu


class BaseLLMAdapter:
    def __init__(self, model: str, timeout: int = 300):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> Optional[str]:
        raise NotImplementedError

class OpenAIAdapter(BaseLLMAdapter):
    def __init__(self, model: str, timeout: int = 300, rate_limit_delay: float = 0.0):
        super().__init__(model, timeout)
        self.rate_limit_delay: float = rate_limit_delay
        self.last_request_time: float = 0.0
        self._client = None
    
    def _initialize_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                import os
                
                api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                
                self._client = OpenAI(api_key=api_key)
                print(f"  -> Initialized OpenAI client with model: {self.model}")
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
    
    def _apply_rate_limit(self) -> None:
        if self.rate_limit_delay > 0:
            current_time: float = time.time()
            time_since_last_request: float = current_time - self.last_request_time
            
            if time_since_last_request < self.rate_limit_delay:
                sleep_time: float = self.rate_limit_delay - time_since_last_request
                print(f"  -> Rate limiting: waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def generate(self, prompt: str) -> Optional[str]:
        try:
            self._initialize_client()
            self._apply_rate_limit()
            
            print(f"  -> Sending code to OpenAI model ({self.model}) for refactoring...", flush=True)
            
            sys.stdout.flush()
            sys.stderr.flush()
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=self.timeout
            )
            
            if not response or not response.choices:
                print("  -> Warning: Empty response from OpenAI", flush=True)
                return None
            
            response_text = response.choices[0].message.content
            
            print("  -> OpenAI response received.", flush=True)
            return response_text
            
        except Exception as e:
            print(f"  -> Warning: OpenAI failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return None


def run_tests(code, test_cases):
    passed_count = 0
    total_execution_time = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        class_match = re.search(r'public\s+class\s+(\w+)', code)
        if not class_match:
            class_match = re.search(r'class\s+(\w+)', code)
            
        if not class_match:
            return 0, 0
        
        class_name = class_match.group(1)
        java_file = os.path.join(tmpdir, f"{class_name}.java")
        
        with open(java_file, "w", encoding='utf-8') as f:
            f.write(code)
        
        try:
            compile_process = subprocess.run(
                ["javac", java_file],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir
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
                    ["java", "-cp", ".", class_name],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=tmpdir
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
    parser = argparse.ArgumentParser(description="Debug Java code using GPT-5.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    model_name = "gpt-5"
    adapter = OpenAIAdapter(model=model_name, timeout=300, rate_limit_delay=1.0)

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

    print(f"Using {model_name} for Java Debugging")

    checkpoint_file = f"{model_name}_java_checkpoint.json"
    output_file = f"{model_name}_java_results.json"
    
    results = []
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

    print(f"Processing {len(dataset)} items...")

    item_counter = 0
    for idx, item in enumerate(dataset, 1):
        if idx - 1 < start_idx:
            continue
            
        if item.get("code_language") == "java":
            item_counter += 1
            item_idx = item.get('idx')
            bug_type = item.get("type", "unknown issue")
            
            print(f"\n{'='*60}")
            print(f"Processing item {item_counter}/{len(dataset)} (ID: {item_idx})")
            print(f"Bug Type: {bug_type}")

            incorrect_solution = item.get("incorrect_solutions", "")
            correct_solution = item.get("solutions", "")

            if not incorrect_solution or not correct_solution:
                print(f"Warning: Missing solution data. Skipping.")
                continue

            for prefix in ["```java\n", "```java", "java\n"]:
                if incorrect_solution.startswith(prefix):
                    incorrect_solution = incorrect_solution[len(prefix):]
                    break
            if incorrect_solution.endswith("```"): incorrect_solution = incorrect_solution[:-3]

            for prefix in ["```java\n", "```java", "java\n"]:
                if correct_solution.startswith(prefix):
                    correct_solution = correct_solution[len(prefix):]
                    break
            if correct_solution.endswith("```"): correct_solution = correct_solution[:-3]

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
                print("Warning: No test cases found. Skipping.")
                continue

            prompt = f"""Fix the bugs in this Java code. Return ONLY the corrected code in a ```java code block, with NO explanation.

Incorrect Code:
```java
{incorrect_solution}

Bug type: {bug_type}

Corrected code:"""

            response_text = adapter.generate(prompt)

            if response_text is None:
                results.append({"id": item.get("idx"), "status": "no response", "pass_rate": 0})
                with open(checkpoint_file, "w") as f:
                    json.dump({"last_processed_index": idx, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
                continue

            debugged_code = ""
            code_match = re.search(r"```java\n(.*?)```", response_text, re.DOTALL)
            if not code_match: code_match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)
            if not code_match: code_match = re.search(r"```(.*?)```", response_text, re.DOTALL)

            if code_match:
                debugged_code = code_match.group(1).strip()
            else:
                debugged_code = response_text.strip()

            if not debugged_code:
                print("Warning: No Java code block found.")
                results.append({"id": item.get("idx"), "status": "no response", "pass_rate": 0})
                continue

            print(f"Compiling and running tests...")
            passed_public_count, public_exec_time = run_tests(debugged_code, public_test_cases)
            passed_private_count, private_exec_time = run_tests(debugged_code, private_test_cases)

            total_passed = passed_public_count + passed_private_count
            total_tests = len(public_test_cases) + len(private_test_cases)
            pass_rate = total_passed / total_tests if total_tests > 0 else 0
            
            loc = len(debugged_code.splitlines())
            
            cc = -1
            cognitive_complexity = -1
            code_smells = False
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.java', delete=False, encoding='utf-8') as temp_f:
                temp_filename = temp_f.name
                temp_f.write(debugged_code)
            
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
            finally:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)

            codebleu_value = -1
            try:
                cb = codebleu.calc_codebleu([correct_solution], [debugged_code], lang="java", weights=(0.25, 0.25, 0.25, 0.25))
                codebleu_value = cb['codebleu']
            except Exception:
                codebleu_value = -1

            status = "no fixes"
            if pass_rate == 1.0: status = "fully correct"
            elif pass_rate > 0: status = "partially correct"

            print(f"Status: {status}, Pass rate: {pass_rate:.2%} ({total_passed}/{total_tests})")

            results.append({
                "id": item.get("idx"),
                "status": status,
                "pass_rate": pass_rate,
                "public_tests_passed": passed_public_count,
                "private_tests_passed": passed_private_count,
                "efficiency": (public_exec_time + private_exec_time) / total_passed if total_passed > 0 else 0,
                "hallucination_rate": 1.0 - pass_rate,
                "codebleu": codebleu_value,
                "code_smells": code_smells,
                "cyclomatic_complexity": cc,
                "loc": loc,
                "cognitive_complexity": cognitive_complexity,
                "model_response": response_text,
                "debugged_code": debugged_code,
            })
            
            with open(checkpoint_file, "w") as f:
                json.dump({"last_processed_index": idx, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
            
            with open(output_file, "w") as f:
                json.dump({"results": results}, f, indent=4)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    if results:
        passed_fully = sum(1 for r in results if r["status"] == "fully correct")
        print("\n" + "=" * 60)
        print(f"SUMMARY: {model_name}")
        print(f"Processed: {len(results)}")
        print(f"Fully Correct: {passed_fully}")
        print("=" * 60)

    with open(output_file, "w") as f:
        json.dump({"results": results}, f, indent=4)

    print(f"\nDebugging complete. Results saved to {output_file}")

if __name__ == "__main__": 
    main()
