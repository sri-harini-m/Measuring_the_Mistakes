import json
import subprocess
import argparse
import os
import re
import time
import tempfile
import ast
import sys
import traceback
from typing import Optional

from radon.complexity import cc_visit
from radon.metrics import h_visit
import codebleu
from cognitive_complexity.api import get_cognitive_complexity
import lizard


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
            print("\n" + "="*80, flush=True)
            print("MODEL OUTPUT:", flush=True)
            print("="*80, flush=True)
            print(response_text, flush=True)
            print("="*80 + "\n", flush=True)
            return response_text
            
        except Exception as e:
            print(f"  -> Warning: OpenAI failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return None


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
    parser = argparse.ArgumentParser(description="Debug Python code using OpenAI Adapter.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    model_name = "gpt-5"
    adapter = OpenAIAdapter(model=model_name, timeout=300, rate_limit_delay=1.0)

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
                    continue

    if not dataset:
        print("Error: No valid data found in dataset")
        return

    print(f"Using {model_name} via OpenAIAdapter")

    checkpoint_file = f"{model_name}_python3_checkpoint.json"
    output_file = f"{model_name}_python3_results.json"
    
    results = []
    start_idx = 0
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                start_idx = checkpoint_data.get("last_processed_index", 0)
                print(f"Resuming from checkpoint at index {start_idx}")
        except Exception:
            pass
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_data = json.load(f)
                results = existing_data.get("results", [])
        except Exception:
            pass

    print(f"Processing {len(dataset)} items...")

    item_counter = 0
    for idx, item in enumerate(dataset, 1):
        if idx - 1 < start_idx:
            continue
            
        if item.get("code_language") == "python3":
            item_counter += 1
            item_idx = item.get('idx')
            bug_type = item.get("type", "unknown issue")
            
            print(f"\nProcessing item {item_counter}/{len(dataset)} (ID: {item_idx}) - Bug: {bug_type}")

            incorrect_solution = item.get("incorrect_solutions", "").strip()
            
            if incorrect_solution.startswith("```"): incorrect_solution = incorrect_solution.strip("`").replace("python\n", "").strip()

            correct_solution = item.get("solutions", "").strip()
            if correct_solution.startswith("```"): correct_solution = correct_solution.strip("`").replace("python\n", "").strip()

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
                continue

            prompt = f"""Fix the bugs in this Python code. Return ONLY the corrected code in a ```python code block, with NO explanation.

Incorrect Code:
```python
{incorrect_solution}

Bug type: {bug_type}

Corrected code:"""

            response_text = adapter.generate(prompt)

            if response_text is None:
                results.append({
                    "id": item.get("idx"),
                    "status": "no response",
                    "pass_rate": 0,
                    "codebleu": -1,
                })
                with open(checkpoint_file, "w") as f:
                    json.dump({"last_processed_index": idx, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
                continue

            debugged_code = ""
            code_match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
            if not code_match: code_match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)
            if not code_match: code_match = re.search(r"```(.*?)```", response_text, re.DOTALL)

            if code_match:
                debugged_code = code_match.group(1).strip()
            else:
                debugged_code = response_text.strip()

            print(f"Running tests...")
            passed_public, pub_time = run_tests(debugged_code, public_test_cases)
            passed_private, priv_time = run_tests(debugged_code, private_test_cases)
            
            total_passed = passed_public + passed_private
            total_tests = len(public_test_cases) + len(private_test_cases)
            pass_rate = total_passed / total_tests if total_tests > 0 else 0

            cc = -1
            try:
                cc = sum(c.complexity for c in cc_visit(debugged_code)) if debugged_code else -1
            except Exception: pass

            codebleu_val = -1
            try:
                cb = codebleu.calc_codebleu([correct_solution], [debugged_code], lang="python", weights=(0.25, 0.25, 0.25, 0.25))
                codebleu_val = cb['codebleu']
            except Exception: pass

            status = "no fixes"
            if pass_rate == 1.0: status = "fully correct"
            elif pass_rate > 0: status = "partially correct"

            print(f"Status: {status}, Pass Rate: {pass_rate:.2%}")

            results.append({
                "id": item.get("idx"),
                "status": status,
                "pass_rate": pass_rate,
                "public_tests_passed": passed_public,
                "private_tests_passed": passed_private,
                "hallucination_rate": 1.0 - pass_rate,
                "codebleu": codebleu_val,
                "cyclomatic_complexity": cc,
                "debugged_code": debugged_code,
                "model_response": response_text
            })

            with open(checkpoint_file, "w") as f:
                json.dump({"last_processed_index": idx, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
            
            with open(output_file, "w") as f:
                json.dump({"results": results}, f, indent=4)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    if results:
        passed = sum(1 for r in results if r["status"] == "fully correct")
        print(f"\nCompleted. Fully correct: {passed}/{len(results)}")

if __name__ == "__main__": 
    main()
