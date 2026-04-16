import os
import json
import argparse
import subprocess
import tempfile
import re
import time
import sys
import asyncio
from tqdm.asyncio import tqdm as atqdm
from typing import Optional, Tuple, Dict, Any


class RateLimitQuotaExceeded(Exception):
    pass


class OpenAIChatAdapter:
    def __init__(self, model: str = "gpt-5", timeout: int = 600, rate_limit_delay: float = 1.0, max_concurrent: int = 5):
        self.model = model
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_concurrent = max_concurrent
        self.last_request_time = 0.0
        self.rate_limit_lock = asyncio.Lock()
        
        try:
            from openai import AsyncOpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self._client = AsyncOpenAI(api_key=api_key)
            print(f"  -> Initialized AsyncOpenAI client with model: {self.model}")
            print(f"  -> Max concurrent requests: {self.max_concurrent}")
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai>=1.0.0")
    
    async def close(self):
        await self._client.close()
    
    async def _apply_rate_limit(self) -> None:
        if self.rate_limit_delay > 0:
            async with self.rate_limit_lock:
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                
                if time_since_last_request < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - time_since_last_request
                    await asyncio.sleep(sleep_time)
                
                self.last_request_time = time.time()
    
    async def generate(self, messages: list) -> Optional[str]:
        try:
            await self._apply_rate_limit()
            
            total_chars = sum(len(str(m.get('content', ''))) for m in messages)
            print(f"  -> Sending {len(messages)} messages ({total_chars:,} chars) to OpenAI model ({self.model})...", flush=True)
            
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=self.timeout
            )
            
            if not response or not response.choices:
                print("  -> Warning: Empty response from OpenAI", flush=True)
                return None
            
            response_text = response.choices[0].message.content
            
            if response_text:
                print(f"  -> Content length: {len(response_text)}", flush=True)
            
            return response_text
            
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            if ('rate_limit' in error_type.lower() or 
                'ratelimit' in error_type.lower() or
                'quota' in error_str or 
                'insufficient_quota' in error_str or
                'rate limit' in error_str or
                '429' in error_str):
                
                print(f"\n{'='*60}", flush=True)
                print(f"RATE LIMIT QUOTA EXCEEDED", flush=True)
                print(f"Error: {type(e).__name__}: {e}", flush=True)
                print(f"{'='*60}\n", flush=True)
                
                raise RateLimitQuotaExceeded(f"Rate limit quota exceeded: {e}")
            
            print(f"  -> Warning: OpenAI failed: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None


class CodeExecutor:
    def __init__(self, language):
        self.language = language

    def clean_code(self, code):
        code = code.strip()
        pattern = r"```[a-zA-Z]*\n(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        lines = code.split('\n')
        if len(lines) > 0 and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if len(lines) > 0 and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def prepare_file(self, code, temp_dir):
        code = self.clean_code(code)
        
        if self.language == "Python":
            filename = "solution.py"
        elif self.language == "Java":
            filename = "Main.java"
            if "public class Main" not in code and "class Main" not in code:
                code = code.replace("public class Solution", "public class Main")
                code = code.replace("class Solution", "class Main")
        elif self.language == "C++":
            filename = "solution.cpp"
        else:
            raise ValueError(f"Unsupported language: {self.language}")

        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "w") as f:
            f.write(code)
        return file_path

    def compile(self, file_path):
        if self.language == "Python":
            return True, ""
        
        if self.language == "C++":
            out_file = file_path.replace(".cpp", "")
            cmd = ["g++", "-O3", file_path, "-o", out_file]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                compile_err = res.stderr
                if len(compile_err) > 2000:
                    compile_err = compile_err[:2000] + "\n...[truncated]..."
                return False, f"Compilation Error:\n{compile_err}"
            return True, ""
        
        if self.language == "Java":
            cmd = ["javac", file_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                compile_err = res.stderr
                if len(compile_err) > 2000:
                    compile_err = compile_err[:2000] + "\n...[truncated]..."
                return False, f"Compilation Error:\n{compile_err}"
            return True, ""
            
        return False, "Unknown language"

    def run_test_case(self, file_path, input_str):
        cmd = []
        if self.language == "Python":
            cmd = ["python3", file_path]
        elif self.language == "C++":
            cmd = [file_path.replace(".cpp", "")]
        elif self.language == "Java":
            wd = os.path.dirname(file_path)
            cmd = ["java", "-cp", wd, "Main"]

        try:
            process = subprocess.run(
                cmd,
                input=input_str,
                capture_output=True,
                text=True,
                timeout=2 
            )
            return process.stdout.strip(), process.stderr.strip(), False
        except subprocess.TimeoutExpired:
            return "", "Time Limit Exceeded", True
        except Exception as e:
            return "", str(e), False


def evaluate_solution(code: str, language: str, io_data: Dict[str, Any]) -> Tuple[float, int, int, str]:
    if not code or not code.strip():
        return 0.0, 0, 0, "Error: Model generated empty code."
    
    executor = CodeExecutor(language)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = executor.prepare_file(code, tmp_dir)
        
        compiled, compile_msg = executor.compile(file_path)
        if not compiled:
            return 0.0, 0, 0, compile_msg

        inputs = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])
        
        passed = 0
        total = len(inputs)
        
        error_groups = {} 

        if total == 0:
            return 0.0, 0, 0, "No test cases found."

        for i, (inp, expected) in enumerate(zip(inputs, outputs)):
            actual, err, timeout = executor.run_test_case(file_path, inp)
            
            error_key = ""
            if timeout:
                error_key = "Time Limit Exceeded"
            elif err:
                clean_err = re.sub(r'/tmp/tmp[a-zA-Z0-9_]+/', '', err.strip())
                if len(clean_err) > 500:
                    clean_err = clean_err[:500] + "...[truncated]"
                error_key = f"Runtime Error:\n{clean_err}"
            elif actual.strip() != expected.strip():
                exp_short = expected.strip()[:100]
                act_short = actual.strip()[:100]
                error_key = f"Wrong Answer. Expected: '{exp_short}...', Got: '{act_short}...'"

            if error_key:
                if error_key not in error_groups:
                    error_groups[error_key] = []
                error_groups[error_key].append(i + 1)
            else:
                passed += 1

        pass_rate = (passed / total) * 100
        
        if passed == total:
            feedback = "No errors. All test cases passed. Please output the code."
        else:
            feedback = f"The code passed {passed}/{total} test cases.\n\nErrors encountered:"
            count = 0
            for err_msg, cases in error_groups.items():
                if count > 5:  # Limit distinct error types to save tokens
                    feedback += "\n\n...[other errors truncated]..."
                    break
                
                if len(cases) > 10:
                    case_str = f"Test Cases {cases[0]} to {cases[-1]} ({len(cases)} total)"
                else:
                    case_str = f"Test Cases {', '.join(map(str, cases))}"
                
                feedback += f"\n\n{case_str}:\n{err_msg}"
                count += 1

            feedback += "\n\nPlease fix these errors and output the corrected code."

        return pass_rate, passed, total, feedback


async def process_datapoint(dp_name: str, base_folder: str, language: str, output_dir: str, adapter: OpenAIChatAdapter) -> bool:
    try:
        dp_path = os.path.join(base_folder, dp_name)
        
        question_output_dir = os.path.join(output_dir, dp_name)
        os.makedirs(question_output_dir, exist_ok=True)
        
        history_file_path = os.path.join(question_output_dir, "history.json")
        
        history_log = []
        start_attempt = 0
        existing_history = None
        
        if os.path.exists(history_file_path):
            try:
                with open(history_file_path, "r") as f:
                    existing_history = json.load(f)
                
                history_log = existing_history.get("history", [])
                completed_attempts = len(history_log)
                
                incomplete_attempts = []
                for i, entry in enumerate(history_log):
                    if not isinstance(entry, dict):
                        incomplete_attempts.append(i+1)
                    elif "attempt" not in entry or "code" not in entry or "pass_rate" not in entry:
                        incomplete_attempts.append(i+1)
                
                if incomplete_attempts:
                    print(f"   > Warning: Incomplete history entries for {dp_name}: {incomplete_attempts}. Rerunning from attempt {incomplete_attempts[0]}")
                    start_attempt = incomplete_attempts[0] - 1
                    history_log = history_log[:start_attempt]
                    completed_attempts = start_attempt
                
                ext = ".py" if language == "Python" else ".cpp" if language == "C++" else ".java"
                missing_files = []
                for i in range(completed_attempts):
                    attempt_file = os.path.join(question_output_dir, f"attempt_{i+1}{ext}")
                    if not os.path.exists(attempt_file):
                        missing_files.append(i+1)
                
                if missing_files:
                    print(f"   > Warning: Missing attempt files for {dp_name}: {missing_files}. Rerunning from attempt {missing_files[0]}")
                    start_attempt = missing_files[0] - 1
                    history_log = history_log[:start_attempt]
                elif completed_attempts >= 5:
                    print(f"   > Skipping {dp_name} (all 5 attempts completed)")
                    return True
                else:
                    start_attempt = completed_attempts
                    print(f"   > Resuming {dp_name} from attempt {start_attempt + 1}/5 ({completed_attempts} completed)")
            except Exception as e:
                print(f"   > Warning: Could not load history for {dp_name}, starting fresh: {e}")
                history_log = []
                start_attempt = 0

        q_path = os.path.join(dp_path, "question.txt")
        if not os.path.exists(q_path):
            return False
        with open(q_path, "r") as f:
            question_text = f.read().strip()

        io_path = os.path.join(dp_path, "input_output.json")
        if not os.path.exists(io_path):
            return False
        with open(io_path, "r") as f:
            io_data = json.load(f)

        metadata = {}
        meta_path = os.path.join(dp_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            with open(os.path.join(question_output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

        system_prompt = "You are an expert programmer. You will be given a coding problem. Output only the code, with no explanations or comments."
        user_prompt = f"QUESTION:\n{question_text}\n"
        if language == "Java":
            user_prompt += " Name your main class 'Main'."
        user_prompt += f"\nWrite a {language} program for the above problem."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if start_attempt > 0:
            for prev_attempt in history_log:
                prev_code = prev_attempt.get("code", "")
                prev_feedback = prev_attempt.get("feedback", "")
                
                messages.append({"role": "assistant", "content": prev_code})
                messages.append({"role": "user", "content": f"USER_FEEDBACK: {prev_feedback}"})

        for attempt in range(start_attempt, 5):
            print(f"   > Attempt {attempt+1}/5 for {dp_name}...")
            
            if len(messages) > 6:
                messages = [messages[0], messages[1], messages[-2], messages[-1]]
            
            generated_code = await adapter.generate(messages)
            
            if not generated_code or not generated_code.strip():
                print(f"     > Error: Failed to generate code (empty response)")
                
                feedback_msg = "Failed to generate code from model. "
                if attempt > 0 and not history_log[-1].get("code"):
                    feedback_msg += "Please ensure you output valid, complete code without truncation."
                else:
                    feedback_msg += "Please try again and output only valid code."
                
                result_entry = {
                    "attempt": attempt + 1,
                    "code": "",
                    "pass_rate": 0.0,
                    "passed": 0,  
                    "total": 0,
                    "feedback": feedback_msg
                }
                history_log.append(result_entry)
                
                messages.append({"role": "assistant", "content": ""})
                messages.append({"role": "user", "content": feedback_msg})
                
                final_output = {
                    "datapoint": dp_name,
                    "language": language,
                    "history": history_log,
                }
                with open(history_file_path, "w") as f:
                    json.dump(final_output, f, indent=2)
                
                continue  # Skip to next attempt after empty response
            
            pass_rate, passed, total, feedback = evaluate_solution(generated_code, language, io_data)

            print(f"     > Result: {passed}/{total} passed ({pass_rate:.1f}%)")

            result_entry = {
                "attempt": attempt + 1,
                "code": generated_code,
                "pass_rate": pass_rate,
                "passed": passed,
                "total": total,
                "feedback": feedback,
            }
            history_log.append(result_entry)

            ext = ".py" if language == "Python" else ".cpp" if language == "C++" else ".java"
            code_filename = f"attempt_{attempt+1}{ext}"
            with open(os.path.join(question_output_dir, code_filename), "w") as f:
                f.write(CodeExecutor(language).clean_code(generated_code))

            messages.append({"role": "assistant", "content": generated_code})
            messages.append({"role": "user", "content": f"USER_FEEDBACK: {feedback}"})

            final_output = {
                "datapoint": dp_name,
                "language": language,
                "history": history_log,
            }

            with open(history_file_path, "w") as f:
                json.dump(final_output, f, indent=2)

        print(f"   > Completed {dp_name}")
        return True
        
    except RateLimitQuotaExceeded as e:
        print(f"\n{'='*60}", flush=True)
        print(f"TERMINATING: Rate limit quota exceeded", flush=True)
        print(f"Error: {e}", flush=True)
        print(f"{'='*60}\n", flush=True)
        raise  # Re-raise to propagate to main
    except Exception as e:
        print(f"\nError processing {dp_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def process_folder(base_folder: str, language: str, output_dir: str = "./gpt5_outputs", model: str = "gpt-5", rate_limit_delay: float = 1.0, single_datapoint: Optional[str] = None, timeout: int = 600, max_concurrent: int = 5):
    
    adapter = OpenAIChatAdapter(model=model, timeout=timeout, rate_limit_delay=rate_limit_delay, max_concurrent=max_concurrent)
    
    dp_list = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    
    if single_datapoint:
        if single_datapoint in dp_list:
            dp_list = [single_datapoint]
            print(f"Running single datapoint: {single_datapoint}")
        else:
            print(f"Error: Datapoint '{single_datapoint}' not found in {base_folder}")
            return
    
    print(f"Processing {len(dp_list)} datapoints with max {max_concurrent} concurrent requests...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(dp_name):
        async with semaphore:
            return await process_datapoint(dp_name, base_folder, language, output_dir, adapter)
    
    successful_dps = 0
    try:
        tasks = [process_with_semaphore(dp_name) for dp_name in dp_list]
        results = []
        
        for coro in atqdm.as_completed(tasks, desc="Processing datapoints", unit="dp", total=len(tasks)):
            try:
                result = await coro
                results.append(result)
                if result:
                    successful_dps += 1
            except RateLimitQuotaExceeded:
                print(f"\nTerminating due to rate limit quota exceeded")
                print(f"Completed {successful_dps} datapoints before quota limit.")
                await adapter.close()
                sys.exit(1)
    except RateLimitQuotaExceeded:
        print(f"\nTerminating due to rate limit quota exceeded")
        print(f"Completed {successful_dps} datapoints before quota limit.")
        await adapter.close()
        sys.exit(1)

    await adapter.close()
    print(f"\nProcessing Complete. Successfully processed {successful_dps}/{len(dp_list)} datapoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run iterative code generation with GPT-5 using async API.")
    parser.add_argument("--folder_name", type=str, required=True, help="Path to folder containing coding problems")
    parser.add_argument("--language", type=str, choices=["Python", "Java", "C++"], required=True, help="Programming language")
    parser.add_argument("--output_dir", type=str, default="./gpt5_outputs", help="Output directory for results")
    parser.add_argument("--model", type=str, default="gpt-5", help="GPT model to use (default: gpt-5)")
    parser.add_argument("--rate_limit_delay", type=float, default=1.0, help="Rate limit delay in seconds between API calls")
    parser.add_argument("--timeout", type=int, default=600, help="API timeout in seconds (default: 600)")
    parser.add_argument("--single_datapoint", type=str, default=None, help="Run only a specific datapoint (e.g., '3005')")
    parser.add_argument("--max_concurrent", type=int, default=5, help="Maximum concurrent API requests (default: 5)")
    
    args = parser.parse_args()
    
    print(f"Starting GPT-5 Self-Fixing Code Generation (Async)")
    print(f"  Model: {args.model}")
    print(f"  Language: {args.language}")
    print(f"  Input folder: {args.folder_name}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Rate limit delay: {args.rate_limit_delay}s")
    print(f"  API timeout: {args.timeout}s")
    print(f"  Max concurrent requests: {args.max_concurrent}")
    print()
    
    asyncio.run(process_folder(
        args.folder_name,
        args.language,
        args.output_dir,
        args.model,
        args.rate_limit_delay,
        args.single_datapoint,
        args.timeout,
        args.max_concurrent
    ))
