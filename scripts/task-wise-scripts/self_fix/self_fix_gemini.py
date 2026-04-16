import os
import json
import argparse
import subprocess
import tempfile
import re
import time
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any


class CodeExecutor:
    def __init__(self, language: str):
        self.language = language

    def clean_code(self, code: str) -> str:
        code = code.strip()
        pattern = r"```[a-zA-Z]*\n(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1).strip()

        lines = code.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def prepare_file(self, code: str, temp_dir: str) -> str:
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

    def compile(self, file_path: str):
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

    def run_test_case(self, file_path: str, input_str: str):
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
                timeout=2,
            )
            return process.stdout.strip(), process.stderr.strip(), False
        except subprocess.TimeoutExpired:
            return "", "Time Limit Exceeded", True
        except Exception as e:
            return "", str(e), False


class GeminiAdapter:

    def __init__(self, model: str = "gemini-2.5-pro", timeout: int = 300, rate_limit_delay: float = 1.0):
        self.model = model
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self._client = None
        self._generation_config = None
        self._project_id = None
        self._location = None

    def _initialize_client(self):
        if self._client is not None:
            return
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, GenerationConfig

            self._project_id = os.getenv("GOOGLE_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID")
            self._location = os.getenv("GOOGLE_LOCATION") or os.getenv("VERTEX_LOCATION") or "us-central1"
            
            if not self._project_id:
                raise ValueError("GOOGLE_PROJECT_ID or VERTEX_PROJECT_ID environment variable must be set")

            vertexai.init(project=self._project_id, location=self._location)

            self._client = GenerativeModel(self.model)

            self._generation_config = GenerationConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
            print(
                f"  -> Initialized Vertex AI with project: {self._project_id}, location: {self._location}"
            )
            print(
                f"  -> Using model: {self.model}, "
                f"temperature=0.0, top_p=1.0, top_k=1"
            )
        except ImportError:
            raise ImportError("google-cloud-aiplatform package not installed. Install with: pip install google-cloud-aiplatform")

    def _apply_rate_limit(self) -> None:
        if self.rate_limit_delay <= 0:
            return
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            print(f"  -> Rate limiting: waiting {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _messages_to_prompt(self, messages: list) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            parts.append(f"{role}:\n{content}\n")
        return "\n".join(parts).strip()

    def generate(self, messages: list) -> Optional[str]:
        max_retries = float('inf')  # Retry indefinitely for 503 errors
        retry_count = 0
        base_delay = 5  # Start with 5 seconds
        max_delay = 300  # Cap at 5 minutes
        
        while retry_count < max_retries:
            try:
                self._initialize_client()
                self._apply_rate_limit()

                prompt = self._messages_to_prompt(messages)
                print(f"  -> Sending code to Vertex AI Gemini model ({self.model})...", flush=True)

                response = self._client.generate_content(
                    contents=prompt,
                    generation_config=self._generation_config,
                )

                if not response:
                    print("  -> Warning: Empty response from Vertex AI", flush=True)
                    return None

                response_text = response.text if hasattr(response, 'text') else None

                if not response_text or not response_text.strip():
                    print("  -> Warning: Response text is empty or whitespace only", flush=True)
                    return None

                print(f"  -> Content length: {len(response_text)}", flush=True)
                return response_text

            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                
                is_503 = '503' in error_str or 'Service Unavailable' in error_str or 'unavailable' in error_str.lower()
                
                if is_503:
                    retry_count += 1
                    delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
                    print(f"  -> 503 Service Unavailable error (attempt {retry_count}). Retrying in {delay} seconds...", flush=True)
                    time.sleep(delay)
                    continue
                else:
                    print(f"  -> Warning: Vertex AI request failed: {error_type}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    return None
        
        return None


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
        error_groups: Dict[str, Any] = {}

        if total == 0:
            return 0.0, 0, 0, "No test cases found."

        for i, (inp, expected) in enumerate(zip(inputs, outputs)):
            actual, err, timeout = executor.run_test_case(file_path, inp)

            error_key = ""
            if timeout:
                error_key = "Time Limit Exceeded"
            elif err:
                clean_err = re.sub(r"/tmp/tmp[a-zA-Z0-9_]+/", "", err.strip())
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
                if count > 5:
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


def process_folder(
    base_folder: str,
    language: str,
    output_dir: str = "./gemini_outputs",
    model: str = "gemini-2.5-pro",
    rate_limit_delay: float = 1.0,
    single_datapoint: Optional[str] = None,
):
    adapter = GeminiAdapter(model=model, rate_limit_delay=rate_limit_delay)

    dp_list = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    if single_datapoint:
        if single_datapoint in dp_list:
            dp_list = [single_datapoint]
            print(f"Running single datapoint: {single_datapoint}")
        else:
            print(f"Error: Datapoint '{single_datapoint}' not found in {base_folder}")
            return

    successful_dps = 0

    for dp_name in tqdm(dp_list, desc="Processing datapoints", unit="dp"):
        try:
            dp_path = os.path.join(base_folder, dp_name)

            question_output_dir = os.path.join(output_dir, dp_name)
            os.makedirs(question_output_dir, exist_ok=True)

            history_file_path = os.path.join(question_output_dir, "history.json")

            q_path = os.path.join(dp_path, "question.txt")
            if not os.path.exists(q_path):
                continue
            with open(q_path, "r") as f:
                question_text = f.read().strip()

            io_path = os.path.join(dp_path, "input_output.json")
            if not os.path.exists(io_path):
                continue
            with open(io_path, "r") as f:
                io_data = json.load(f)

            metadata: Dict[str, Any] = {}
            meta_path = os.path.join(dp_path, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                with open(os.path.join(question_output_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

            history_log: list[Dict[str, Any]] = []
            attempt_start = 0

            if os.path.exists(history_file_path):
                try:
                    with open(history_file_path, "r") as f:
                        existing_data = json.load(f)
                        existing_history = existing_data.get("history", [])
                    attempts_done = sum(1 for attempt in existing_history 
                                        if attempt.get("code") and len(attempt.get("code", "").strip()) > 0)
                    if attempts_done >= 5:
                        print(f"   > Skipping {dp_name} (already has {attempts_done} attempts)")
                        successful_dps += 1
                        continue

                    attempt_start = attempts_done
                    print(
                        f"   > Resuming {dp_name}: already has {attempts_done} attempts, "
                        f"starting from attempt {attempt_start + 1}/5"
                    )
                except Exception as e:
                    print(f"   > Warning: Could not load existing history for {dp_name}: {e}. Restarting from scratch.")
                    history_log = []
                    attempt_start = 0

            system_prompt = (
                "You are an expert programmer. You will be given a coding problem. "
                "Output only the code, with no explanations or comments."
            )
            user_prompt = f"QUESTION:\n{question_text}\n"
            if language == "Java":
                user_prompt += " Name your main class 'Main'."
            user_prompt += f"\nWrite a {language} program for the above problem."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            if history_log:
                for past in history_log:
                    prev_code = past.get("code", "") or ""
                    prev_feedback = past.get("feedback", "") or ""

                    if prev_code.strip():
                        messages.append({"role": "assistant", "content": prev_code})
                    else:
                        messages.append({"role": "assistant", "content": "Failed to generate code"})

                    messages.append({"role": "user", "content": f"USER_FEEDBACK: {prev_feedback}"})

            for attempt in range(attempt_start, 5):
                print(f"   > Attempt {attempt+1}/5 for {dp_name}...")

                generated_code = adapter.generate(messages)

                if not generated_code:
                    print("     > Error: Failed to generate code")
                    result_entry = {
                        "attempt": attempt + 1,
                        "code": "",
                        "pass_rate": 0.0,
                        "passed": 0,
                        "total": 0,
                        "feedback": "Failed to generate code from model",
                    }
                    history_log.append(result_entry)

                    messages.append({"role": "assistant", "content": "Failed to generate code"})
                    messages.append({"role": "user", "content": "Please try again and output only valid code."})
                    continue

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
            successful_dps += 1
            print(f"   > Completed {dp_name}")

        except Exception as e:
            print(f"\nError processing {dp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nProcessing Complete. Processed {successful_dps} datapoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run iterative code generation with Gemini via Vertex AI.")
    parser.add_argument("--folder_name", type=str, required=True, help="Path to folder containing coding problems")
    parser.add_argument("--language", type=str, choices=["Python", "Java", "C++"], required=True, help="Programming language")
    parser.add_argument("--output_dir", type=str, default="./gemini_outputs", help="Output directory for results")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Vertex AI Gemini model name (e.g., gemini-2.5-pro)")
    parser.add_argument("--rate_limit_delay", type=float, default=1.0, help="Rate limit delay in seconds between API calls")
    parser.add_argument("--single_datapoint", type=str, default=None, help="Run only a specific datapoint (e.g., '3005')")

    args = parser.parse_args()

    print("Starting Gemini (Vertex AI) Self-Fixing Code Generation")
    print(f"  Model: {args.model}")
    print(f"  Language: {args.language}")
    print(f"  Input folder: {args.folder_name}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Rate limit delay: {args.rate_limit_delay}s")
    print()

    process_folder(
        args.folder_name,
        args.language,
        args.output_dir,
        args.model,
        args.rate_limit_delay,
        args.single_datapoint,
    )
