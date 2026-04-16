import os
import json
import torch
import argparse
import subprocess
import tempfile
import re
import gc
import sys
import importlib.util
from tqdm import tqdm

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import types
flash_attn_module = types.ModuleType('flash_attn')
flash_attn_module.flash_attn_func = None
flash_attn_module.flash_attn_varlen_func = None
flash_attn_spec = importlib.util.spec_from_loader('flash_attn', loader=None)
flash_attn_module.__spec__ = flash_attn_spec
flash_attn_module.__version__ = "2.0.0"  # Mock version
sys.modules['flash_attn'] = flash_attn_module
sys.modules['flash_attn.flash_attn_interface'] = flash_attn_module

from transformers import AutoTokenizer, AutoModelForCausalLM


class CodeExecutor:
    def __init__(self, language):
        self.language = language

    def clean_code(self, code):
        code = code.strip()
        pattern = r"```[a-zA-Z]*\n(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1)
        return code

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
                return False, f"Compilation Error:\n{res.stderr}"

            return True, ""

        if self.language == "Java":
            cmd = ["javac", file_path]
            res = subprocess.run(cmd, capture_output=True, text=True)

            if res.returncode != 0:
                return False, f"Compilation Error:\n{res.stderr}"

            return True, ""

        return False, "Unknown language"

    def run_test_case(self, file_path, input_str):

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


def run_model(messages, model, tokenizer, device):

    try:
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass  # GPU might be in bad state from previous error

    max_position_embeddings = getattr(model.config, "max_position_embeddings", 16384)
    max_input_length = min(max_position_embeddings - 1024, 8192)

    if len(messages) > 11:
        print(f"   > Conversation too long ({len(messages)} messages). Trimming...")
        messages = [messages[0], messages[1]] + messages[-6:]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True
        )
        
        vocab_size = len(tokenizer)
        if torch.any(inputs >= vocab_size) or torch.any(inputs < 0):
            print(f"   > Warning: Token IDs out of range detected. Clamping to valid range [0, {vocab_size-1}]")
            inputs = torch.clamp(inputs, 0, vocab_size - 1)
        
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            if tokenizer.bos_token_id >= vocab_size:
                print(f"   > Warning: bos_token_id ({tokenizer.bos_token_id}) exceeds vocab size ({vocab_size})")
                tokenizer.bos_token_id = tokenizer.eos_token_id
        
        try:
            inputs = inputs.to(device)
        except RuntimeError as cuda_err:
            if "CUDA" in str(cuda_err):
                print(f"   > CUDA error when moving tokens to device: {cuda_err}")
                print(f"   > This indicates GPU state corruption. Exiting to allow restart.")
                sys.exit(1)
            raise

    except Exception as e:
        error_msg = str(e)
        print(f"   > Error tokenizing input: {error_msg}")
        if "CUDA" in error_msg.upper() or "assert" in error_msg.lower():
            print(f"   > CUDA error detected during tokenization. Exiting for fresh restart.")
            sys.exit(1)
        return ""

    try:
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        response = outputs[0][inputs.shape[-1]:]
        decoded = tokenizer.decode(response, skip_special_tokens=True)

        del inputs
        del outputs
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

        return decoded
        
    except Exception as e:
        error_str = str(e)
        if "assert" in error_str.lower():
            print(f"   > CUDA assertion error detected. GPU state corrupted.")
            print(f"   > Exiting program. Restart to continue with fresh GPU state.")
            print(f"   > Completed datapoints will be skipped on restart.")
            import sys
            sys.exit(1)
        else:
            print(f"   > Error generating code: {error_str[:200]}")
        
        try:
            if 'inputs' in locals():
                del inputs
        except:
            pass
        try:
            if 'outputs' in locals():
                del outputs
        except:
            pass
        try:
            gc.collect()
        except:
            pass
        
        return ""


def evaluate_solution(code, language, io_data):

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

            for err_msg, cases in error_groups.items():

                if len(cases) > 10:
                    case_str = f"Test Cases {cases[0]} to {cases[-1]} ({len(cases)} total)"
                else:
                    case_str = f"Test Cases {', '.join(map(str, cases))}"

                feedback += f"\n\n{case_str}:\n{err_msg}"

            feedback += "\n\nPlease fix these errors and output the corrected code."

        return pass_rate, passed, total, feedback


def process_folder(base_folder, language, model_id, output_dir="./outputs", gpu_id=0, multi_gpu=False):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    gc.collect()
    torch.cuda.empty_cache()

    if multi_gpu:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("multi_gpu=True but fewer than 2 CUDA devices are available.")
        
        print(f"Loading Model: {model_id} across {torch.cuda.device_count()} GPUs (device_map='auto')...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_cache=True,
            attn_implementation="eager"  # Use eager attention (no flash-attn required)
        )
        device = torch.device("cuda:0")
        print(f"Model loaded successfully. Device map:")
        for name, param in model.named_parameters():
            if param.device.type == 'cuda':
                print(f"  Layer {name[:50]}: {param.device}")
                break  # Just print first layer to confirm splitting
        print(f"  ... (model sharded across GPUs)")
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    else:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        print(f"Loading Model: {model_id} on GPU {gpu_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": gpu_id},
            low_cpu_mem_usage=True,
            use_cache=True,
            attn_implementation="eager"  # Use eager attention (no flash-attn required)
        )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")
    
    for token_name in ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id']:
        token_id = getattr(tokenizer, token_name, None)
        if token_id is not None:
            if token_id >= vocab_size or token_id < 0:
                print(f"  WARNING: {token_name}={token_id} is out of range [0, {vocab_size-1}]!")
                print(f"  Setting {token_name} to eos_token_id={tokenizer.eos_token_id}")
                setattr(tokenizer, token_name, tokenizer.eos_token_id)
            else:
                print(f"  {token_name}: {token_id} âœ“")

    try:
        torch.cuda.synchronize()
        test_tensor = torch.tensor([1.0], device=device)
        _ = test_tensor * 2
        del test_tensor
        torch.cuda.empty_cache()
        print("CUDA device check: âœ“")
    except Exception as e:
        print(f"ERROR: CUDA device check failed: {e}")
        print("GPU state may be corrupted. Please restart the script.")
        sys.exit(1)

    dp_list = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    successful_dps = 0

    for dp_name in tqdm(dp_list, desc="Processing datapoints"):

        dp_path = os.path.join(base_folder, dp_name)

        q_path = os.path.join(dp_path, "question.txt")
        io_path = os.path.join(dp_path, "input_output.json")

        if not os.path.exists(q_path) or not os.path.exists(io_path):
            continue

        with open(q_path) as f:
            question_text = f.read().strip()

        with open(io_path) as f:
            io_data = json.load(f)

        question_output_dir = os.path.join(output_dir, dp_name)
        os.makedirs(question_output_dir, exist_ok=True)

        history_file = os.path.join(question_output_dir, "history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    existing_data = json.load(f)
                    existing_history = existing_data.get("history", [])
                    valid_attempts = sum(1 for attempt in existing_history 
                                        if attempt.get("code") and len(attempt.get("code", "").strip()) > 0)
                    if valid_attempts >= 5:
                        print(f"   > Skipping {dp_name} (already has {valid_attempts} valid attempts)")
                        successful_dps += 1
                        continue
            except:
                pass  # If can't read, proceed with processing

        history_log = []

        system_prompt = "You are an expert programmer. You will be given a coding problem. Output only the code, with no explanations or comments."

        user_prompt = f"QUESTION:\n{question_text}\n"

        if language == "Java":
            user_prompt += " Name your main class 'Main'."

        user_prompt += f"\nWrite a {language} program for the above problem."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(5):

            print(f"   > Attempt {attempt+1}/5 for {dp_name}")

            generated_code = run_model(messages, model, tokenizer, device)
            
            if not generated_code or len(generated_code.strip()) == 0:
                print(f"   > Failed to generate code for {dp_name} on attempt {attempt+1}. Skipping datapoint.")
                break

            pass_rate, passed, total, feedback = evaluate_solution(generated_code, language, io_data)

            print(f"     > Result: {passed}/{total} passed ({pass_rate:.1f}%)")

            result_entry = {
                "attempt": attempt + 1,
                "code": generated_code,
                "pass_rate": pass_rate,
                "passed": passed,
                "total": total,
                "feedback": feedback
            }

            history_log.append(result_entry)

            ext = ".py" if language == "Python" else ".cpp" if language == "C++" else ".java"

            with open(os.path.join(question_output_dir, f"attempt_{attempt+1}{ext}"), "w") as f:
                f.write(CodeExecutor(language).clean_code(generated_code))

            messages.append({"role": "assistant", "content": generated_code})
            messages.append({"role": "user", "content": f"USER_FEEDBACK: {feedback}"})

        if len(history_log) == 5:
            with open(os.path.join(question_output_dir, "history.json"), "w") as f:
                json.dump({"datapoint": dp_name, "language": language, "history": history_log}, f, indent=2)
            successful_dps += 1
        else:
            print(f"   > Incomplete attempts for {dp_name}. Will retry on next run.")

    print(f"\nProcessing Complete. Processed {successful_dps} datapoints.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--language", type=str, choices=["Python", "Java", "C++"], required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--multi_gpu", action="store_true", help="Shard the model across all available GPUs using device_map='auto'.")

    args = parser.parse_args()

    process_folder(
        args.folder_name,
        args.language,
        args.model,
        args.output_dir,
        args.gpu_id,
        args.multi_gpu
    )
