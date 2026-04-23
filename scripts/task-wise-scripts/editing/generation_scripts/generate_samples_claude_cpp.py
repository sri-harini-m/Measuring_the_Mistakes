import json
import os
import re
from anthropic import Anthropic
from tqdm import tqdm
from data import stream_jsonl, write_jsonl

MODEL_NAME = "claude-sonnet-4-5-20250929"
PROBLEM_FILE = "edit_eval_cpp_temp.jsonl"
OUTPUT_FILE = "claude-sonnet_cpp_samples.jsonl"
MAX_TOKENS = 2048

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

def strip_code_block(code: str) -> str:
    """
    Strips the markdown code block formatting from a string.
    Handles cases with and without language identifiers.
    """
    if not isinstance(code, str) or not code:
        return code
    
    m = re.search(r"```(?:[a-zA-Z0-9_+\-.]+)?\s*\n(.*?)\n```", code, re.DOTALL)
    if m:
        return m.group(1).strip()
    return code.strip()

def generate_one_sample(prompt: str):
    """Generates a single code completion from Claude."""
    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        response = message.content[0].text
        
        return strip_code_block(response)
    except Exception as e:
        print(f"Error during code generation: {e}")
        return ""

def main():
    """
    Main function to generate C++ samples for the evaluation.
    """
    print(f"Starting C++ sample generation for model: {MODEL_NAME}")
    print(f"Loading problems from: {PROBLEM_FILE}")

    try:
        problems = list(stream_jsonl(PROBLEM_FILE))
    except FileNotFoundError:
        print(f"Error: Problem file not found at {PROBLEM_FILE}")
        print("Please make sure you have run the translation script to create the C++ dataset.")
        return
        
    samples = []

    for task in tqdm(problems, desc="Generating C++ samples"):
        instruction = task["instruction"]
        input_code = task["input"]

        prompt = f"""You are an expert C++ programmer. Your task is to edit the C++ code based on the provided instruction.
Do not add any explanations, comments, or extra text. Only output the raw, complete, and edited C++ code.

**Instruction:**
{instruction}

**Input Code:**
```cpp
{input_code}
```

**Your C++ Code:**
"""
        
        completion = generate_one_sample(prompt)
        
        sample = {
            "task_id": task["task_id"],
            "completion": completion
        }
        samples.append(sample)

    write_jsonl(OUTPUT_FILE, samples)
    print(f"\nSuccessfully generated {len(samples)} C++ samples.")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
