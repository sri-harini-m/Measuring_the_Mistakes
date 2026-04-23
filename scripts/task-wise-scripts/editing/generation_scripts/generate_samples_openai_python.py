import json
import os
import re
from openai import OpenAI
from tqdm import tqdm
from data import stream_jsonl, write_jsonl

MODEL_NAME = "gpt-5-mini"
PROBLEM_FILE = "edit_eval.jsonl"
OUTPUT_FILE = "gpt-5-mini_python_samples.jsonl"
MAX_TOKENS = 2048

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

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
    """Generates a single code completion from OpenAI."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        if not response.choices:
            print(f"Warning: No choices returned from OpenAI")
            return ""
        
        response_text = response.choices[0].message.content
        
        return strip_code_block(response_text)
    except Exception as e:
        print(f"Error during code generation: {e}")
        return ""

def main():
    """
    Main function to generate Python samples for the evaluation.
    """
    print(f"Starting Python sample generation for model: {MODEL_NAME}")
    print(f"Loading problems from: {PROBLEM_FILE}")

    try:
        problems = list(stream_jsonl(PROBLEM_FILE))
    except FileNotFoundError:
        print(f"Error: Problem file not found at {PROBLEM_FILE}")
        print("Please make sure you have the Python dataset file.")
        return
        
    samples = []

    for task in tqdm(problems, desc="Generating Python samples"):
        instruction = task["instruction"]
        input_code = task["input"]

        prompt = f"""Your task is to edit the Python code based on the provided instruction.
Do not add any explanations, comments, or extra text. Only output the edited Python code without any explanations.

**Instruction:**
{instruction}

**Input Code:**
```python
{input_code}
```

**Your Python Code:**
"""
        completion = generate_one_sample(prompt)
        
        sample = {
            "task_id": task["task_id"],
            "completion": completion
        }
        samples.append(sample)

    write_jsonl(OUTPUT_FILE, samples)
    print(f"\nSuccessfully generated {len(samples)} Python samples.")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
