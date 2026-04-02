import json
import os
import re
import google.generativeai as genai
from tqdm import tqdm
from data import stream_jsonl, write_jsonl

MODEL_NAME = "gemini-2.5-pro"
PROBLEM_FILE = "edit_eval_temp.jsonl"
OUTPUT_FILE = "gemini-pro_python_samples_temp.jsonl"
MAX_TOKENS = 8192

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)

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
    """Generates a single code completion from Gemini."""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
            ),
        )
        
        if not response.candidates:
            print(f"Warning: No candidates returned. Response: {response}")
            return ""
        
        candidate = response.candidates[0]
        if not candidate.content.parts:
            print(f"Warning: No parts in response. Finish reason: {candidate.finish_reason}")
            return ""
        
        response_text = response.text
        
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
