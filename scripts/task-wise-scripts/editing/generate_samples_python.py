import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from data import stream_jsonl, write_jsonl

MODEL_NAME = "microsoft/phi-4"
PROBLEM_FILE = "edit_eval_python.jsonl"
OUTPUT_FILE = "phi-4_python_samples.jsonl"

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)

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
    """Generates a single code completion from the model."""
    try:

        messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return strip_code_block(response)
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

        prompt = f""" Your task is to edit the Python code based on the provided instruction.
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
