import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from data import stream_jsonl, write_jsonl
import argparse

PROBLEM_FILE = "edit_eval_java.jsonl"

tokenizer = None
model = None

def strip_code_block(code: str) -> str:
    """Remove markdown code fences and return raw code."""
    if not isinstance(code, str) or not code:
        return code
    m = re.search(r"```(?:[a-zA-Z0-9_+\-.]+)?\s*\n(.*?)\n```", code, re.DOTALL)
    if m:
        return m.group(1).strip()
    return code.strip()

def generate_one_sample(prompt: str) -> str:
    """Generate a single Java completion from the model."""
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
                max_new_tokens=2048,
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
    """Generate Java samples for the evaluation set."""
    global tokenizer, model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    MODEL_NAME = args.model
    OUTPUT_FILE = args.output_file
    print(f"Starting Java sample generation for model: {MODEL_NAME}")
    print(f"Loading problems from: {PROBLEM_FILE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    try:
        problems = list(stream_jsonl(PROBLEM_FILE))
    except FileNotFoundError:
        print(f"Error: Problem file not found at {PROBLEM_FILE}")
        print("Please make sure you have the Java dataset file.")
        return

    samples = []
    for task in tqdm(problems, desc="Generating Java samples"):
        instruction = task["instruction"]
        input_code = task["input"]
        prompt = f""" Your task is to edit the Java code based on the provided instruction.
Do not add any explanations, comments, or extra text. Only output the edited Java code without any explanations.

**Instruction:**
{instruction}

**Input Code:**
```java
{input_code}
```

**Your Java Code:**
"""
        completion = generate_one_sample(prompt)
        samples.append({"task_id": task["task_id"], "completion": completion})

    write_jsonl(OUTPUT_FILE, samples)
    print(f"\nSuccessfully generated {len(samples)} Java samples.")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
