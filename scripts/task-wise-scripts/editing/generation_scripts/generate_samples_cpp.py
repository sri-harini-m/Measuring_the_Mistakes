import json
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from data import stream_jsonl, write_jsonl

# --- Configuration ---
MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
PROBLEM_FILE = "edit_eval_cpp_temp.jsonl"
OUTPUT_FILE = "deepseek_cpp_samples.jsonl"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate C++ code-edit samples using a Hugging Face causal LM."
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"Hugging Face model name or path (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--output-file",
        default=OUTPUT_FILE,
        help=f"Path to output JSONL file (default: {OUTPUT_FILE})",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def strip_code_block(code: str) -> str:
    """
    Strips the markdown code block formatting from a string.
    Handles cases with and without language identifiers.
    """
    if not isinstance(code, str) or not code:
        return code
    
    # Regex to find code within ```...```, handling optional language specifier
    m = re.search(r"```(?:[a-zA-Z0-9_+\-.]+)?\s*\n(.*?)\n```", code, re.DOTALL)
    if m:
        return m.group(1).strip()
    return code.strip()

def generate_one_sample(prompt: str, model, tokenizer):
    """Generates a single code completion from the model."""
    try:
        # Format the prompt as a conversation for the instruct model
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Prepare model inputs
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract only the new tokens (response)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Decode the generated text
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Strip markdown and return the raw code
        return strip_code_block(response)
    except Exception as e:
        print(f"Error during code generation: {e}")
        return ""

def main():
    """
    Main function to generate C++ samples for the evaluation.
    """
    args = parse_args()
    model_name = args.model_name
    output_file = args.output_file

    model, tokenizer = load_model_and_tokenizer(model_name)

    print(f"Starting C++ sample generation for model: {model_name}")
    print(f"Loading problems from: {PROBLEM_FILE}")

    # Load problems and generate samples
    try:
        problems = list(stream_jsonl(PROBLEM_FILE))
    except FileNotFoundError:
        print(f"Error: Problem file not found at {PROBLEM_FILE}")
        print("Please make sure you have run the translation script to create the C++ dataset.")
        return
        
    samples = []

    for task in tqdm(problems, desc="Generating C++ samples"):
        # The 'input' field contains the code to be edited
        # and the 'instruction' field contains the edit instruction.
        instruction = task["instruction"]
        input_code = task["input"]

        # Construct the prompt for a C++ code editing task
        prompt = f""" Your task is to edit the C++ code based on the provided instruction.
Do not add any explanations, comments, or extra text. Only output the edited C++ code without any explanations.

**Instruction:**
{instruction}

**Input Code:**
```cpp
{input_code}
```

**Your C++ Code:**
"""
        
        completion = generate_one_sample(prompt, model, tokenizer)
        
        sample = {
            "task_id": task["task_id"],
            "completion": completion
        }
        samples.append(sample)

    # Write the generated samples to the output file
    write_jsonl(output_file, samples)
    print(f"\nSuccessfully generated {len(samples)} C++ samples.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
