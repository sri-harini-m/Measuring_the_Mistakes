import os
import json
import argparse
from tqdm import tqdm
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN environment variable not set")
login(token=hf_token)

def generate_prompt(question_text, language, tokenizer):
    system_message = f"You are an expert {language} programmer. Write a solution for the following problem. Output only the executable code. Do not include any explanations, markdown formatting, or comments."
    prompt = "\nQUESTION:\n"
    prompt += question_text + "\n"
    prompt += f"\nWrite a {language} program for the above problem. Output only the code, with no explanations or comments.\n"

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt

def run_model(prompt, model, tokenizer, device):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=False,     
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                use_cache=True,
            )
        
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    except Exception as e:
        print(f"\nGeneration error: {e}")
        return ""

def process_folder(base_folder, language, model_id, output_dir="./outputs", gpu_id=1):
    os.makedirs(output_dir, exist_ok=True)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    num_gpus = torch.cuda.device_count()
    if gpu_id >= num_gpus:
        raise RuntimeError(f"GPU {gpu_id} not available. Available GPUs: {num_gpus}")
    
    print(f"Using device: cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    
    print(f"Loading Model: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        device_map={"": gpu_id},
        low_cpu_mem_usage=True,
    )
    
    model.eval()

    print(f"\nModel loaded on GPU {gpu_id}")
    print(f"VRAM Allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB")
    
    dp_list = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    successful = 0
    failed = 0

    for dp_name in tqdm(dp_list, desc="Processing datapoints", unit="dp"):
        try:
            dp_path = os.path.join(base_folder, dp_name)
            output_file_path = os.path.join(output_dir, f"{dp_name}_generated.json")
            
            if os.path.exists(output_file_path): 
                successful += 1
                continue
                
            q_path = os.path.join(dp_path, "question.txt")
            if not os.path.exists(q_path): 
                continue

            metadata_path = os.path.join(dp_path, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            with open(q_path, "r") as f:
                question_text = f.read().strip()

            prompt = generate_prompt(question_text, language, tokenizer)
            output_code = run_model(prompt, model=model, tokenizer=tokenizer, device=f"cuda:{gpu_id}")

            result = {
                "datapoint": dp_name,
                "language": language,
                "prompt": prompt,
                "model_output": output_code,
                "metadata": metadata,
                "model_id": model_id
            }

            with open(output_file_path, "w") as f:
                json.dump(result, f, indent=2)
            
            successful += 1
                
        except Exception as e:
            failed += 1
            print(f"\nError processing {dp_name}: {e}")
            torch.cuda.empty_cache()
            continue
    
    print(f"\n\n=== Processing Complete ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run code generation with Phi-4 in bfloat16.")
    parser.add_argument("--folder_name", type=str, required=True, help="Folder containing APPS datapoints")
    parser.add_argument("--language", type=str, choices=["Python", "Java", "C++"], required=True, help="Target language")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="microsoft/phi-4", help="Model ID")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU ID (default: 1)")
    
    args = parser.parse_args()
    
    process_folder(args.folder_name, args.language, args.model, args.output_dir, args.gpu_id)
