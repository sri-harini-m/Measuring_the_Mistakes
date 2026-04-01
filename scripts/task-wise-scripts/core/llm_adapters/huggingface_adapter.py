from typing import Optional

from core.llm_adapters.base import BaseLLMAdapter
import os


class HuggingFaceAdapter(BaseLLMAdapter):
    def __init__(self, model: str, timeout: int = 300, use_api: bool = False):
        super().__init__(model, timeout)
        self.use_api: bool = use_api
        self._pipeline = None
        self._client = None

    def _initialize_pipeline(self):
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline

                token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

                if torch.cuda.is_available():
                    print(f"  -> Loading HuggingFace model: {self.model} on GPU with bfloat16...")
                    self._pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        device_map="auto",
                        model_kwargs={"dtype": torch.bfloat16},
                        token=token,
                    )
                    print("  -> Model loaded successfully with automatic device mapping")
                else:
                    print(f"  -> Loading HuggingFace model: {self.model} on CPU with bfloat16...")
                    self._pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        device="cpu",
                        model_kwargs={"dtype": torch.bfloat16},
                        token=token,
                    )
                    print("  -> Model loaded successfully on CPU")

                if self._pipeline.tokenizer.pad_token_id is None:
                    self._pipeline.tokenizer.pad_token = self._pipeline.tokenizer.eos_token

            except ImportError:
                raise ImportError("transformers package not installed. Install with: pip install transformers torch")

    def _initialize_api_client(self):
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient

                api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
                if not api_key:
                    raise ValueError(
                        "HUGGINGFACE_API_KEY or HF_TOKEN not set. Export one of them."
                    )

                self._client = InferenceClient(token=api_key)
                print(f"  -> Initialized HuggingFace API client for model: {self.model}")
            except ImportError:
                raise ImportError("huggingface_hub package not installed. Install with: pip install huggingface_hub")

    def _generate_with_pipeline(self, prompt: str) -> Optional[str]:
        try:
            tokenizer = self._pipeline.tokenizer
            
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            eos_id = tokenizer.eos_token_id
            try:
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            except Exception:
                eot_id = None
            terminators = [eid for eid in [eos_id, eot_id] if eid is not None]

            out = self._pipeline(
                formatted_prompt,
                max_new_tokens=4096,
                eos_token_id=terminators,
                do_sample=False,
                temperature=0.0,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False,
            )

            if not out:
                return None

            if isinstance(out[0].get("generated_text"), str):
                return out[0]["generated_text"]

            if isinstance(out[0].get("generated_text"), list):
                last = out[0]["generated_text"][-1]
                if isinstance(last, dict) and "content" in last:
                    return last["content"]

            return None

        except Exception as e:
            print(f"  -> Warning: Pipeline generation failed: {e}")
            return None

    def _generate_with_api(self, prompt: str) -> Optional[str]:
        try:
            response = self._client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=4096,
                temperature=0.0,
                return_full_text=False
            )
            return response if response else None
        except Exception as e:
            print(f"  -> Warning: API generation failed: {e}")
            return None

    def generate(self, prompt: str) -> Optional[str]:
        try:
            if self.use_api:
                self._initialize_api_client()
                print(f"  -> Sending code to HuggingFace API ({self.model}) for refactoring...")
                response: Optional[str] = self._generate_with_api(prompt)
            else:
                self._initialize_pipeline()
                print(f"  -> Generating with local HuggingFace model ({self.model})...")
                response: Optional[str] = self._generate_with_pipeline(prompt)

            if response:
                print("  -> HuggingFace response received.")
                print("\n" + "="*80)
                print("MODEL OUTPUT:")
                print("="*80)
                print(response)
                print("="*80 + "\n")
            return response

        except Exception as e:
            print(f"  -> Warning: HuggingFace failed: {e}")
            return None
