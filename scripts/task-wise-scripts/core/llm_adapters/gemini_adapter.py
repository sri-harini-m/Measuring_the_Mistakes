import time
from typing import Optional

from core.llm_adapters.base import BaseLLMAdapter


class GeminiAdapter(BaseLLMAdapter):
    def __init__(self, model: str, timeout: int = 300, rate_limit_delay: float = 0.0):
        super().__init__(model, timeout)
        self.rate_limit_delay: float = rate_limit_delay
        self.last_request_time: float = 0.0
        self._client = None
    
    def _initialize_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                import os
                
                api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable not set")
                
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(self.model)
                print(f"  -> Initialized Gemini client with model: {self.model}")
            except ImportError:
                raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
    
    def _apply_rate_limit(self) -> None:
        if self.rate_limit_delay > 0:
            current_time: float = time.time()
            time_since_last_request: float = current_time - self.last_request_time
            
            if time_since_last_request < self.rate_limit_delay:
                sleep_time: float = self.rate_limit_delay - time_since_last_request
                print(f"  -> Rate limiting: waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def generate(self, prompt: str) -> Optional[str]:
        try:
            self._initialize_client()
            self._apply_rate_limit()
            
            print(f"  -> Sending code to Gemini model ({self.model}) for refactoring...", flush=True)
            
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            response = self._client.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.0,
                    'max_output_tokens': 65535,
                },
                request_options={'timeout': self.timeout}
            )
            
            if not response or not response.text:
                print("  -> Warning: Empty response from Gemini", flush=True)
                return None
            
            print("  -> Gemini response received.", flush=True)
            print("\n" + "="*80, flush=True)
            print("MODEL OUTPUT:", flush=True)
            print("="*80, flush=True)
            print(response.text, flush=True)
            print("="*80 + "\n", flush=True)
            return response.text
            
        except Exception as e:
            print(f"  -> Warning: Gemini failed: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

