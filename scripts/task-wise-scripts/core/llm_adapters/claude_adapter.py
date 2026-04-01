import time
from typing import Optional

from core.llm_adapters.base import BaseLLMAdapter


class ClaudeAdapter(BaseLLMAdapter):
    def __init__(self, model: str, timeout: int = 300, rate_limit_delay: float = 0.0):
        super().__init__(model, timeout)
        self.rate_limit_delay: float = rate_limit_delay
        self.last_request_time: float = 0.0
        self._client = None
    
    def _initialize_client(self):
        if self._client is None:
            try:
                import anthropic
                import os
                
                api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                
                self._client = anthropic.Anthropic(api_key=api_key)
                print(f"  -> Initialized Claude client with model: {self.model}")
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
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
            
            print(f"  -> Sending code to Claude model ({self.model}) for refactoring...", flush=True)
            
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            message = self._client.messages.create(
                model=self.model,
                max_tokens=64000,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=self.timeout
            )
            
            if not message or not message.content:
                print("  -> Warning: Empty response from Claude", flush=True)
                return None
            
            response_text = message.content[0].text
            
            print("  -> Claude response received.", flush=True)
            print("\n" + "="*80, flush=True)
            print("MODEL OUTPUT:", flush=True)
            print("="*80, flush=True)
            print(response_text, flush=True)
            print("="*80 + "\n", flush=True)
            return response_text
            
        except Exception as e:
            print(f"  -> Warning: Claude failed: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
