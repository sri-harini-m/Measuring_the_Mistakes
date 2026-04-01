from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseLLMAdapter(ABC):
    def __init__(self, model: str, timeout: int = 300):
        self.model: str = model
        self.timeout: int = timeout
    
    @abstractmethod
    def generate(self, prompt: str) -> Optional[str]:
        pass
    
    def validate_response(self, response: Optional[str]) -> bool:
        return response is not None and len(response.strip()) > 0

