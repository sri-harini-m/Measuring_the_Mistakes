from abc import ABC, abstractmethod
from typing import Dict, Optional

from core.models import KataData


class BaseDatasetLoader(ABC):
    def __init__(self, input_dir: str, language_filter: Optional[str] = None):
        self.input_dir: str = input_dir
        self.language_filter: Optional[str] = language_filter
    
    @abstractmethod
    def load_data(self) -> Dict[str, KataData]:
        pass
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        pass

