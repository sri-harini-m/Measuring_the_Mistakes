from pathlib import Path
from typing import Dict, Optional

from core.config import CODE_EXTENSIONS, INSTRUCTIONS_FILE, SUPPORTED_LANGUAGES
from core.models import KataData
from core.dataset_loaders.base import BaseDatasetLoader


class KataLoader(BaseDatasetLoader):
    def get_dataset_name(self) -> str:
        return "katas"
    
    def load_data(self) -> Dict[str, KataData]:
        kata_groups: Dict[str, KataData] = {}
        input_path: Path = Path(self.input_dir)
        
        if not input_path.exists():
            print(f"Error: Input directory '{self.input_dir}' does not exist.")
            return {}
        
        for kata_dir in input_path.iterdir():
            if not kata_dir.is_dir():
                continue
            
            kata_name: str = kata_dir.name
            
            for lang_dir in kata_dir.iterdir():
                if not lang_dir.is_dir():
                    continue
                
                lang_name: str = lang_dir.name.lower()
                if lang_name not in SUPPORTED_LANGUAGES:
                    continue
                
                if self.language_filter and lang_name != self.language_filter:
                    continue
                
                code_files: Dict[str, str] = {}
                instructions: Optional[str] = None
                
                for file_path in lang_dir.iterdir():
                    if not file_path.is_file():
                        continue
                    
                    if file_path.name == INSTRUCTIONS_FILE:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            instructions = f.read()
                    elif file_path.suffix.lower() in CODE_EXTENSIONS:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            code_files[file_path.name] = f.read()
                
                if code_files:
                    group_key: str = f"{kata_name}/{lang_name}"
                    kata_groups[group_key] = KataData(
                        kata_name=kata_name,
                        code_files=code_files,
                        instructions=instructions,
                        language=lang_name,
                        public_tests=None,
                        private_tests=None
                    )
        
        if self.language_filter:
            print(f"Found {len(kata_groups)} {self.language_filter} kata(s) to refactor.")
        else:
            print(f"Found {len(kata_groups)} kata(s) to refactor.")
        
        return kata_groups

