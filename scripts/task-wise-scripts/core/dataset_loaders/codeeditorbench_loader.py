import json
from pathlib import Path
from typing import Dict, Optional, Any, List

from core.config import SUPPORTED_LANGUAGES, EXTENSION_TO_LANGUAGE
from core.models import KataData, TestCase
from core.dataset_loaders.base import BaseDatasetLoader


class CodeEditorBenchLoader(BaseDatasetLoader):
    def get_dataset_name(self) -> str:
        return "codeeditorbench"
    
    def load_data(self) -> Dict[str, KataData]:
        problems: Dict[str, KataData] = {}
        input_path: Path = Path(self.input_dir)
        
        jsonl_file: Path = input_path / "problems.jsonl"
        
        if not jsonl_file.exists():
            print(f"Error: JSONL file '{jsonl_file}' does not exist.")
            return {}
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry: Dict[str, Any] = json.loads(line.strip())
                    problem_data: Optional[KataData] = self._parse_entry(entry, line_num)
                    
                    if problem_data:
                        if self.language_filter and problem_data['language'] != self.language_filter:
                            continue
                        
                        problem_key: str = f"{problem_data['kata_name']}/{problem_data['language']}"
                        problems[problem_key] = problem_data
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
        
        if self.language_filter:
            print(f"Found {len(problems)} {self.language_filter} CodeEditorBench problem(s) to refactor.")
        else:
            print(f"Found {len(problems)} CodeEditorBench problem(s) to refactor.")
        
        return problems
    
    def _parse_entry(self, entry: Dict[str, Any], line_num: int) -> Optional[KataData]:
        problem_id: str = str(entry.get('idx', f'problem_{line_num}'))
        code: str = entry.get('solutions', '')
        
        if not code:
            print(f"Warning: No solutions field for problem at line {line_num}")
            return None
        
        language_hint: Optional[str] = entry.get('code_language', None)
        
        language: Optional[str] = self._infer_language(language_hint)
        
        if not language or language not in SUPPORTED_LANGUAGES:
            print(f"Warning: Unsupported or unknown language '{language_hint}' for problem at line {line_num}")
            return None
        
        code = self._clean_code(code)
        
        filename: str = f"problem_{problem_id}.{self._get_extension(language)}"
        code_files: Dict[str, str] = {filename: code}
        
        public_tests: List[TestCase] = self._parse_test_cases(
            entry.get('public_tests_input', ''),
            entry.get('public_tests_output', '')
        )
        
        private_tests: List[TestCase] = self._parse_test_cases(
            entry.get('private_tests_input', []),
            entry.get('private_tests_output', [])
        )
        
        return KataData(
            kata_name=f"problem_{problem_id}",
            code_files=code_files,
            instructions=None,
            language=language,
            public_tests=public_tests if public_tests else None,
            private_tests=private_tests if private_tests else None
        )
    
    def _clean_code(self, code: str) -> str:
        code = code.strip()
        
        markdown_prefixes: List[str] = [
            "```python\n", "```cpp\n", "```java\n", "```c++\n",
            "```python", "```cpp", "```java", "```c++",
            "```\n", "```"
        ]
        
        for prefix in markdown_prefixes:
            if code.startswith(prefix):
                code = code[len(prefix):].lstrip()
                break
        
        if code.endswith("```"):
            code = code[:-3].rstrip()
        
        return code
    
    def _parse_test_cases(self, inputs: Any, outputs: Any) -> List[TestCase]:
        test_cases: List[TestCase] = []
        
        if isinstance(inputs, str):
            inputs = [inputs] if inputs else []
        if isinstance(outputs, str):
            outputs = [outputs] if outputs else []
        
        if not inputs or not outputs:
            return test_cases
        
        for test_input, test_output in zip(inputs, outputs):
            test_cases.append(TestCase(
                input=str(test_input),
                output=str(test_output)
            ))
        
        return test_cases
    
    def _infer_language(self, language_hint: Optional[str]) -> Optional[str]:
        if not language_hint:
            return None
        
        lang_lower: str = language_hint.lower()
        
        if lang_lower == 'python3':
            return 'python'
        elif lang_lower in SUPPORTED_LANGUAGES:
            return lang_lower
        elif lang_lower in ['c++', 'c', 'cc', 'cxx']:
            return 'cpp'
        
        return None
    
    def _get_extension(self, language: str) -> str:
        ext_map: Dict[str, str] = {
            'python': 'py',
            'java': 'java',
            'cpp': 'cpp'
        }
        return ext_map.get(language, 'txt')

