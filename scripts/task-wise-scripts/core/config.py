import os
from pathlib import Path
from typing import Final, Set

try:
    from dotenv import load_dotenv

    env_path: Path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

COMPLEXITY_THRESHOLD: Final[int] = 10
MAX_FUNCTION_LINES: Final[int] = 50
MAX_PARAMETERS: Final[int] = 5

CODE_EXTENSIONS: Final[Set[str]] = {'.py', '.java', '.cpp', '.c', '.cc', '.cxx', '.h', '.hpp', '.hxx'}
INSTRUCTIONS_FILE: Final[str] = "instructions.txt"

LANGUAGE_EXTENSIONS: Final[dict[str, str]] = {'python': 'py', 'java': 'java', 'cpp': 'cpp'}

EXTENSION_TO_LANGUAGE: Final[dict[str, str]] = {'.py': 'python', '.java': 'java', '.cpp': 'cpp', '.cc': 'cpp',
    '.cxx': 'cpp', '.c': 'cpp', '.h': 'cpp', '.hpp': 'cpp', '.hxx': 'cpp'}

SUPPORTED_LANGUAGES: Final[Set[str]] = {'python', 'java', 'cpp'}

SUPPORTED_DATASETS: Final[Set[str]] = {'katas', 'codeeditorbench'}
DEFAULT_DATASET: Final[str] = 'katas'

DATASET_INPUT_PATHS: Final[dict[str, str]] = {'katas': 'input/katas', 'codeeditorbench': 'input/codeeditorbench'}

DATASET_OUTPUT_PATHS: Final[dict[str, str]] = {'katas': 'output/katas', 'codeeditorbench': 'output/codeeditorbench'}

DATASET_RESULTS_PATHS: Final[dict[str, str]] = {'katas': 'results/katas', 'codeeditorbench': 'results/codeeditorbench'}

REFACTORING_PROMPT_TEMPLATE: Final[str] = """You are an expert code refactoring assistant. Your task is to refactor the following {language} code to improve its quality, extensibility, maintainability, and readability.

{code_section}

Please refactor the code following these principles:
1. Improve code quality, extensibility, maintainability, and readability
2. Remove all code smells
3. Ensure that the refactored code maintains the same functionality
4. Add meaningful variable/function names if needed
5. Remove code duplication
6. Do NOT modify the input/output format - the code must read input and produce output exactly as the original
7. Do NOT create new files - put all refactored code in the SAME files they originally came from
8. Do NOT rename files - use the EXACT same filenames as the input

Output ONLY the refactored code. Do NOT include any explanations, summaries, descriptions of changes, or commentary before or after the code.

Format your response EXACTLY as:
FILENAME: <filename>
```{language}
<refactored code>
```

Make sure to include all files, even if some don't require changes.{single_file_constraint}"""

SINGLE_FILE_CONSTRAINT: Final[str] = """

IMPORTANT: The input contains a single file. You must output exactly ONE file with ONLY the refactored code. Do not split the code into multiple files."""

ADDITIONAL_INSTRUCTIONS_TEMPLATE: Final[str] = """Additional code-specific refactoring instructions/hints:
{instructions}

"""

DEFAULT_PROVIDER: Final[str] = "huggingface"
DEFAULT_MODEL: Final[str] = "qwen3-coder_16k"
DEFAULT_TIMEOUT: Final[int] = 300
DEFAULT_RATE_LIMIT_DELAY: Final[float] = 0.0

PROVIDER_MODELS: Final[dict[str, dict[str, str]]] = {
    'gemini': {'default': 'gemini-2.5-pro', 'examples': 'gemini-2.5-pro, gemini-2.5-flash'},
    'huggingface': {'default': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'examples': 'Qwen/Qwen2.5-Coder-7B-Instruct, codellama/CodeLlama-7b-Instruct-hf'},
    'claude': {'default': 'claude-sonnet-4-5', 'examples': 'claude-sonnet-4-5'},
    'openai': {'default': 'gpt-5', 'examples': 'gpt-5-mini'}}
