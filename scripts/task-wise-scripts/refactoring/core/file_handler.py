import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.models import KataData, RefactoringResult
from core.dataset_loaders.base import BaseDatasetLoader
from core.dataset_loaders.kata_loader import KataLoader
from core.dataset_loaders.codeeditorbench_loader import CodeEditorBenchLoader


def create_dataset_loader(
    dataset_type: str,
    input_dir: str,
    language_filter: Optional[str] = None
) -> BaseDatasetLoader:
    dataset_type_lower: str = dataset_type.lower()
    
    if dataset_type_lower == "katas":
        return KataLoader(input_dir, language_filter)
    elif dataset_type_lower == "codeeditorbench":
        return CodeEditorBenchLoader(input_dir, language_filter)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: katas, codeeditorbench")


def load_dataset(
    dataset_type: str,
    input_dir: str,
    language_filter: Optional[str] = None
) -> Dict[str, KataData]:
    loader: BaseDatasetLoader = create_dataset_loader(dataset_type, input_dir, language_filter)
    return loader.load_data()


def write_refactored_files(
    kata_name: str,
    language: str,
    refactored_files: Dict[str, str],
    output_dir: str
) -> List[str]:
    output_path: Path = Path(output_dir) / kata_name / language
    output_path.mkdir(parents=True, exist_ok=True)
    
    written_files: List[str] = []
    for filename, code in refactored_files.items():
        file_path: Path = output_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        written_files.append(str(file_path))
        print(f"  -> Written: {file_path}")
    
    return written_files


def append_to_jsonl(
    output_file: str,
    result: RefactoringResult,
    refactored_code: str
) -> None:
    output_path: Path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_entry: Dict[str, Any] = {
        'idx': result['kata_name'].replace('problem_', ''),
        'code_language': result['language'],
        'refactored_solution': refactored_code,
        'refactoring_time_seconds': result['refactoring_time_seconds'],
        'num_files': result['num_files'],
        'num_new_files': result['num_new_files'],
        'avg_codebleu': result['avg_codebleu'],
        'file_metrics': result['file_metrics'],
        'improvement': result['improvement'],
        'test_results': result.get('test_results')
    }
    
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(output_entry) + '\n')
    
    print(f"  -> Appended to: {output_path}")

