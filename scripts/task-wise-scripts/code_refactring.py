import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

from core.config import (DEFAULT_PROVIDER, DEFAULT_TIMEOUT, DEFAULT_RATE_LIMIT_DELAY, DEFAULT_DATASET,
                         SUPPORTED_LANGUAGES, SUPPORTED_DATASETS, PROVIDER_MODELS, DATASET_INPUT_PATHS,
                         DATASET_OUTPUT_PATHS)
from core.file_handler import load_dataset, write_refactored_files, append_to_jsonl
from core.llm_adapters.base import BaseLLMAdapter
from core.llm_client import create_llm_adapter, get_llm_refactored_code
from core.metrics import calculate_code_metrics, calculate_codebleu_score, calculate_improvement
from core.models import KataData, RefactoringResult, FileMetrics, ImprovementMetrics, TestResults
from core.reporter import generate_summary_report
from core.test_runner import run_tests


def process_kata(kata_key: str, kata_data: KataData, adapter: BaseLLMAdapter, output_dir: str, dataset_type: str) -> tuple[RefactoringResult, Dict[str, str]]:
    print(f"\n  Files: {', '.join(kata_data['code_files'].keys())}")
    print(f"  Language: {kata_data['language']}")

    if kata_data['instructions']:
        print(f"  Instructions found: {len(kata_data['instructions'])} characters")

    start_time: float = time.time()

    refactored_files: Dict[str, str] = get_llm_refactored_code(kata_data['code_files'], kata_data['instructions'],
        kata_data['language'], adapter)

    if not refactored_files:
        raise ValueError("No valid code blocks found in LLM response")

    refactoring_time: float = time.time() - start_time
    
    is_single_file_input: bool = len(kata_data['code_files']) == 1
    
    if is_single_file_input and len(refactored_files) > 1:
        print(f"  -> Warning: Expected 1 output file but got {len(refactored_files)}. Using first file only.")
        first_filename: str = list(refactored_files.keys())[0]
        refactored_files = {first_filename: refactored_files[first_filename]}

    output_paths: List[str] = []
    if dataset_type == 'katas':
        output_paths = write_refactored_files(kata_data['kata_name'], kata_data['language'], refactored_files,
            output_dir)

    file_metrics: List[FileMetrics] = []
    total_codebleu: float = 0.0
    num_new_files: int = 0

    for filename, refactored_code in refactored_files.items():
        if not refactored_code:
            continue

        is_new_file: bool = filename not in kata_data['code_files']
        
        if is_single_file_input and is_new_file:
            original_filename: str = list(kata_data['code_files'].keys())[0]
            original_code: str = kata_data['code_files'][original_filename]
            
            original_metrics = calculate_code_metrics(original_code, kata_data['language'])
            refactored_metrics = calculate_code_metrics(refactored_code, kata_data['language'])
            codebleu: float = calculate_codebleu_score(original_code, refactored_code, kata_data['language'])
            total_codebleu += codebleu
            
            file_metric: FileMetrics = FileMetrics(
                filename=filename,
                original_metrics=original_metrics,
                refactored_metrics=refactored_metrics,
                codebleu=codebleu,
                is_new_file=False
            )
            file_metrics.append(file_metric)
            
            print(f"  -> {filename} (renamed from {original_filename}):")
            print(f"     Cyclomatic Complexity: {original_metrics.get('cyclomatic_complexity_total', 'N/A')} -> {refactored_metrics.get('cyclomatic_complexity_total', 'N/A')}")
            print(f"     LOC: {original_metrics.get('loc', 'N/A')} -> {refactored_metrics.get('loc', 'N/A')}")
            print(f"     CodeBLEU: {codebleu:.4f}")
        elif is_new_file:
            num_new_files += 1
            refactored_metrics = calculate_code_metrics(refactored_code, kata_data['language'])
            
            file_metric: FileMetrics = FileMetrics(
                filename=filename,
                original_metrics=None,
                refactored_metrics=refactored_metrics,
                codebleu=None,
                is_new_file=True
            )
            file_metrics.append(file_metric)
            
            print(f"  -> {filename} (NEW FILE):")
            print(f"     Cyclomatic Complexity: {refactored_metrics.get('cyclomatic_complexity_total', 'N/A')}")
            print(f"     LOC: {refactored_metrics.get('loc', 'N/A')}")
        else:
            original_code: str = kata_data['code_files'][filename]
            original_metrics = calculate_code_metrics(original_code, kata_data['language'])
            refactored_metrics = calculate_code_metrics(refactored_code, kata_data['language'])
            codebleu: float = calculate_codebleu_score(original_code, refactored_code, kata_data['language'])
            total_codebleu += codebleu

            file_metric: FileMetrics = FileMetrics(
                filename=filename,
                original_metrics=original_metrics,
                refactored_metrics=refactored_metrics,
                codebleu=codebleu,
                is_new_file=False
            )
            file_metrics.append(file_metric)

            print(f"  -> {filename}:")
            print(f"     Cyclomatic Complexity: {original_metrics.get('cyclomatic_complexity_total', 'N/A')} -> {refactored_metrics.get('cyclomatic_complexity_total', 'N/A')}")
            print(f"     LOC: {original_metrics.get('loc', 'N/A')} -> {refactored_metrics.get('loc', 'N/A')}")
            print(f"     CodeBLEU: {codebleu:.4f}")

    modified_file_metrics: List[FileMetrics] = [fm for fm in file_metrics if not fm['is_new_file']]
    
    improvement: ImprovementMetrics = ImprovementMetrics(
        complexity_reduction=calculate_improvement(modified_file_metrics, 'cyclomatic_complexity_total'),
        loc_reduction=calculate_improvement(modified_file_metrics, 'loc'))
    
    test_results: Optional[TestResults] = None
    if kata_data.get('public_tests') or kata_data.get('private_tests'):
        print("  -> Running test cases...")
        test_results = run_test_cases(kata_data, refactored_files)
        if test_results:
            print(f"  -> Public tests: {test_results.get('public_passed', 0)}/{test_results.get('public_total', 0)} passed")
            print(f"  -> Private tests: {test_results.get('private_passed', 0)}/{test_results.get('private_total', 0)} passed")
            print(f"  -> Overall pass rate: {test_results.get('total_pass_rate', 0):.2%}")

    num_codebleu_files: int = len([fm for fm in file_metrics if fm.get('codebleu') is not None])
    avg_codebleu: float = total_codebleu / num_codebleu_files if num_codebleu_files > 0 else 0.0

    result: RefactoringResult = RefactoringResult(
        kata_key=kata_key,
        kata_name=kata_data['kata_name'],
        language=kata_data['language'],
        num_files=len(refactored_files),
        num_new_files=num_new_files,
        refactoring_time_seconds=refactoring_time,
        avg_codebleu=avg_codebleu,
        file_metrics=file_metrics,
        improvement=improvement,
        output_paths=output_paths,
        test_results=test_results
    )

    return result, refactored_files


def run_test_cases(kata_data: KataData, refactored_files: Dict[str, str]) -> Optional[TestResults]:
    if not refactored_files:
        return None
    
    code: str = list(refactored_files.values())[0]
    language: str = kata_data['language']
    
    public_tests = kata_data.get('public_tests', [])
    private_tests = kata_data.get('private_tests', [])
    
    if not public_tests and not private_tests:
        return None
    
    public_results = run_tests(code, language, public_tests) if public_tests else {'passed': 0, 'total': 0, 'pass_rate': 0.0, 'execution_time': 0.0}
    private_results = run_tests(code, language, private_tests) if private_tests else {'passed': 0, 'total': 0, 'pass_rate': 0.0, 'execution_time': 0.0}
    
    total_passed: int = public_results['passed'] + private_results['passed']
    total_tests: int = public_results['total'] + private_results['total']
    
    return TestResults(
        public_passed=public_results['passed'],
        public_total=public_results['total'],
        public_pass_rate=public_results['pass_rate'],
        private_passed=private_results['passed'],
        private_total=private_results['total'],
        private_pass_rate=private_results['pass_rate'],
        total_passed=total_passed,
        total_tests=total_tests,
        total_pass_rate=total_passed / total_tests if total_tests > 0 else 0.0,
        execution_time=public_results['execution_time'] + private_results['execution_time']
    )


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Code Refactoring Analysis with LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=f"""
Provider Examples:
  Gemini:      {PROVIDER_MODELS['gemini']['examples']}
  HuggingFace: {PROVIDER_MODELS['huggingface']['examples']}
  Claude:      {PROVIDER_MODELS['claude']['examples']}
  OpenAI:      {PROVIDER_MODELS['openai']['examples']}
Environment Variables:
  GEMINI_API_KEY:      Required for Gemini provider
  ANTHROPIC_API_KEY:   Required for Claude provider
  OPENAI_API_KEY:      Required for OpenAI provider
  HUGGINGFACE_API_KEY: Optional for HuggingFace API mode
""")
    parser.add_argument('--dataset', '-d', default=DEFAULT_DATASET, choices=list(SUPPORTED_DATASETS),
        help=f'Dataset type (default: {DEFAULT_DATASET})')
    parser.add_argument('--input', '-i', help='Input directory (defaults based on dataset type if not specified)')
    parser.add_argument('--output', '-o', help='Output directory (defaults based on dataset type if not specified)')
    parser.add_argument('--provider', '-p', default=DEFAULT_PROVIDER, choices=['gemini', 'huggingface', 'claude', 'openai'],
        help=f'LLM provider (default: {DEFAULT_PROVIDER})')
    parser.add_argument('--model', '-m', help='Model name (provider-specific, see examples below)')
    parser.add_argument('--language', '-l', choices=list(SUPPORTED_LANGUAGES),
        help='Filter by language (python, java, or cpp)')
    parser.add_argument('--timeout', '-t', type=int, default=DEFAULT_TIMEOUT,
        help=f'Request timeout in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--rate-limit', type=float, default=DEFAULT_RATE_LIMIT_DELAY,
        help='Rate limit delay in seconds between requests (for Gemini/Claude, default: 0)')
    parser.add_argument('--use-api', action='store_true',
        help='Use HuggingFace API instead of local model (HuggingFace only)')
    parser.add_argument('--gpu', type=str, default='0',
        help='GPU device(s) to use (e.g., "0" for single GPU, "0,1,2" for multiple GPUs, default: "0")')
    parser.add_argument('--results', '-r', default='refactoring_results.json',
        help='Path for detailed results JSON file')
    parser.add_argument('--summary', '-s', default='refactoring_summary.json',
        help='Path for summary results JSON file')
    parser.add_argument('--output-jsonl', help='Output JSONL file for codeeditorbench (defaults to output dir if not specified)')

    args: argparse.Namespace = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_name: str = args.model if args.model else PROVIDER_MODELS[args.provider]['default']

    input_dir: str = args.input if args.input else DATASET_INPUT_PATHS[args.dataset]
    output_dir: str = args.output if args.output else DATASET_OUTPUT_PATHS[args.dataset]
    
    output_jsonl_file: Optional[str] = None
    if args.dataset == 'codeeditorbench':
        if args.output_jsonl:
            output_jsonl_file = args.output_jsonl
        else:
            language_suffix: str = f"_{args.language}" if args.language else ""
            output_jsonl_file = f"{output_dir}/refactored_solutions{language_suffix}.jsonl"

    print("=== Code Refactoring Analysis ===")
    print(f"Dataset: {args.dataset}")
    print(f"Provider: {args.provider}")
    print(f"Model: {model_name}")
    print(f"GPU Device(s): {args.gpu}")
    print(f"Input Directory: {input_dir}")
    if args.dataset == 'katas':
        print(f"Output Directory: {output_dir}")
    else:
        print(f"Output JSONL: {output_jsonl_file}")
    if args.language:
        print(f"Language Filter: {args.language}")
    if args.provider in ['gemini', 'claude', 'openai'] and args.rate_limit > 0:
        print(f"Rate Limit: {args.rate_limit}s between requests")
    if args.provider == 'huggingface' and args.use_api:
        print(f"Using HuggingFace API")
    print()

    try:
        adapter: BaseLLMAdapter = create_llm_adapter(provider=args.provider, model=model_name, timeout=args.timeout,
            rate_limit_delay=args.rate_limit, use_api=args.use_api)
    except Exception as e:
        print(f"Error creating LLM adapter: {e}")
        return

    data_items: Dict[str, KataData] = load_dataset(args.dataset, input_dir, args.language)

    if not data_items:
        print("No data items found. Exiting.")
        return

    results: List[RefactoringResult] = []
    total_attempted: int = len(data_items)
    total_failed: int = 0
    
    if args.dataset == 'codeeditorbench' and output_jsonl_file:
        output_jsonl_path: Path = Path(output_jsonl_file)
        if output_jsonl_path.exists():
            output_jsonl_path.unlink()
            print(f"Cleared existing output file: {output_jsonl_file}\n")

    for idx, (item_key, item_data) in enumerate(data_items.items(), 1):
        print(f"\n[{idx}/{len(data_items)}] Processing: {item_key}")

        try:
            result: RefactoringResult
            refactored_files: Dict[str, str]
            result, refactored_files = process_kata(item_key, item_data, adapter, output_dir, args.dataset)
            results.append(result)
            
            if args.dataset == 'codeeditorbench' and output_jsonl_file:
                refactored_code: str = '\n\n'.join([
                    f"# File: {filename}\n{code}" 
                    for filename, code in refactored_files.items()
                ])
                append_to_jsonl(output_jsonl_file, result, refactored_code)
                
        except Exception as e:
            print(f"  -> Error processing: {e}")
            total_failed += 1
            continue

    results_dir: str = os.path.dirname(args.results)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    with open(args.results, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to {args.results}")

    summary_dir: str = os.path.dirname(args.summary)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    generate_summary_report(results, args.summary, args.dataset, total_attempted, total_failed)

    print("\n=== Refactoring Complete ===")


if __name__ == "__main__":
    main()
