import json
import csv
from pathlib import Path
import sys

def load_dataset(dataset_path):
    dataset = {}
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            dataset[entry['idx']] = entry
    return dataset


def load_model_results(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {entry['id']: entry for entry in data.get('results', [])}

def create_test_cases_json(dataset_entry):
    test_cases = []

    if dataset_entry.get('public_tests_input') or dataset_entry.get('public_tests_output'):
        test_cases.append({
            'input': str(dataset_entry.get('public_tests_input', '')).strip(),
            'output': str(dataset_entry.get('public_tests_output', '')).strip()
        })

    for inp, out in zip(
        dataset_entry.get('private_tests_input', []),
        dataset_entry.get('private_tests_output', [])
    ):
        test_cases.append({
            'input': str(inp).strip(),
            'output': str(out).strip()
        })

    return json.dumps(test_cases)

def process_language(model_name, model_dir, dataset_dir, language):
    rows = []

    if language == 'cpp':
        results_file = model_dir / language / f"{model_name}_{language}_results_sampled.json"
        dataset_file = dataset_dir / "verified_cpp_dataset_sampled.jsonl"
    elif language == 'java':
        results_file = model_dir / language / f"{model_name}_{language}_results.json"
        dataset_file = dataset_dir / "verified_java_dataset.jsonl"
    elif language == 'python3':
        results_file = model_dir / language / f"{model_name}_{language}_results.json"
        dataset_file = dataset_dir / "verified_python3_dataset.jsonl"
    else:
        return []

    if not results_file.exists() or not dataset_file.exists():
        return []

    dataset = load_dataset(dataset_file)
    results = load_model_results(results_file)

    for idx in dataset:
        if idx not in results:
            continue

        dataset_entry = dataset[idx]
        result_entry = results[idx]

        code = result_entry.get('model_response', '')
        test_cases_json = create_test_cases_json(dataset_entry)

        rows.append({
            'model_name': model_name,
            'language': language,
            'code': code,
            'test_cases': test_cases_json
        })

    return rows

def create_combined_csv(base_dir, output_path):
    dataset_dir = base_dir / "processed_dataset"

    all_rows = []

    for parent in ["close_source", "open_source"]:
        parent_dir = base_dir / parent
        if not parent_dir.exists():
            continue

        for model_dir in parent_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            print(f"Processing model: {model_name}")

            for lang in ['cpp', 'java', 'python3']:
                rows = process_language(model_name, model_dir, dataset_dir, lang)
                all_rows.extend(rows)

    if not all_rows:
        print("No data found!")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['model_name', 'language', 'code', 'test_cases']
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCombined CSV created: {output_path}")
    print(f"Total rows: {len(all_rows)}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    output_dir = base_dir / "chi_csv"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "all_models_chi.csv"

    create_combined_csv(base_dir, output_path)
