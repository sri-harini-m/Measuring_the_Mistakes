#!/usr/bin/env python3
"""Combine per-model CHI CSVs into one file.

Rules:
1) Read input CSVs from chi_csv/.
2) If both <model>_results_chi.csv and <model>_results_clean_chi.csv exist,
   use the clean file for that model.
3) Add a model_name column to each output row.
"""

import csv
import re
import sys
from pathlib import Path


REGULAR_RE = re.compile(r"^(?P<model>.+?)_results_chi\.csv$")
CLEAN_RE = re.compile(r"^(?P<model>.+?)_results_clean_chi\.csv$")


def configure_csv_field_limit():
    """Raise csv field size limit to support very large code cells."""
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 10


def discover_model_files(chi_dir: Path):
    """Return mapping of model -> chosen CSV path, preferring clean files."""
    candidates = {}

    for path in chi_dir.glob("*_chi.csv"):
        name = path.name

        if name == "all_models_chi.csv":
            continue

        clean_match = CLEAN_RE.match(name)
        if clean_match:
            model = clean_match.group("model")
            entry = candidates.setdefault(model, {})
            entry["clean"] = path
            continue

        regular_match = REGULAR_RE.match(name)
        if regular_match:
            model = regular_match.group("model")
            entry = candidates.setdefault(model, {})
            entry["regular"] = path

    selected = {}
    for model, files in candidates.items():
        if "clean" in files:
            selected[model] = files["clean"]
        elif "regular" in files:
            selected[model] = files["regular"]

    return selected


def combine_chi_csvs(chi_dir: Path, output_path: Path):
    configure_csv_field_limit()
    model_to_path = discover_model_files(chi_dir)

    if not model_to_path:
        raise RuntimeError(f"No model CSV files found in {chi_dir}")

    all_rows = []
    input_fieldnames = None

    for model_name in sorted(model_to_path):
        source_path = model_to_path[model_name]
        print(f"Using {source_path.name} for model {model_name}")

        with open(source_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"Skipping empty file: {source_path}")
                continue

            if input_fieldnames is None:
                input_fieldnames = list(reader.fieldnames)

            for row in reader:
                output_row = {"model_name": model_name}
                for col in reader.fieldnames:
                    output_row[col] = row.get(col, "")
                all_rows.append(output_row)

    if not all_rows:
        raise RuntimeError("No rows were loaded from selected model CSV files")

    if input_fieldnames is None:
        raise RuntimeError("Failed to detect input CSV columns")

    output_fields = ["model_name"] + input_fieldnames

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCreated combined CSV: {output_path}")
    print(f"Total models: {len(model_to_path)}")
    print(f"Total rows: {len(all_rows)}")


def main():
    base_dir = Path(__file__).parent
    chi_dir = base_dir / "chi_csv"
    output_path = chi_dir / "all_models_chi.csv"

    combine_chi_csvs(chi_dir, output_path)


if __name__ == "__main__":
    main()