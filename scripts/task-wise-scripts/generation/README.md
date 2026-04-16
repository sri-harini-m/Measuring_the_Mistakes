# Generation Tasks Scripts Quick Guide

1. Generation task scripts are:
   - `code_gen.py` for per-datapoint code generation.
   - `create_chi_csv.py` for combining per-model CHI CSV files into one file.

2. To run generation with `code_gen.py`, use:
   - `python code_gen.py --folder_name <APPS_FOLDER> --language Python --output_dir ./outputs --model microsoft/phi-4 --gpu_id 1`
   - `python code_gen.py --folder_name <APPS_FOLDER> --language Java --output_dir ./outputs --model microsoft/phi-4 --gpu_id 1`
   - `python code_gen.py --folder_name <APPS_FOLDER> --language "C++" --output_dir ./outputs --model microsoft/phi-4 --gpu_id 1`

3. Input and output behavior for `code_gen.py`:
   - It reads datapoint folders from `--folder_name`.
   - Each datapoint should contain `question.txt` (and optionally `metadata.json`).
   - It writes one JSON file per datapoint into `--output_dir` as `<datapoint>_generated.json`.
   - If an output JSON already exists for a datapoint, that datapoint is skipped.

4. To combine CHI CSV files after evaluation, run:
   - `python create_chi_csv.py`

5. CSV combine behavior for `create_chi_csv.py`:
   - It reads CSVs from `chi_csv/` in the same folder.
   - For each model, if both `<model>_results_chi.csv` and `<model>_results_clean_chi.csv` exist, the clean file is used.
   - It writes the final combined CSV to `chi_csv/all_models_chi.csv`.
