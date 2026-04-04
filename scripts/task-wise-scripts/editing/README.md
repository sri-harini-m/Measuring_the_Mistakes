# Editing Scripts Quick Guide

1. Generation scripts all start with `generate`. For closed-source models, the parent organization appears in the filename (for example: `generate_samples_openai_python.py`, `generate_samples_claude_java.py`, `generate_samples_gemini_cpp.py`).

2. To run evaluation for any model/language, use:
   - `evaluate_functional_correctness_cpp.py`
   - `evaluate_functional_correctness_java.py`
   - `evaluate_functional_correctness_python.py`

3. To obtain CHI values:
   - First run `convert_to_chi_format.py` to generate `model_language_chi.csv` for each of the three languages.
   - Combine the three language CSV files by running `combine_chi_csvs.py` to obtain `model_chi.csv`.
   - Manually combine the evaluation outputs from step 2 into one file named `model_results.jsonl` inside a folder named `output` (order: python, cpp, java).
   - Run `chi_updated_script.py`.
