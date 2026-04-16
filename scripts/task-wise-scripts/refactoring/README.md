# Refactoring Tasks Scripts Quick Guide

1. Refactoring task entry scripts are:
   - `main.py` for running LLM-based refactoring and evaluation.
   - `create_chi_csv.py` for preparing a combined CHI input CSV from refactoring outputs.

2. To run refactoring with `main.py`, use:
   - `python main.py --dataset codeeditorbench --provider openai --model gpt-5 --language python --gpu 0`
   - `python main.py --dataset codeeditorbench --provider claude --model claude-sonnet-4-5 --language java --gpu 0`
   - `python main.py --dataset codeeditorbench --provider gemini --model gemini-2.5-pro --language cpp --gpu 0`

3. Key options for `main.py`:
   - `--dataset`: `katas` or `codeeditorbench`.
   - `--provider`: `gemini`, `huggingface`, `claude`, or `openai`.
   - `--model`: provider-specific model name.
   - `--language`: `python`, `java`, or `cpp`.
   - `--gpu`: GPU id(s), for example `0` or `0,1`.
   - `--rate-limit`: delay between requests for API-based providers.
   - `--use-api`: use Hugging Face API mode (Hugging Face provider only).

4. Required credentials by provider:
   - OpenAI requires `OPENAI_API_KEY`.
   - Claude requires `ANTHROPIC_API_KEY`.
   - Gemini requires `GEMINI_API_KEY`.
   - Hugging Face API mode can use `HUGGINGFACE_API_KEY`.

5. To create CHI input CSV after refactoring, run:
   - `python create_chi_csv.py`

6. CSV combine behavior for `create_chi_csv.py`:
   - It reads refactored outputs from `output/codeeditorbench/*/refactored_solutions_*.jsonl`.
   - It reads test cases from `refactoring/input/codeeditorbench/problems.jsonl`.
   - It writes the final combined CSV to `output/chi_input_codeeditorbench_all_models.csv`.
