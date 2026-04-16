# Debugging Tasks Scripts Quick Guide

1. Debugging task scripts all start with `debug`. For closed-source models, the provider appears in the filename (for example: `debug_python_gpt.py`, `debug_java_claude.py`, `debug_cpp_gemini.py`). Open-source scripts are in `scripts/open_source/` as `debug_python.py`, `debug_cpp.py`, and `debug_java.py`.

2. Set required credentials before running:
   - OpenAI scripts (`scripts/gpt/*`) require `OPENAI_API_KEY`.
   - Claude scripts (`scripts/claude/*`) require `ANTHROPIC_API_KEY`.
   - Gemini scripts (`scripts/gemini/*`) require `GEMINI_API_KEY`.
   - Open-source scripts (`scripts/open_source/*`) require `--hf_token` (and optionally `--model_id`).

3. Run one script per language/provider combination. Typical examples:
   - `python scripts/open_source/debug_python.py --hf_token YOUR_HF_TOKEN`
   - `python scripts/claude/debug_python_claude.py`
   - `python scripts/gemini/debug_cpp_gemini.py`
   - `python scripts/gpt/debug_java_gpt.py`

4. Dataset and output behavior:
   - Scripts read from `processed_dataset/verified_{language}_dataset.jsonl`.
   - Progress is saved to checkpoint files, and final outputs are saved to provider/model-specific results JSON files.
   - Re-running a script resumes automatically from the saved checkpoint when available.
