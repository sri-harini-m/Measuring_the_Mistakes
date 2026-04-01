from typing import Dict, Optional

from core.config import (
    REFACTORING_PROMPT_TEMPLATE,
    ADDITIONAL_INSTRUCTIONS_TEMPLATE,
    SINGLE_FILE_CONSTRAINT
)
from core.parser import parse_llm_response
from core.llm_adapters.base import BaseLLMAdapter
from core.llm_adapters.gemini_adapter import GeminiAdapter
from core.llm_adapters.huggingface_adapter import HuggingFaceAdapter
from core.llm_adapters.claude_adapter import ClaudeAdapter
from core.llm_adapters.openai_adapter import OpenAIAdapter


def create_llm_adapter(provider: str, model: str, timeout: int = 300, rate_limit_delay: float = 0.0,
    use_api: bool = False) -> BaseLLMAdapter:
    provider_lower: str = provider.lower()

    if provider_lower == "gemini":
        return GeminiAdapter(model=model, timeout=timeout, rate_limit_delay=rate_limit_delay)
    elif provider_lower == "huggingface":
        return HuggingFaceAdapter(model=model, timeout=timeout, use_api=use_api)
    elif provider_lower == "claude":
        return ClaudeAdapter(model=model, timeout=timeout, rate_limit_delay=rate_limit_delay)
    elif provider_lower == "openai":
        return OpenAIAdapter(model=model, timeout=timeout, rate_limit_delay=rate_limit_delay)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: gemini, huggingface, claude, openai")


def get_llm_refactored_code(code_files: Dict[str, str], instructions: Optional[str], language: str,
    adapter: BaseLLMAdapter) -> Dict[str, str]:
    prompt: str = build_refactoring_prompt(code_files, instructions, language)
    response: Optional[str] = adapter.generate(prompt)
    
    if not response:
        return {}
    
    refactored_files: Dict[str, str] = parse_llm_response(response, code_files.keys(), language)
    return refactored_files


def build_refactoring_prompt(
    code_files: Dict[str, str],
    instructions: Optional[str],
    language: str
) -> str:
    code_blocks: list[str] = []
    for filename, code in code_files.items():
        code_blocks.append(f"File: {filename}\n{code}")
    
    code_section: str = "\n\n".join(code_blocks)
    
    single_file_constraint: str = SINGLE_FILE_CONSTRAINT if len(code_files) == 1 else ""
    
    prompt: str = REFACTORING_PROMPT_TEMPLATE.format(
        language=language,
        code_section=code_section,
        single_file_constraint=single_file_constraint
    )
    
    if instructions:
        prompt += ADDITIONAL_INSTRUCTIONS_TEMPLATE.format(instructions=instructions)
    
    return prompt

