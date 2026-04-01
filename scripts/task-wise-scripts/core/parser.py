import re
from typing import Dict, Set, Optional, List


def strip_markdown_fences(code: str) -> str:
    code = code.strip()
    code = re.sub(r'^```[a-zA-Z+]*\s*\n?', '', code)
    code = re.sub(r'\n?```\s*$', '', code)
    return code.strip()


def strip_filename_prefix(code: str) -> str:
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.match(r'^FILENAME:\s*\S+\.(py|java|cpp|cc|c|h|hpp|cxx)\s*$', line, re.IGNORECASE):
            continue
        if re.match(r'^(//|#)\s*File:\s*\S+\.(py|java|cpp|cc|c|h|hpp|cxx)\s*$', line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return strip_markdown_fences('\n'.join(cleaned_lines).strip())


def extract_filename_from_comment(code: str) -> Optional[str]:
    lines = code.strip().split('\n')
    if not lines:
        return None
    first_line = lines[0].strip()
    match = re.match(r'^(//|#)\s*File:\s*(\S+\.(py|java|cpp|cc|c|h|hpp|cxx))\s*$', first_line, re.IGNORECASE)
    if match:
        return match.group(2)
    return None


def parse_llm_response(response: str, original_filenames: Set[str], language: str) -> Dict[str, str]:
    refactored_files: Dict[str, str] = {}
    
    file_pattern: re.Pattern = re.compile(
        r'FILENAME:\s*(\S+)\s*\n(.*?)(?=FILENAME:|$)', 
        re.DOTALL | re.IGNORECASE
    )
    matches: list = file_pattern.findall(response)
    
    if not matches:
        file_comment_pattern: re.Pattern = re.compile(
            r'(//|#)\s*File:\s*(\S+\.(py|java|cpp|cc|c|h|hpp|cxx))\s*\n(.*?)(?=(//|#)\s*File:\s*\S+\.(py|java|cpp|cc|c|h|hpp|cxx)|$)',
            re.DOTALL | re.IGNORECASE
        )
        comment_matches: list = file_comment_pattern.findall(response)
        if comment_matches:
            for _, filename, _, code, *_ in comment_matches:
                cleaned_filename: str = clean_filename(filename)
                extracted_code: Optional[str] = extract_code_from_markdown(code.strip(), language)
                if extracted_code:
                    refactored_files[cleaned_filename] = extracted_code
                elif code.strip():
                    refactored_files[cleaned_filename] = strip_filename_prefix(code.strip())
            if refactored_files:
                return refactored_files
    
    if matches:
        for filename, code in matches:
            cleaned_filename: str = clean_filename(filename)
            extracted_code: Optional[str] = extract_code_from_markdown(code.strip(), language)
            if extracted_code:
                refactored_files[cleaned_filename] = extracted_code
    else:
        markdown_bold_pattern: re.Pattern = re.compile(
            r'\*\*([^\*]+\.(py|java|cpp|cc|h|hpp|c))\*\*\s*\n(.*?)(?=\*\*[^\*]+\.(py|java|cpp|cc|h|hpp|c)\*\*|$)',
            re.DOTALL | re.IGNORECASE
        )
        markdown_matches: list = markdown_bold_pattern.findall(response)
        
        if markdown_matches:
            for filename, _, code, *_ in markdown_matches:
                cleaned_filename: str = clean_filename(filename)
                extracted_code: Optional[str] = extract_code_from_markdown(code.strip(), language)
                if extracted_code:
                    refactored_files[cleaned_filename] = extracted_code
        else:
            plain_filename_pattern: re.Pattern = re.compile(
                r'^([^\s]+\.(py|java|cpp|cc|h|hpp|c))\s*\n(.*?)(?=^[^\s]+\.(py|java|cpp|cc|h|hpp|c)\s*\n|$)',
                re.DOTALL | re.IGNORECASE | re.MULTILINE
            )
            plain_matches: list = plain_filename_pattern.findall(response)
            
            if plain_matches:
                for filename, _, code, *_ in plain_matches:
                    cleaned_filename: str = clean_filename(filename)
                    extracted_code: Optional[str] = extract_code_from_markdown(code.strip(), language)
                    if extracted_code:
                        refactored_files[cleaned_filename] = extracted_code
            elif len(original_filenames) == 1:
                extracted_code: Optional[str] = extract_code_from_markdown(response.strip(), language)
                if extracted_code:
                    refactored_files[list(original_filenames)[0]] = extracted_code
    
    if not refactored_files:
        orphan_code_blocks: List[str] = extract_all_code_blocks(response, language)
        if orphan_code_blocks:
            if len(original_filenames) == 1:
                refactored_files[list(original_filenames)[0]] = orphan_code_blocks[0]
            else:
                extension: str = get_file_extension(language)
                for idx, code in enumerate(orphan_code_blocks):
                    extracted_filename = extract_filename_from_comment(code)
                    if extracted_filename:
                        refactored_files[extracted_filename] = strip_filename_prefix(code)
                    else:
                        filename: str = f"refactored_{idx}{extension}"
                        refactored_files[filename] = code
    
    return refactored_files


def extract_code_from_markdown(text: str, language: str) -> Optional[str]:
    lang_identifiers = {
        'python': ['python', 'py', 'python3'],
        'java': ['java'],
        'cpp': ['cpp', 'c++', 'c', 'cc', 'cxx']
    }
    
    identifiers = lang_identifiers.get(language, [language])
    
    for identifier in identifiers:
        pattern = re.compile(rf'```{identifier}\n(.*?)```', re.DOTALL)
        match = pattern.search(text)
        if match:
            return strip_filename_prefix(match.group(1).strip())
    
    generic_pattern = re.compile(r'```\n(.*?)```', re.DOTALL)
    match = generic_pattern.search(text)
    if match:
        return strip_filename_prefix(match.group(1).strip())
    
    no_newline_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    match = no_newline_pattern.search(text)
    if match:
        code = match.group(1).strip()
        for identifier in identifiers:
            if code.startswith(identifier + '\n'):
                code = code[len(identifier) + 1:].strip()
                break
        return strip_filename_prefix(code)
    
    return None


def extract_all_code_blocks(text: str, language: str) -> List[str]:
    code_blocks: List[str] = []
    
    lang_identifiers = {
        'python': ['python', 'py', 'python3'],
        'java': ['java'],
        'cpp': ['cpp', 'c++', 'c', 'cc', 'cxx']
    }
    
    identifiers = lang_identifiers.get(language, [language])
    
    for identifier in identifiers:
        pattern = re.compile(rf'```{identifier}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            code_blocks.extend([strip_filename_prefix(match.strip()) for match in matches])
    
    if not code_blocks:
        generic_pattern = re.compile(r'```\n(.*?)```', re.DOTALL)
        matches = generic_pattern.findall(text)
        if matches:
            code_blocks.extend([strip_filename_prefix(match.strip()) for match in matches])
    
    if not code_blocks:
        no_newline_pattern = re.compile(r'```(.*?)```', re.DOTALL)
        matches = no_newline_pattern.findall(text)
        for match in matches:
            code = match.strip()
            for identifier in identifiers:
                if code.startswith(identifier + '\n'):
                    code = code[len(identifier) + 1:].strip()
                    break
            if code:
                code_blocks.append(strip_filename_prefix(code))
    
    return code_blocks


def get_file_extension(language: str) -> str:
    extensions = {
        'python': '.py',
        'java': '.java',
        'cpp': '.cpp',
        'c': '.c'
    }
    return extensions.get(language, '.txt')


def clean_filename(filename: str) -> str:
    filename = filename.strip()
    filename = re.sub(r'[*_`]', '', filename)
    filename = re.sub(r'^["\']|["\']$', '', filename)
    return filename

