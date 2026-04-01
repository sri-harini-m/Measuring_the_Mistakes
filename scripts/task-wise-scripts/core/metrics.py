import os
from typing import Union
import lizard
from codebleu import calc_codebleu
from radon.complexity import cc_visit
from radon.metrics import h_visit

from core.config import LANGUAGE_EXTENSIONS
from core.models import CodeMetrics, HalsteadMetrics


def calculate_code_metrics(code: str, language: str = 'python') -> CodeMetrics:
    metrics: CodeMetrics = CodeMetrics(
        cyclomatic_complexity_total='N/A',
        loc='N/A',
        num_functions='N/A',
        token_count='N/A',
        cognitive_complexity_total='N/A',
        halstead={}
    )
    
    temp_filename: str = f"temp_code_for_analysis.{get_file_extension(language)}"
    
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(code)
    
    try:
        analysis = lizard.analyze_file(temp_filename)
        
        if analysis.function_list:
            metrics['cyclomatic_complexity_total'] = sum(
                f.cyclomatic_complexity for f in analysis.function_list
            )
        else:
            num_conditionals: int = code.count('if ') + code.count('elif ') + code.count('else:')
            num_loops: int = code.count('for ') + code.count('while ')
            num_logical_ops: int = code.count(' and ') + code.count(' or ')
            estimated_complexity: int = 1 + num_conditionals + num_loops + num_logical_ops
            metrics['cyclomatic_complexity_total'] = estimated_complexity
        
        metrics['loc'] = analysis.nloc
        metrics['num_functions'] = len(analysis.function_list)
        metrics['token_count'] = analysis.token_count
    except Exception:
        pass
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    if language == 'python':
        try:
            blocks = cc_visit(code)
            metrics['cognitive_complexity_total'] = sum(b.complexity for b in blocks)
        except Exception:
            pass
        
        try:
            halstead = h_visit(code)
            if halstead and halstead.total:
                metrics['halstead'] = halstead.total._asdict()
        except Exception:
            pass
    
    return metrics


def calculate_codebleu_score(original: str, refactored: str, language: str = 'python') -> float:
    try:
        result: dict = calc_codebleu([original], [refactored], language)
        return result.get('codebleu', 0.0)
    except Exception as e:
        print(f"     CodeBLEU calculation error: {type(e).__name__}: {e}")
        return 0.0


def get_file_extension(language: str) -> str:
    return LANGUAGE_EXTENSIONS.get(language, 'txt')


def calculate_improvement(
    file_metrics: list,
    metric_name: str
) -> Union[int, float]:
    original_total: Union[int, float] = sum(
        fm['original_metrics'].get(metric_name, 0)
        for fm in file_metrics
        if fm.get('original_metrics') is not None 
        and isinstance(fm['original_metrics'].get(metric_name), (int, float))
    )
    refactored_total: Union[int, float] = sum(
        fm['refactored_metrics'].get(metric_name, 0)
        for fm in file_metrics
        if fm.get('original_metrics') is not None
        and isinstance(fm['refactored_metrics'].get(metric_name), (int, float))
    )
    return original_total - refactored_total

