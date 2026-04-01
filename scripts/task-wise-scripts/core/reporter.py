import json
from typing import List, Union, Dict, Any

from core.models import RefactoringResult, SummaryReport


def generate_summary_report(results: List[RefactoringResult], output_file: str, dataset_type: str = 'katas', 
                            total_attempted: int = 0, total_failed: int = 0) -> None:
    if not results:
        print("No results to summarize.")
        return
    
    cyclomatic_improvements: List[Union[int, float]] = []
    cognitive_improvements: List[Union[int, float]] = []
    loc_improvements: List[Union[int, float]] = []
    num_functions_changes: List[Union[int, float]] = []
    token_count_changes: List[Union[int, float]] = []
    
    input_cyclomatic: List[Union[int, float]] = []
    input_cognitive: List[Union[int, float]] = []
    input_locs: List[Union[int, float]] = []
    input_num_functions: List[Union[int, float]] = []
    input_token_counts: List[Union[int, float]] = []
    
    output_cyclomatic: List[Union[int, float]] = []
    output_cognitive: List[Union[int, float]] = []
    output_locs: List[Union[int, float]] = []
    output_num_functions: List[Union[int, float]] = []
    output_token_counts: List[Union[int, float]] = []
    
    input_halstead_volume: List[Union[int, float]] = []
    input_halstead_difficulty: List[Union[int, float]] = []
    input_halstead_effort: List[Union[int, float]] = []
    input_halstead_bugs: List[Union[int, float]] = []
    
    output_halstead_volume: List[Union[int, float]] = []
    output_halstead_difficulty: List[Union[int, float]] = []
    output_halstead_effort: List[Union[int, float]] = []
    output_halstead_bugs: List[Union[int, float]] = []
    
    total_new_files: int = 0
    total_all_tests_passed: int = 0
    total_tests_passed: int = 0
    total_tests: int = 0
    test_pass_rates_per_problem: List[float] = []
    
    for result in results:
        total_new_files += result.get('num_new_files', 0)
        
        if 'test_results' in result and result['test_results'] is not None:
            test_res = result['test_results']
            passed = test_res.get('total_passed', 0)
            total = test_res.get('total_tests', 0)
            
            total_tests_passed += passed
            total_tests += total
            
            if total > 0:
                pass_rate = (passed / total) * 100
                test_pass_rates_per_problem.append(pass_rate)
                
                if passed == total:
                    total_all_tests_passed += 1
        
        if 'file_metrics' in result:
            for file_metric in result['file_metrics']:
                orig = file_metric.get('original_metrics')
                refac = file_metric.get('refactored_metrics')
                
                if orig and refac:
                    orig_cyc = orig.get('cyclomatic_complexity_total')
                    refac_cyc = refac.get('cyclomatic_complexity_total')
                    if isinstance(orig_cyc, (int, float)) and isinstance(refac_cyc, (int, float)):
                        cyclomatic_improvements.append(orig_cyc - refac_cyc)
                    
                    orig_cog = orig.get('cognitive_complexity_total')
                    refac_cog = refac.get('cognitive_complexity_total')
                    if isinstance(orig_cog, (int, float)) and isinstance(refac_cog, (int, float)):
                        cognitive_improvements.append(orig_cog - refac_cog)
                    
                    orig_loc = orig.get('loc')
                    refac_loc = refac.get('loc')
                    if isinstance(orig_loc, (int, float)) and isinstance(refac_loc, (int, float)):
                        loc_improvements.append(orig_loc - refac_loc)
                    
                    orig_funcs = orig.get('num_functions')
                    refac_funcs = refac.get('num_functions')
                    if isinstance(orig_funcs, (int, float)) and isinstance(refac_funcs, (int, float)):
                        num_functions_changes.append(orig_funcs - refac_funcs)
                    
                    orig_tokens = orig.get('token_count')
                    refac_tokens = refac.get('token_count')
                    if isinstance(orig_tokens, (int, float)) and isinstance(refac_tokens, (int, float)):
                        token_count_changes.append(orig_tokens - refac_tokens)
                    
                if orig:
                    cyc = orig.get('cyclomatic_complexity_total')
                    if isinstance(cyc, (int, float)):
                        input_cyclomatic.append(cyc)
                    
                    cog = orig.get('cognitive_complexity_total')
                    if isinstance(cog, (int, float)):
                        input_cognitive.append(cog)
                    
                    loc = orig.get('loc')
                    if isinstance(loc, (int, float)):
                        input_locs.append(loc)
                    
                    funcs = orig.get('num_functions')
                    if isinstance(funcs, (int, float)):
                        input_num_functions.append(funcs)
                    
                    tokens = orig.get('token_count')
                    if isinstance(tokens, (int, float)):
                        input_token_counts.append(tokens)
                    
                    halstead: Dict[str, Any] = orig.get('halstead', {})
                    if halstead:
                        for value, metric_list, key in [
                            (halstead.get('volume'), input_halstead_volume, 'volume'),
                            (halstead.get('difficulty'), input_halstead_difficulty, 'difficulty'),
                            (halstead.get('effort'), input_halstead_effort, 'effort'),
                            (halstead.get('bugs'), input_halstead_bugs, 'bugs')
                        ]:
                            if isinstance(value, (int, float)):
                                metric_list.append(value)
                
                if refac:
                    cyc = refac.get('cyclomatic_complexity_total')
                    if isinstance(cyc, (int, float)):
                        output_cyclomatic.append(cyc)
                    
                    cog = refac.get('cognitive_complexity_total')
                    if isinstance(cog, (int, float)):
                        output_cognitive.append(cog)
                    
                    loc = refac.get('loc')
                    if isinstance(loc, (int, float)):
                        output_locs.append(loc)
                    
                    funcs = refac.get('num_functions')
                    if isinstance(funcs, (int, float)):
                        output_num_functions.append(funcs)
                    
                    tokens = refac.get('token_count')
                    if isinstance(tokens, (int, float)):
                        output_token_counts.append(tokens)
                    
                    halstead: Dict[str, Any] = refac.get('halstead', {})
                    if halstead:
                        for value, metric_list, key in [
                            (halstead.get('volume'), output_halstead_volume, 'volume'),
                            (halstead.get('difficulty'), output_halstead_difficulty, 'difficulty'),
                            (halstead.get('effort'), output_halstead_effort, 'effort'),
                            (halstead.get('bugs'), output_halstead_bugs, 'bugs')
                        ]:
                            if isinstance(value, (int, float)):
                                metric_list.append(value)
    
    avg_cognitive_reduction: float = calculate_safe_average(cognitive_improvements)
    avg_input_cognitive: float = calculate_safe_average(input_cognitive)
    avg_output_cognitive: float = calculate_safe_average(output_cognitive)
    
    count_key = 'total_problems_processed' if dataset_type == 'codeeditorbench' else 'total_katas_processed'
    attempted_key = 'total_problems_attempted' if dataset_type == 'codeeditorbench' else 'total_katas_attempted'
    failed_key = 'total_problems_failed' if dataset_type == 'codeeditorbench' else 'total_katas_failed'
    
    summary_dict = {
        count_key: len(results),
        attempted_key: total_attempted if total_attempted > 0 else len(results),
        failed_key: total_failed,
        'total_files_refactored': sum(r.get('num_files', 0) for r in results),
        'total_new_files_created': total_new_files,
        'average_codebleu': f"{calculate_safe_average([r.get('avg_codebleu', 0) for r in results]):.4f}",
        'average_cyclomatic_complexity_reduction': f"{calculate_safe_average(cyclomatic_improvements):.2f}",
        'average_loc_reduction': f"{calculate_safe_average(loc_improvements):.2f}",
        'average_num_functions_change': f"{calculate_safe_average(num_functions_changes):.2f}",
        'average_token_count_change': f"{calculate_safe_average(token_count_changes):.2f}",
        'average_input_cyclomatic_complexity': f"{calculate_safe_average(input_cyclomatic):.2f}",
        'average_input_loc': f"{calculate_safe_average(input_locs):.2f}",
        'average_input_num_functions': f"{calculate_safe_average(input_num_functions):.2f}",
        'average_input_token_count': f"{calculate_safe_average(input_token_counts):.2f}",
        'average_output_cyclomatic_complexity': f"{calculate_safe_average(output_cyclomatic):.2f}",
        'average_output_loc': f"{calculate_safe_average(output_locs):.2f}",
        'average_output_num_functions': f"{calculate_safe_average(output_num_functions):.2f}",
        'average_output_token_count': f"{calculate_safe_average(output_token_counts):.2f}"
    }
    
    if total_tests > 0:
        summary_dict['total_problems_all_tests_passed'] = total_all_tests_passed
        summary_dict['total_test_cases_passed'] = total_tests_passed
        summary_dict['total_test_cases'] = total_tests
        summary_dict['average_test_pass_rate_per_problem'] = f"{calculate_safe_average(test_pass_rates_per_problem):.2f}%"
        summary_dict['overall_test_pass_rate'] = f"{(total_tests_passed / total_tests * 100) if total_tests > 0 else 0:.2f}%"
    
    summary: SummaryReport = SummaryReport(**summary_dict)
    
    if avg_cognitive_reduction != 0:
        summary['average_cognitive_complexity_reduction'] = f"{avg_cognitive_reduction:.2f}"
    if avg_input_cognitive != 0:
        summary['average_input_cognitive_complexity'] = f"{avg_input_cognitive:.2f}"
    if avg_output_cognitive != 0:
        summary['average_output_cognitive_complexity'] = f"{avg_output_cognitive:.2f}"
    
    if input_halstead_volume:
        summary['average_input_halstead_volume'] = f"{calculate_safe_average(input_halstead_volume):.2f}"
    if input_halstead_difficulty:
        summary['average_input_halstead_difficulty'] = f"{calculate_safe_average(input_halstead_difficulty):.2f}"
    if input_halstead_effort:
        summary['average_input_halstead_effort'] = f"{calculate_safe_average(input_halstead_effort):.2f}"
    if input_halstead_bugs:
        summary['average_input_halstead_bugs'] = f"{calculate_safe_average(input_halstead_bugs):.4f}"
    
    if output_halstead_volume:
        summary['average_output_halstead_volume'] = f"{calculate_safe_average(output_halstead_volume):.2f}"
    if output_halstead_difficulty:
        summary['average_output_halstead_difficulty'] = f"{calculate_safe_average(output_halstead_difficulty):.2f}"
    if output_halstead_effort:
        summary['average_output_halstead_effort'] = f"{calculate_safe_average(output_halstead_effort):.2f}"
    if output_halstead_bugs:
        summary['average_output_halstead_bugs'] = f"{calculate_safe_average(output_halstead_bugs):.4f}"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary report saved to {output_file}")
    if dataset_type == 'codeeditorbench':
        print(f"  Total problems attempted: {summary['total_problems_attempted']}")
        print(f"  Total problems processed successfully: {summary['total_problems_processed']}")
        print(f"  Total problems failed: {summary['total_problems_failed']}")
    else:
        print(f"  Total katas attempted: {summary['total_katas_attempted']}")
        print(f"  Total katas processed successfully: {summary['total_katas_processed']}")
        print(f"  Total katas failed: {summary['total_katas_failed']}")
    print(f"  Total files refactored: {summary['total_files_refactored']}")
    print(f"  Total new files created: {summary['total_new_files_created']}")
    
    if 'total_test_cases' in summary and summary['total_test_cases'] > 0:
        print(f"\n  Test Results:")
        print(f"    Problems with all tests passed: {summary['total_problems_all_tests_passed']}")
        print(f"    Total test cases passed: {summary['total_test_cases_passed']}/{summary['total_test_cases']}")
        print(f"    Overall test pass rate: {summary['overall_test_pass_rate']}")
        print(f"    Average test pass rate per problem: {summary['average_test_pass_rate_per_problem']}")
    
    print(f"\n  Quality Metrics:")
    print(f"    Average CodeBLEU: {summary['average_codebleu']}")
    print(f"\n  Improvement Metrics:")
    print(f"    Cyclomatic complexity reduction: {summary['average_cyclomatic_complexity_reduction']}")
    if 'average_cognitive_complexity_reduction' in summary:
        print(f"    Cognitive complexity reduction: {summary['average_cognitive_complexity_reduction']}")
    print(f"    LOC reduction: {summary['average_loc_reduction']}")
    print(f"    Num functions change: {summary['average_num_functions_change']}")
    print(f"    Token count change: {summary['average_token_count_change']}")
    print(f"\n  Input Code Metrics:")
    print(f"    Cyclomatic complexity: {summary['average_input_cyclomatic_complexity']}")
    if 'average_input_cognitive_complexity' in summary:
        print(f"    Cognitive complexity: {summary['average_input_cognitive_complexity']}")
    print(f"    LOC: {summary['average_input_loc']}")
    print(f"    Num functions: {summary['average_input_num_functions']}")
    print(f"    Token count: {summary['average_input_token_count']}")
    if 'average_input_halstead_volume' in summary:
        print(f"    Halstead volume: {summary['average_input_halstead_volume']}")
    if 'average_input_halstead_difficulty' in summary:
        print(f"    Halstead difficulty: {summary['average_input_halstead_difficulty']}")
    if 'average_input_halstead_effort' in summary:
        print(f"    Halstead effort: {summary['average_input_halstead_effort']}")
    if 'average_input_halstead_bugs' in summary:
        print(f"    Halstead bugs: {summary['average_input_halstead_bugs']}")
    print(f"\n  Output Code Metrics:")
    print(f"    Cyclomatic complexity: {summary['average_output_cyclomatic_complexity']}")
    if 'average_output_cognitive_complexity' in summary:
        print(f"    Cognitive complexity: {summary['average_output_cognitive_complexity']}")
    print(f"    LOC: {summary['average_output_loc']}")
    print(f"    Num functions: {summary['average_output_num_functions']}")
    print(f"    Token count: {summary['average_output_token_count']}")
    if 'average_output_halstead_volume' in summary:
        print(f"    Halstead volume: {summary['average_output_halstead_volume']}")
    if 'average_output_halstead_difficulty' in summary:
        print(f"    Halstead difficulty: {summary['average_output_halstead_difficulty']}")
    if 'average_output_halstead_effort' in summary:
        print(f"    Halstead effort: {summary['average_output_halstead_effort']}")
    if 'average_output_halstead_bugs' in summary:
        print(f"    Halstead bugs: {summary['average_output_halstead_bugs']}")


def calculate_safe_average(metric_list: List[Union[int, float]]) -> float:
    values: List[Union[int, float]] = [v for v in metric_list if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else 0.0

