from typing import TypedDict, Dict, List, Optional, Union


class HalsteadMetrics(TypedDict, total=False):
    h1: Union[int, float]
    h2: Union[int, float]
    N1: Union[int, float]
    N2: Union[int, float]
    vocabulary: Union[int, float]
    length: Union[int, float]
    calculated_length: Union[int, float]
    volume: Union[int, float]
    difficulty: Union[int, float]
    effort: Union[int, float]
    time: Union[int, float]
    bugs: Union[int, float]


class CodeMetrics(TypedDict):
    cyclomatic_complexity_total: Union[int, float, str]
    loc: Union[int, float, str]
    num_functions: Union[int, float, str]
    token_count: Union[int, float, str]
    cognitive_complexity_total: Union[int, float, str]
    halstead: Union[HalsteadMetrics, Dict]


class FileMetrics(TypedDict):
    filename: str
    original_metrics: Optional[CodeMetrics]
    refactored_metrics: CodeMetrics
    codebleu: Optional[float]
    is_new_file: bool


class ImprovementMetrics(TypedDict):
    complexity_reduction: Union[int, float]
    loc_reduction: Union[int, float]


class TestResults(TypedDict, total=False):
    public_passed: int
    public_total: int
    public_pass_rate: float
    private_passed: int
    private_total: int
    private_pass_rate: float
    total_passed: int
    total_tests: int
    total_pass_rate: float
    execution_time: float


class RefactoringResult(TypedDict):
    kata_key: str
    kata_name: str
    language: str
    num_files: int
    num_new_files: int
    refactoring_time_seconds: float
    avg_codebleu: float
    file_metrics: List[FileMetrics]
    improvement: ImprovementMetrics
    output_paths: List[str]
    test_results: Optional[TestResults]


class SummaryReport(TypedDict, total=False):
    total_katas_processed: int
    total_files_refactored: int
    total_new_files_created: int
    average_codebleu: str
    average_cyclomatic_complexity_reduction: str
    average_cognitive_complexity_reduction: Optional[str]
    average_loc_reduction: str
    average_num_functions_change: str
    average_token_count_change: str
    average_input_cyclomatic_complexity: str
    average_input_cognitive_complexity: Optional[str]
    average_input_loc: str
    average_input_num_functions: str
    average_input_token_count: str
    average_output_cyclomatic_complexity: str
    average_output_cognitive_complexity: Optional[str]
    average_output_loc: str
    average_output_num_functions: str
    average_output_token_count: str
    average_halstead_volume_change: Optional[str]
    average_halstead_difficulty_change: Optional[str]
    average_halstead_effort_change: Optional[str]
    average_halstead_time_change: Optional[str]
    average_halstead_bugs_change: Optional[str]
    average_input_halstead_volume: Optional[str]
    average_input_halstead_difficulty: Optional[str]
    average_input_halstead_effort: Optional[str]
    average_input_halstead_bugs: Optional[str]
    average_output_halstead_volume: Optional[str]
    average_output_halstead_difficulty: Optional[str]
    average_output_halstead_effort: Optional[str]
    average_output_halstead_bugs: Optional[str]


class TestCase(TypedDict):
    input: str
    output: str


class KataData(TypedDict):
    kata_name: str
    code_files: Dict[str, str]
    instructions: Optional[str]
    language: str
    public_tests: Optional[List[TestCase]]
    private_tests: Optional[List[TestCase]]

