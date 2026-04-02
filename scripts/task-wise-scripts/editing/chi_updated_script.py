import pandas as pd
import tempfile
import os
import json
import math
import sys
import re
import lizard
import logging
import ast
from typing import Optional, List
import numpy as np

try:
    from radon.complexity import cc_visit

    RADON_AVAILABLE = True
except ImportError:
    print("Warning: radon not installed. Complexity analysis will be limited.")
    print("Install with: pip install radon")
    RADON_AVAILABLE = False

try:
    from cognitive_complexity.api import get_cognitive_complexity

    COGNITIVE_COMPLEXITY_AVAILABLE = True
except ImportError:
    print("Warning: cognitive-complexity not installed. Python cognitive complexity disabled.")
    print("Install with: pip install cognitive-complexity")
    COGNITIVE_COMPLEXITY_AVAILABLE = False

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    import tree_sitter_java as tsjava

    TREE_SITTER_AVAILABLE = True
except ImportError:
    print("Warning: tree-sitter or language bindings not installed.")
    print("Install with: pip install tree-sitter tree-sitter-cpp tree-sitter-java")
    TREE_SITTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# -------------------------
# Tree-sitter node type sets (cognitive complexity)
# -------------------------

CONTROL_NODES_WITH_NESTING = {
    "java": {
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "switch_expression",
        "catch_clause",
        "conditional_expression",
        "ternary_expression",
        "synchronized_statement",
    },
    "cpp": {
        "if_statement",
        "for_statement",
        "range_based_for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "catch_clause",
        "conditional_expression",
    },
}

CONTROL_NODES_NO_NESTING = {
    "java": {"else"},
    "cpp": {"else_clause"},
}

JUMP_NODES = {
    "java": set(),
    "cpp": {"goto_statement"},
}

FUNCTION_NODES = {
    "java": {"method_declaration", "constructor_declaration"},
    "cpp": {"function_definition"},
}

NESTING_IMMUNE = {
    "java": {"lambda_expression"},
    "cpp": {"lambda_expression"},
}

JUMP_STATEMENTS = {
    "java": {"break_statement", "continue_statement"},
    "cpp": set(),
}

CONDITION_BEARING_NODES = {
    "java": {
        "if_statement",
        "while_statement",
        "for_statement",
        "do_statement",
        "conditional_expression",
        "ternary_expression",
    },
    "cpp": {
        "if_statement",
        "while_statement",
        "for_statement",
        "do_statement",
        "conditional_expression",
    },
}

# -------------------------
# Hallucination category definitions
# -------------------------

HALLUCINATION_CATEGORIES = {
    'Mapping_Hallucination': [
        'Data_Compliance_Hallucination',
        'Structural_Access_Hallucination'
    ],
    'Naming_Hallucination': [
        'Identification_Hallucination',
        'External_Source_Hallucination'
    ],
    'Resource_Hallucination': [
        'Physical_Constraint_Hallucination',
        'Calculate_Boundary_Hallucination'
    ],
    'Logical_Hallucination': [
        'Logic_Deviation',
        'Logic_Breakdown'
    ],
    'Syntax_Error': [
        'Syntax_Error'
    ],
    'TimeError': [
        'TimeError'
    ]
}

MAIN_CATEGORIES = ['Mapping_Hallucination', 'Naming_Hallucination', 'Resource_Hallucination', 'Logical_Hallucination']

programming_halus_cpp = {
    "Data_Compliance_Hallucination": {
        "Type mismatch": "Type mismatch",
        "Invalid conversion": "Invalid conversion",
        "Division by zero": "Division by zero",
    },
    "Structural_Access_Hallucination": {
        "Array index out of bounds": "Array index out of bounds",
        "Segmentation fault": "Segmentation fault",
        "Out of range": "Out of range"
    },
    "Identification_Hallucination": {
        "Undeclared identifier": "Undeclared identifier",
        "Not declared": "Not declared in this scope",
        "Undefined reference": "Undefined reference",
    },
    "External_Source_Hallucination": {
        "No such file": "No such file or directory",
        "Cannot find": "Cannot find",
    },
    "Physical_Constraint_Hallucination": {
        "Stack overflow": "Stack overflow",
        "Memory exhausted": "Memory exhausted",
    },
    "Calculate_Boundary_Hallucination": {
        "Overflow": "Overflow",
        "Arithmetic exception": "Arithmetic exception"
    },
    "Logic_Deviation": {
        "Logic_Deviation": "Logic_Deviation",
        "Assertion_Error": "Assertion"
    },
    "Logic_Breakdown": {
        "Logic_Breakdown": "Logic_Breakdown"
    },
    "Syntax_Error": {
        "Syntax error": "Syntax error",
        "Parse error": "Expected",
        "Compilation error": "error:"
    },
    "TimeError": {
        "TimeError": "TimeError"
    }
}

programming_halus_python = {
    "Data_Compliance_Hallucination": {
        "TypeError": "TypeError",
        "ValueError": "ValueError",
        "ZeroDivisionError": "ZeroDivisionError",
    },
    "Structural_Access_Hallucination": {
        "IndexError": "IndexError",
        "KeyError": "KeyError"
    },
    "Identification_Hallucination": {
        "NameError": "NameError",
        "AttributeError": "AttributeError",
        "UnboundLocalError": "UnboundLocalError",
    },
    "External_Source_Hallucination": {
        "ImportError": "ImportError",
        "ModuleNotFoundError": "ModuleNotFoundError"
    },
    "Physical_Constraint_Hallucination": {
        "RecursionError": "RecursionError",
        "MemoryError": "MemoryError",
    },
    "Calculate_Boundary_Hallucination": {
        "OverflowError": "OverflowError",
        "StopIteration": "StopIteration"
    },
    "Logic_Deviation": {
        "Logic_Deviation": "Logic_Deviation",
        "AssertionError": "AssertionError",
    },
    "Logic_Breakdown": {
        "Logic_Breakdown": "Logic_Breakdown"
    },
    "Syntax_Error": {
        "SyntaxError": "SyntaxError",
        "IndentationError": "IndentationError"
    },
    "TimeError": {
        "TimeError": "TimeError"
    }
}

programming_halus_java = {
    "Data_Compliance_Hallucination": {
        "Type mismatch": "incompatible types",
        "NumberFormatException": "NumberFormatException",
        "ArithmeticException": "ArithmeticException",
    },
    "Structural_Access_Hallucination": {
        "ArrayIndexOutOfBoundsException": "ArrayIndexOutOfBoundsException",
        "IndexOutOfBoundsException": "IndexOutOfBoundsException",
        "NullPointerException": "NullPointerException"
    },
    "Identification_Hallucination": {
        "Cannot find symbol": "cannot find symbol",
        "Package does not exist": "package .* does not exist",
        "Cannot resolve": "cannot be resolved",
    },
    "External_Source_Hallucination": {
        "ClassNotFoundException": "ClassNotFoundException",
        "NoClassDefFoundError": "NoClassDefFoundError",
    },
    "Physical_Constraint_Hallucination": {
        "StackOverflowError": "StackOverflowError",
        "OutOfMemoryError": "OutOfMemoryError",
    },
    "Calculate_Boundary_Hallucination": {
        "Overflow": "Overflow",
        "ArithmeticException": "ArithmeticException"
    },
    "Logic_Deviation": {
        "Logic_Deviation": "Logic_Deviation",
        "AssertionError": "AssertionError",
    },
    "Logic_Breakdown": {
        "Logic_Breakdown": "Logic_Breakdown"
    },
    "Syntax_Error": {
        "Syntax error": "error:",
        "Parse error": "expected",
        "Compilation error": "error:"
    },
    "TimeError": {
        "TimeError": "TimeError"
    }
}


# =========================
# Error categorization utilities
# =========================

def categorize_error(error_msg, language):
    """Categorize error message into hallucination subcategory."""
    if language in ["python", "python3"]:
        error_patterns = programming_halus_python
    elif language == "cpp":
        error_patterns = programming_halus_cpp
    elif language == "java":
        error_patterns = programming_halus_java
    else:
        return "Unknown"
    
    for halu_type, error_mapping in error_patterns.items():
        for error_key, error_pattern in error_mapping.items():
            if re.search(error_pattern, error_msg, re.IGNORECASE):
                return halu_type

    return "Unknown"


def calculate_hallucination_breakdown(errors_dict, language, total):
    """Calculate breakdown of errors into 8 hallucination subcategories."""
    # errors_dict already has categorized error names as keys
    # (e.g., "Data_Compliance_Hallucination", "Identification_Hallucination", etc.)
    # So we just need to count them directly

    hallucination_breakdown = {}

    for halu_type, count in errors_dict.items():
        if count > 0:
            hallucination_breakdown[halu_type] = {
                'count': count,
                'percentage': round(count / total * 100, 2) if total > 0 else 0
            }

    return hallucination_breakdown

def calculate_category_rates(errors_dict, total):
    """Aggregate error subcategories directly into the 4 main hallucination categories.

    Takes the raw errors_dict (subcategory -> count) and total sample count,
    returns dict mapping each main category to {'count': int, 'percentage': float}.
    """
    category_counts = {cat: 0 for cat in MAIN_CATEGORIES}

    for error_type, count in errors_dict.items():
        for category in MAIN_CATEGORIES:
            if error_type in HALLUCINATION_CATEGORIES.get(category, []):
                category_counts[category] += count
                break

    return {
        cat: {
            'count': cnt,
            'percentage': round(cnt / total * 100, 2) if total > 0 else 0.0
        }
        for cat, cnt in category_counts.items()
    }


# =========================
# Normalization utilities
# =========================

def minmax_norm(x, xmin, xmax):
    """Normalize value to [0, 1] range."""
    if xmax == xmin:
        return 0.0
    return float(np.clip((x - xmin) / (xmax - xmin), 0.0, 1.0))


def harmonic_mean(a, b, eps=1e-8):
    """Calculate harmonic mean of two values."""
    return float(2.0 / (1.0 / (a + eps) + 1.0 / (b + eps)))

def weighted_softmax_complexity(cyc, cog):
    """Weighted combination of complexity metrics."""
    if cyc == 0.0 and cog == 0.0:
        return 0.0
    exp_cyc = np.exp(cyc ** 2)
    exp_cog = np.exp(cog ** 2)
    numerator = cyc * exp_cyc + cog * exp_cog
    denominator = exp_cyc + exp_cog
    return float(numerator / denominator)

def logic_severity(cyc, cog, pass_rate):
    """Calculate logic hallucination severity score."""
    return 0.5 * (weighted_softmax_complexity(cyc, cog) + (1 - pass_rate))

# =========================
# Compilation utilities
# =========================

def strip_markdown_code_fences(code, language):
    """Remove markdown code fences from code if present."""

    if not isinstance(code, str):
        if code is None or (isinstance(code, float) and np.isnan(code)):
            return ""  
        code = str(code)
    code = code.strip()

    patterns = [
        f"```{language}\n",
        f"```{language}",
        "```cpp\n",
        "```cpp",
        "```python\n",
        "```python",
        "```java\n",
        "```java",
        "```\n",
        "```"
    ]

    for pattern in patterns:
        if code.startswith(pattern):
            code = code[len(pattern):]
            break

    if code.endswith("```"):
        code = code[:-3]

    return code.strip()


# =========================
# Cyclomatic Complexity
# =========================

def _tree_sitter_cyclomatic_complexity(code: str, language: str) -> float:
    """Calculate cyclomatic complexity using tree-sitter for Java/C++.

    CC per function = 1 + number of decision points (if, for, while, do,
    case, catch, &&, ||, ternary).
    """
    if not TREE_SITTER_AVAILABLE:
        logger.warning("tree-sitter not available for cyclomatic complexity")
        return 1.0

    try:
        if language in {"cpp", "c++", "c"}:
            lang_obj = Language(tscpp.language())
            lang_key = "cpp"
        elif language == "java":
            lang_obj = Language(tsjava.language())
            lang_key = "java"
        else:
            return 1.0

        parser = Parser(lang_obj)
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)

        decision_types = {
            "java": {
                "if_statement", "for_statement", "enhanced_for_statement",
                "while_statement", "do_statement", "catch_clause",
                "conditional_expression", "ternary_expression",
            },
            "cpp": {
                "if_statement", "for_statement", "range_based_for_statement",
                "while_statement", "do_statement", "catch_clause",
                "conditional_expression",
            },
        }

        case_types = {
            "java": {"switch_label"},
            "cpp": {"case_statement"},
        }

        dt = decision_types.get(lang_key, set())
        ct = case_types.get(lang_key, set())
        fn = FUNCTION_NODES.get(lang_key, set())

        def count_decisions(node):
            count = 0
            if node.type in dt:
                count += 1
            elif node.type in ct:
                count += 1
            elif node.type == "binary_expression":
                for child in node.children:
                    if child.type in {"&&", "||"}:
                        count += 1
                        # break
            for child in node.children:
                if child.type not in fn:
                    count += count_decisions(child)
            return count

        total_cc = 0
        function_count = 0

        def visit(node):
            nonlocal total_cc, function_count
            if node.type in fn:
                function_count += 1
                total_cc += 1 + count_decisions(node)
            else:
                for child in node.children:
                    visit(child)

        visit(tree.root_node)

        if function_count == 0:
            total_cc = 1 + count_decisions(tree.root_node)

        return float(total_cc) if total_cc > 0 else 1.0

    except Exception as e:
        logger.warning(f"Tree-sitter cyclomatic complexity failed: {e}")
        return 1.0

def cyclomatic_complexity(code: str, language: str) -> float:
    language = language.lower()

    if language in ["python", "python3"]:
        if not RADON_AVAILABLE:
            logger.warning("radon not available for Python cyclomatic complexity")
            return 1.0
        try:
            results = cc_visit(code)
            total_cc = sum(block.complexity for block in results)
            return float(total_cc) if total_cc > 0 else 1.0
        except Exception as e:
            logger.warning(f"Radon cyclomatic complexity failed: {e}")
            return 1.0

    if language in {"java", "cpp", "c++", "c"}:
        return _tree_sitter_cyclomatic_complexity(code, language)

    logger.warning(f"Unsupported language for cyclomatic complexity: {language}")
    return 1.0

# -------------------------
# Cognitive Complexity
# -------------------------

def _get_logical_operators(node, code_bytes: bytes, language: str) -> List[str]:
    """
    Extract all logical operators (&&, ||) from a condition in source order
    using in-order traversal of binary expressions.
    """
    ops = []

    def visit(n):
        if n.type == "binary_expression":
            children = list(n.children)
            op_child = None
            op_idx = -1
            for i, child in enumerate(children):
                if child.type in {"&&", "||"}:
                    op_child = child
                    op_idx = i
                    break

            if op_child is not None:
                # In-order: left subtree, operator, right subtree
                for child in children[:op_idx]:
                    visit(child)
                op_text = code_bytes[op_child.start_byte:op_child.end_byte].decode("utf-8", errors="ignore")
                if op_text in {"&&", "||"}:
                    ops.append(op_text)
                for child in children[op_idx + 1:]:
                    visit(child)
            else:
                for child in children:
                    visit(child)
        else:
            for child in n.children:
                visit(child)

    visit(node)
    return ops


def _get_condition_node(node, language: str):
    if node.type in {"if_statement", "while_statement", "do_statement"}:
        for child in node.children:
            if child.type in {"condition", "parenthesized_expression"}:
                return child
        for child in node.children:
            if child.type == "binary_expression":
                return child

    if node.type in {"for_statement"}:
        for child in node.children:
            if child.type == "condition":
                return child
        semicolon_count = 0
        for child in node.children:
            if child.type == ";":
                semicolon_count += 1
            elif semicolon_count == 1 and child.type not in {";", "(", ")"}:
                return child

    if node.type in {"conditional_expression", "ternary_expression"}:
        for child in node.children:
            if child.type not in {"?", ":"}:
                return child

    return None


def _count_boolean_operator_complexity(node, code_bytes: bytes, language: str) -> int:
    condition_node = _get_condition_node(node, language)
    if condition_node is None:
        condition_node = node
    
    ops = _get_logical_operators(condition_node, code_bytes, language)

    if not ops:
        return 0

    score = 1

    for i in range(1, len(ops)):
        if ops[i] != ops[i - 1]:
            score += 1

    return score


def _is_else_if(node, language: str) -> bool:
    """Check if this if_statement is part of an else-if chain.

    In Java, tree-sitter makes the inner if_statement a direct child of the
    outer if_statement (via the 'alternative' field).  In C++, the inner
    if_statement is a child of an else_clause node.
    """
    if node.type != "if_statement":
        return False

    parent = node.parent
    if not parent:
        return False

    if language == "java":
        if parent.type == "if_statement":
            alt = parent.child_by_field_name("alternative")
            return alt is not None and alt.id == node.id
        return False
    elif language in {"cpp", "c++"}:
        return parent.type == "else_clause"

    return False

def _else_is_part_of_else_if(node, language: str) -> bool:
    """Check if this else/else_clause is followed by an if (else-if chain).

    When true, the else node itself should NOT get a +1 increment — the
    child if_statement (detected as else-if by _is_else_if) carries the +1.
    """
    if language == "java" and node.type == "else":
        parent = node.parent
        if parent and parent.type == "if_statement":
            alt = parent.child_by_field_name("alternative")
            return alt is not None and alt.type == "if_statement"
    elif language in {"cpp", "c++"} and node.type == "else_clause":
        for child in node.children:
            if child.type == "if_statement":
                return True
    return False


def _get_function_name(node, code_bytes: bytes, language: str) -> Optional[str]:
    if language == "java":
        for child in node.children:
            if child.type == "identifier":
                return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
    elif language in {"cpp", "c++"}:
        for child in node.children:
            if child.type == "function_declarator":
                for subchild in child.children:
                    if subchild.type in {"identifier", "field_identifier"}:
                        return code_bytes[subchild.start_byte:subchild.end_byte].decode("utf-8", errors="ignore")

    return None


def _is_labeled_jump(node, language: str) -> bool:
    if node.type not in JUMP_STATEMENTS.get(language, set()):
        return False
    for child in node.children:
        if child.type == "identifier":
            return True
    
    return False


def _is_recursive_call(node, current_function: Optional[str], code_bytes: bytes, language: str) -> bool:
    if not current_function:
        return False

    valid_call_types = {"call_expression", "method_invocation"}
    if node.type not in valid_call_types:
        return False

    if not node.children:
        return False

    if node.type == "method_invocation":
        for child in node.children:
            if child.type == "identifier":
                called_name = code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
                if called_name == current_function:
                    return True
        return False

    callee = node.children[0]

    if callee.type == "identifier":
        called_name = code_bytes[callee.start_byte:callee.end_byte].decode("utf-8", errors="ignore")
        return called_name == current_function

    if callee.type in {"field_expression", "member_expression"}:
        for child in callee.children:
            if child.type in {"identifier", "field_identifier"}:
                called_name = code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
                if called_name == current_function:
                    return True

    return False


def _sonar_style_cognitive_complexity(code: str, language: str) -> int:
    if not TREE_SITTER_AVAILABLE:
        logger.error("tree-sitter or languages not available")
        return 0

    try:
        if language in {"cpp", "c++", "c"}:
            lang_obj = Language(tscpp.language())
            language = "cpp"
        elif language == "java":
            lang_obj = Language(tsjava.language())
        else:
            logger.warning(f"Unsupported language for tree-sitter analysis: {language}")
            return 0
        parser = Parser(lang_obj)
    except Exception as e:
        logger.error(f"Failed to initialize parser for {language}: {e}")
        return 0
    
    if not isinstance(code, str):
        logger.error(f"Expected code to be str, got {type(code)}. Skipping cognitive complexity computation.")
        return 0.0
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    complexity = 0
    nesting = 0
    current_function = None

    def visit(node):
        nonlocal complexity, nesting, current_function

        if node.type in FUNCTION_NODES.get(language, set()):
            prev_function = current_function
            prev_nesting = nesting
            current_function = _get_function_name(node, code_bytes, language)
            nesting = 0

            for child in node.children:
                visit(child)

            current_function = prev_function
            nesting = prev_nesting
            return

        if node.type in NESTING_IMMUNE.get(language, set()):
            prev_nesting = nesting
            nesting = 0
            for child in node.children:
                visit(child)
            nesting = prev_nesting
            return

        if _is_recursive_call(node, current_function, code_bytes, language):
            complexity += 1

        if _is_labeled_jump(node, language):
            complexity += 1

        is_control_with_nesting = node.type in CONTROL_NODES_WITH_NESTING.get(language, set())
        is_control_no_nesting = node.type in CONTROL_NODES_NO_NESTING.get(language, set())
        is_jump_node = node.type in JUMP_NODES.get(language, set())
        has_condition = node.type in CONDITION_BEARING_NODES.get(language, set())

        if is_jump_node:
            complexity += 1 + nesting
            for child in node.children:
                visit(child)
        elif is_control_no_nesting:
            # Only +1 for a plain else; skip if this is part of an else-if chain
            # (the child if_statement will carry the +1 via _is_else_if).
            if not _else_is_part_of_else_if(node, language):
                complexity += 1
            for child in node.children:
                visit(child)
        elif is_control_with_nesting:
            is_else_if = _is_else_if(node, language)

            if is_else_if:
                complexity += 1
            else:
                complexity += 1 + nesting

            if has_condition:
                bool_complexity = _count_boolean_operator_complexity(node, code_bytes, language)
                complexity += bool_complexity

            if not is_else_if:
                nesting += 1

            for child in node.children:
                visit(child)

            if not is_else_if:
                nesting -= 1
        else:
            for child in node.children:
                visit(child)

    visit(root)
    return complexity


def cognitive_complexity(code: str, language: str) -> float:
    language = language.lower()

    if language in ["python", "python3"]:
        if not COGNITIVE_COMPLEXITY_AVAILABLE:
            logger.error("cognitive complexity not available")
            return 0.0

        try:
            tree = ast.parse(code)
            total_complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_complexity += get_cognitive_complexity(node)
            return total_complexity
        except Exception as e:
            print(f"  Warning: Python cognitive complexity failed: {e}")
            return 0.0

    if language in {"java", "cpp", "c++", "c"}:
        result = _sonar_style_cognitive_complexity(code, language)
        return float(result) if result is not None else 0.0

    logger.warning(f"Unsupported language: {language}")
    return 0.0


# =========================
# Results loading helpers
# =========================

def _derive_model_results_path(csv_path: str) -> str:
    """Infer model results path (outputs/{model}_results.jsonl) from the CSV path."""
    base = os.path.basename(csv_path)
    model = base.replace("_chi.csv", "").replace(".csv", "")
    return os.path.join(os.path.dirname(csv_path), "outputs", f"{model}_results.jsonl")


def _load_model_results(results_path: str) -> list:
    """Load lines from the model results JSONL file."""
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except Exception:
                continue
    return results


# =========================
# Main CHI computation
# =========================

def compute_chi(csv_path):
    """Calculate Code Hallucination Index using precomputed EditEval results.

    Args:
        csv_path: Path to CSV file with code samples (model name inferred from *_chi.csv)
    """

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    required_cols = ["language", "code", "test_code"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    N = len(df)
    if N == 0:
        raise ValueError("CSV file is empty")

    # Load model results JSONL for this CSV
    results_path = _derive_model_results_path(csv_path)
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Expected results file not found: {results_path}")
    model_results = _load_model_results(results_path)

    if len(model_results) < N:
        print(f"Warning: results file has fewer entries ({len(model_results)}) than CSV rows ({N}); missing rows will be treated as failures.")

    print(f"Processing {N} code samples...\n")

    # Track rates per sample (1 if sample had that category, 0 otherwise)
    pM_count = pN_count = pR_count = pL_count = 0
    syntax_error_count = time_error_count = unknown_error_count = 0

    time_per_assert_values = []
    contributing_memories = []
    cyc_vals = []
    cog_vals = []

    sample_records = []

    language_stats = {}
    overall_errors_dict = {}

    overall_total_problems = 0
    overall_passed_problems = 0
    overall_total_asserts = 0
    overall_passed_asserts = 0
    overall_tcpr_sum = 0.0  # Sum of per-sample assert pass rates (assert_passed/assert_total)
    failed_ids = []

    for sample_num, (idx, row) in enumerate(df.iterrows(), start=1):
        print(f"Processing sample {sample_num}/{N}...")

        language = row["language"].lower()
        code = row["code"]
        datapoint_id = row.get("task_id") or row.get("id") or idx
        test_code = row.get("test_code", "")

        # Initialize language stats
        if language not in language_stats:
            language_stats[language] = {
                'total_problems': 0,
                'passed_problems': 0,
                'total_asserts': 0,
                'passed_asserts': 0,
                'tcpr_sum': 0.0,
                'pM_count': 0,
                'pN_count': 0,
                'pR_count': 0,
                'pL_count': 0,
                'syntax_error_count': 0,
                'time_error_count': 0,
                'unknown_error_count': 0,
                'errors_dict': {},
            }

        # Pull matching result row (by index)
        result_row = model_results[idx] if idx < len(model_results) else {}

        # Extract assert stats and errors from results file
        assert_total = int(result_row.get("assert_total", 0) or 0)
        assert_passed = int(result_row.get("assert_passed", 0) or 0)
        passed_flag = bool(result_row.get("passed", False))
        error_list = result_row.get("error") or []
        if not isinstance(error_list, list):
            error_list = [str(error_list)]

        # Extract time and memory from results file (not from execution)
        raw_t = float(result_row.get("exec_time_s", 0.0) or 0.0)
        m = float(result_row.get("peak_mem_mb", 0.0) or 0.0)
        observed_assert_total = int(result_row.get("observed_assert_total", assert_total) or 0)

        # Time is normalized per observed assert; ignore sample time if observed asserts is 0 or time is 0.
        if raw_t > 0.0 and observed_assert_total > 0:
            t = raw_t / observed_assert_total
            time_per_assert_values.append(t)
            contributing_memories.append(m)
        else:
            t = None

        # Avoid zero totals: if nothing recorded, fall back to a single-assert view
        if assert_total <= 0:
            assert_total = 1
            assert_passed = 1 if passed_flag else 0

        pass_rate = assert_passed / assert_total if assert_total > 0 else 0.0
        overall_tcpr_sum += pass_rate

        # Build error dict: categorize each error in the list (count all occurrences, not unique)
        errors_dict = {}
        if error_list:
            for err in error_list:
                err_msg = str(err)
                category = categorize_error(err_msg, language)
                errors_dict[category] = errors_dict.get(category, 0) + 1
        else:
            # If some asserts failed but no messages, attribute to unknown
            if assert_passed < assert_total:
                errors_dict["Unknown"] = 1

        cyc_raw = cyclomatic_complexity(code, language)
        cog_raw = cognitive_complexity(code, language)

        cyc_vals.append(cyc_raw)
        cog_vals.append(cog_raw)

        # Update counts (problem-level and assert-level)
        language_stats[language]['total_problems'] += 1
        language_stats[language]['passed_problems'] += 1 if assert_passed == assert_total else 0
        language_stats[language]['total_asserts'] += assert_total
        language_stats[language]['passed_asserts'] += assert_passed
        language_stats[language]['tcpr_sum'] += pass_rate

        overall_total_problems += 1
        overall_passed_problems += 1 if assert_passed == assert_total else 0
        overall_total_asserts += assert_total
        overall_passed_asserts += assert_passed

        if assert_passed != assert_total:
            failed_ids.append(datapoint_id)

        if t is not None:
            print(f"  Resources: {raw_t:.3f}s raw, {t:.6f}s/assert, {m:.2f}MB")
        else:
            print(f"  Resources: {raw_t:.3f}s raw, time skipped (observed_assert_total={observed_assert_total}), {m:.2f}MB")
        print(f"  Logic: Assert pass rate {pass_rate:.2%}, Complexity (cyc={cyc_raw:.3f}, cog={cog_raw:.3f})")

        sample_records.append({
            "language": language,
            "time_per_assert": t,
            "memory": m,
            "cyc": cyc_raw,
            "cog": cog_raw,
            "pass_rate": pass_rate,
            "errors_dict": errors_dict,
        })

        # Categorize errors for this sample (per-problem presence)
        if errors_dict:
            num_errors = 1  # Count per problem, not per test case (matches chi_script.py)
            sample_breakdown = calculate_hallucination_breakdown(errors_dict, language, num_errors)
            sample_categories = calculate_category_rates(errors_dict, num_errors)

            # Problem-level category flags
            if sample_categories['Mapping_Hallucination']['count'] > 0:
                pM_count += 1
                language_stats[language]['pM_count'] += 1
            if sample_categories['Naming_Hallucination']['count'] > 0:
                pN_count += 1
                language_stats[language]['pN_count'] += 1
            if sample_categories['Resource_Hallucination']['count'] > 0:
                pR_count += 1
                language_stats[language]['pR_count'] += 1
            if sample_categories['Logical_Hallucination']['count'] > 0:
                pL_count += 1
                language_stats[language]['pL_count'] += 1

            if 'Syntax_Error' in sample_breakdown:
                syntax_error_count += 1
                language_stats[language]['syntax_error_count'] += 1
            if 'TimeError' in sample_breakdown:
                time_error_count += 1
                language_stats[language]['time_error_count'] += 1
            if 'Unknown' in sample_breakdown:
                unknown_error_count += 1
                language_stats[language]['unknown_error_count'] += 1

            # Aggregate per-problem presence into language/global stats (cap at 1 per category per problem)
            for halu_type in sample_breakdown:
                language_stats[language]['errors_dict'][halu_type] = language_stats[language]['errors_dict'].get(halu_type, 0) + 1
                overall_errors_dict[halu_type] = overall_errors_dict.get(halu_type, 0) + 1

        print()

    # --- Aggregate results ---
    has_valid_times = len(time_per_assert_values) > 0
    # Use 5th/95th percentile bounds so a single extreme outlier does not collapse
    # the entire normalization scale (min-max is outlier-sensitive by design).
    if has_valid_times:
        t_min = float(np.percentile(time_per_assert_values, 5))
        t_max = float(np.percentile(time_per_assert_values, 95))
        m_min = float(np.percentile(contributing_memories, 5))
        m_max = float(np.percentile(contributing_memories, 95))
    else:
        t_min = t_max = m_min = m_max = 0.0
    cyc_min, cyc_max = min(cyc_vals), max(cyc_vals)
    cog_min, cog_max = min(cog_vals), max(cog_vals)

    S_vals, Q_vals_passed, Q_vals_failed = [], [], []

    for t_val, m_val in zip(time_per_assert_values, contributing_memories):
        m_n = minmax_norm(m_val, m_min, m_max)
        t_n = minmax_norm(t_val, t_min, t_max)
        s_val = 0.5 * (t_n + m_n)
        S_vals.append(s_val)

    for rec in sample_records:
        cyc_n = minmax_norm(rec["cyc"], cyc_min, cyc_max)
        cog_n = minmax_norm(rec["cog"], cog_min, cog_max)

        q_val = logic_severity(cyc_n, cog_n, rec["pass_rate"])

        if rec["pass_rate"] == 1.0:
            Q_vals_passed.append(q_val)
        else:
            Q_vals_failed.append(q_val)

    # Calculate hallucination rates (problem-level)
    pM = pM_count / N if N > 0 else 0
    pN = pN_count / N if N > 0 else 0
    pR = pR_count / N if N > 0 else 0
    pL = pL_count / N if N > 0 else 0
    syntax_error_rate = syntax_error_count / N if N > 0 else 0
    time_error_rate = time_error_count / N if N > 0 else 0

    lambda_ = 1.0

    # Calculate final hallucination scores
    Hm = pM
    Hn = pN
    mean_s = float(np.mean(S_vals)) if len(S_vals) > 0 else 0.0
    HR = pR + lambda_ * mean_s if N > 0 else 0
    if len(Q_vals_failed) > 0:
        Q_bar_fail = np.mean(Q_vals_failed)
    else:
        Q_bar_fail = 0.0

    if len(Q_vals_passed) > 0:
        Q_bar_pass = np.mean(Q_vals_passed)
    else:
        Q_bar_pass = 0.0

    HL = (pL * Q_bar_fail + (1 - pL) * Q_bar_pass) if N > 0 else 0.0

    CHI = 0.25 * (Hm + Hn + HR + HL)

    # Language-level breakdowns (match chi_script_updated.py representation)
    language_hallucination_stats = {}
    for lang, stats in language_stats.items():
        total_probs = stats['total_problems']
        if total_probs > 0:
            tcpr = stats['tcpr_sum'] / total_probs
            language_hallucination_stats[lang] = {
                'total_problems': total_probs,
                'passed_problems': stats['passed_problems'],
                    'total_asserts': stats['total_asserts'],
                    'passed_asserts': stats['passed_asserts'],
                'pass_rate': round(stats['passed_problems'] / total_probs * 100, 2),
                'tcpr': round(tcpr * 100, 2),
                'pM': stats['pM_count'] / total_probs,
                'pN': stats['pN_count'] / total_probs,
                'pR': stats['pR_count'] / total_probs,
                'pL': stats['pL_count'] / total_probs,
                'syntax_error_count': stats['syntax_error_count'],
                'syntax_error_rate': stats['syntax_error_count'] / total_probs,
                'time_error_count': stats['time_error_count'],
                'time_error_rate': stats['time_error_count'] / total_probs,
                'unknown_error_count': stats['unknown_error_count'],
                'unknown_error_rate': stats['unknown_error_count'] / total_probs,
            }

    # Overall hallucination breakdown (across all languages)
    overall_hallucination_breakdown = {}
    if overall_total_problems > 0:
        for lang, stats in language_stats.items():
            breakdown = calculate_hallucination_breakdown(
                stats['errors_dict'], lang, stats['total_problems']
            )
            for halu_type, data in breakdown.items():
                if halu_type not in overall_hallucination_breakdown:
                    overall_hallucination_breakdown[halu_type] = {'count': 0, 'percentage': 0}
                overall_hallucination_breakdown[halu_type]['count'] += data['count']

        for halu_type in overall_hallucination_breakdown:
            count = overall_hallucination_breakdown[halu_type]['count']
            denom = overall_total_problems if overall_total_problems > 0 else 1
            overall_hallucination_breakdown[halu_type]['percentage'] = round(count / denom * 100, 2)

    overall_category_rates = calculate_category_rates(overall_errors_dict, overall_total_problems)
    overall_tcpr = overall_tcpr_sum / N if N > 0 else 0.0

    # Save failed IDs
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    failed_ids_file = f"{csv_name}_failed_ids.txt"
    with open(failed_ids_file, 'w', encoding='utf-8') as f:
        for failed_id in failed_ids:
            f.write(f"{failed_id}\n")
    print(f"\nSaved {len(failed_ids)} failed datapoint IDs to {failed_ids_file}")

    return {
        "H_mapping": Hm,
        "H_rate_mapping": pM,
        "H_naming": Hn,
        "H_rate_naming": pN,
        "H_resource": HR,
        "H_rate_resource": pR,
        "H_logic": HL,
        "H_rate_logic": pL,
        "syntax_error_count": syntax_error_count,
        "syntax_error_rate": syntax_error_rate,
        "time_error_count": time_error_count,
        "time_error_rate": time_error_rate,
        "unknown_error_count": unknown_error_count,
        "unknown_error_rate": unknown_error_count / N if N > 0 else 0,
        "CHI": CHI,
        "language_hallucination_stats": language_hallucination_stats,
        "overall_hallucination_breakdown": overall_hallucination_breakdown,
        "overall_category_rates": overall_category_rates,
        "overall_pass_rate": round(overall_passed_problems / overall_total_problems * 100, 2) if overall_total_problems > 0 else 0,
        # "overall_assert_pass_rate": round(overall_passed_asserts / overall_total_asserts * 100, 2) if overall_total_asserts > 0 else 0,
        "total_asserts": overall_total_asserts,
        "passed_asserts": overall_passed_asserts,
        "overall_tcpr": round(overall_tcpr * 100, 2),
        "failed_ids": failed_ids,
    }


# =========================
# CLI entry
# =========================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chi_updated_script.py data.csv")
        print("\nArguments:")
        print("  data.csv: Path to CSV file with code samples")
        print("\nExpected CSV columns:")
        print("  - language: 'python', 'cpp', or 'java'")
        print("  - code: source code string")
        print("  - test_code: assertion-based test code")
        print("\nNote: Q_vals computed using per-sample assert pass rates (assert_passed/assert_total)")
        sys.exit(1)

    try:
        results = compute_chi(sys.argv[1])
        print("\n" + "=" * 50)
        print("CHI RESULTS:")
        print("=" * 50)
        print(f"H_mapping (Hm): {results['H_mapping']:.4f}")
        print(f"  Rate (pM): {results['H_rate_mapping']:.4f} ({results['H_rate_mapping'] * 100:.2f}%)")
        print(f"H_naming (Hn): {results['H_naming']:.4f}")
        print(f"  Rate (pN): {results['H_rate_naming']:.4f} ({results['H_rate_naming'] * 100:.2f}%)")
        print(f"H_resource (HR): {results['H_resource']:.4f}")
        print(f"  Rate (pR): {results['H_rate_resource']:.4f} ({results['H_rate_resource'] * 100:.2f}%)")
        print(f"H_logic (HL): {results['H_logic']:.4f}")
        print(f"  Rate (pL): {results['H_rate_logic']:.4f} ({results['H_rate_logic'] * 100:.2f}%)")
        print(f"\nSyntax Errors: {results['syntax_error_count']} samples ({results['syntax_error_rate'] * 100:.2f}%)")
        print(f"Time Errors: {results['time_error_count']} samples ({results['time_error_rate'] * 100:.2f}%)")
        print(f"Unknown Errors: {results['unknown_error_count']} samples ({results['unknown_error_rate'] * 100:.2f}%)")
        print(f"\nCHI: {results['CHI']:.4f}")
        print(f"Overall Assert: {results['total_asserts']} total asserts, {results['passed_asserts']} passed asserts")

        print("\n" + "=" * 50)
        print("OVERALL HALLUCINATION BREAKDOWN (4 Main Categories):")
        print("=" * 50)
        print(f"Overall Pass Rate (All tests passed): {results['overall_pass_rate']:.2f}%")
        print(f"Overall TCPR (Average test case pass rate): {results['overall_tcpr']:.2f}%")
        print("\nMain Categories:")
        for category, data in sorted(results['overall_category_rates'].items()):
            print(f"  {category}: {data['count']} samples ({data['percentage']:.2f}%)")

        print("\n" + "=" * 50)
        print("LANGUAGE-WISE HALLUCINATION BREAKDOWN:")
        print("=" * 50)
        for lang, stats in sorted(results['language_hallucination_stats'].items()):
            print(f"\nLanguage: {lang.upper()}")
            print(f"  Total Problems: {stats['total_problems']}")
            print(f"  Passed Problems (all tests): {stats['passed_problems']}")
            print(f"  Total Asserts: {stats['total_asserts']}")
            print(f"  Passed Asserts: {stats['passed_asserts']}")
            print(f"  Pass Rate (all tests passed): {stats['pass_rate']:.2f}%")
            print(f"  TCPR (average test case pass rate): {stats['tcpr']:.2f}%")

            print(f"\n  Hallucination Rates:")
            print(f"    pM (Mapping): {stats['pM']:.4f} ({stats['pM'] * 100:.2f}%)")
            print(f"    pN (Naming): {stats['pN']:.4f} ({stats['pN'] * 100:.2f}%)")
            print(f"    pR (Resource): {stats['pR']:.4f} ({stats['pR'] * 100:.2f}%)")
            print(f"    pL (Logic): {stats['pL']:.4f} ({stats['pL'] * 100:.2f}%)")
            print(f"\n    Syntax Errors: {stats['syntax_error_count']} samples ({stats['syntax_error_rate'] * 100:.2f}%)")
            print(f"    Time Errors: {stats['time_error_count']} samples ({stats['time_error_rate'] * 100:.2f}%)")
            print(f"    Unknown Errors: {stats['unknown_error_count']} samples ({stats['unknown_error_rate'] * 100:.2f}%)")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
