import os
import json
import subprocess
import tempfile
import csv
import shutil
import re
from multiprocessing import Pool, cpu_count

RESULTS_DIR = "self_fix_results"
DATA_DIR = "data"

LANGUAGES = ["Python", "C++", "Java"]
SPLITS = ["competition", "interview"]

MAX_ATTEMPTS = 5
TIMEOUT = 5

programming_halus_cpp = {
    "Data_Compliance_Hallucination": ["Type mismatch", "Invalid conversion", "Division by zero"],
    "Structural_Access_Hallucination": ["Array index out of bounds", "Segmentation fault", "Out of range"],
    "Identification_Hallucination": ["Undeclared identifier", "Not declared in this scope", "Undefined reference"],
    "External_Source_Hallucination": ["No such file or directory", "Cannot find"],
    "Physical_Constraint_Hallucination": ["Stack overflow", "Memory exhausted"],
    "Calculate_Boundary_Hallucination": ["Overflow", "Arithmetic exception"],
    "Logic_Deviation": ["Logic_Deviation"],
    "Logic_Breakdown": ["Logic_Breakdown"],
}

programming_halus_python = {
    "Data_Compliance_Hallucination": ["TypeError", "ValueError", "ZeroDivisionError"],
    "Structural_Access_Hallucination": ["IndexError", "KeyError"],
    "Identification_Hallucination": ["NameError", "AttributeError", "UnboundLocalError"],
    "External_Source_Hallucination": ["ImportError", "ModuleNotFoundError"],
    "Physical_Constraint_Hallucination": ["RecursionError", "MemoryError"],
    "Calculate_Boundary_Hallucination": ["OverflowError", "StopIteration"],
    "Logic_Deviation": ["Logic_Deviation"],
    "Logic_Breakdown": ["Logic_Breakdown"],
}

programming_halus_java = {
    "Data_Compliance_Hallucination": ["incompatible types", "NumberFormatException", "ClassCastException"],
    "Structural_Access_Hallucination": ["ArrayIndexOutOfBoundsException", "IndexOutOfBoundsException", "NullPointerException"],
    "Identification_Hallucination": ["cannot find symbol", "package .* does not exist", "cannot be resolved"],
    "External_Source_Hallucination": ["ClassNotFoundException", "NoClassDefFoundError"],
    "Physical_Constraint_Hallucination": ["StackOverflowError", "OutOfMemoryError"],
    "Calculate_Boundary_Hallucination": ["ArithmeticException", "Overflow"],
    "Logic_Deviation": ["Logic_Deviation"],
    "Logic_Breakdown": ["Logic_Breakdown"],
}

TAXONOMY_MAP = {
    "Python": programming_halus_python,
    "C++": programming_halus_cpp,
    "Java": programming_halus_java,
}

ERROR_SUBCATS = [
    "Data_Compliance_Hallucination",
    "Structural_Access_Hallucination",
    "Identification_Hallucination",
    "External_Source_Hallucination",
    "Physical_Constraint_Hallucination",
    "Calculate_Boundary_Hallucination",
]
LOGIC_SUBCATS = ["Logic_Deviation", "Logic_Breakdown"]
ALL_SUBCATS = ERROR_SUBCATS + LOGIC_SUBCATS
MAIN_CATS = ["Mapping_Hallucination", "Naming_Hallucination", "Resource_Hallucination", "Logical_Hallucination"]


def subcat_to_main(subcat):
    if subcat in ["Data_Compliance_Hallucination", "Structural_Access_Hallucination"]:
        return "Mapping_Hallucination"
    if subcat in ["Identification_Hallucination", "External_Source_Hallucination"]:
        return "Naming_Hallucination"
    if subcat in ["Physical_Constraint_Hallucination", "Calculate_Boundary_Hallucination"]:
        return "Resource_Hallucination"
    if subcat in ["Logic_Deviation", "Logic_Breakdown"]:
        return "Logical_Hallucination"
    return None

class CodeExecutor:
    def __init__(self, language):
        self.language = language

    def clean_code(self, code):
        if not code:
            return ""
        match = re.search(r"```(?:\w+)?\s*\n(.*?)```", code, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
        indicators = {
            "Python": ["def ", "import ", "print("],
            "C++": ["#include", "int main", "std::"],
            "Java": ["public class", "System.out", "import java"],
        }
        check_list = indicators.get(self.language, [])
        if any(ind in code for ind in check_list):
            return code.strip()
        return code.strip()

    def prepare_file(self, code, temp_dir):
        code = self.clean_code(code)

        if self.language == "Python":
            file_path = os.path.join(temp_dir, "solution.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            return file_path

        elif self.language == "Java":
            match = re.search(r"public\s+class\s+(\w+)", code)
            if not match:
                match = re.search(
                    r"class\s+(\w+)\s*\{[^}]*public\s+static\s+void\s+main",
                    code, re.DOTALL
                )
            class_name = match.group(1) if match else "Main"
            file_path = os.path.join(temp_dir, f"{class_name}.java")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            return file_path

        else:
            raise ValueError(f"Unsupported language: {self.language}")


    def compile(self, file_path):
        if self.language == "Python":
            return True, ""

        if self.language == "C++":
            exe = os.path.join(os.path.dirname(file_path), "solution")
            compile_proc = subprocess.run(
                ["g++", file_path, "-O2", "-std=c++17", "-o", exe],
                capture_output=True, text=True,
            )
            if compile_proc.returncode != 0:
                return False, compile_proc.stderr
            return True, ""

        if self.language == "Java":
            compile_proc = subprocess.run(
                ["javac", file_path],
                capture_output=True, text=True,
            )
            if compile_proc.returncode != 0:
                return False, compile_proc.stderr
            return True, ""

        return False, "Unknown language"

    def run_test_case(self, file_path, input_str):
        if self.language == "Python":
            cmd = ["python3", file_path]

        elif self.language == "C++":
            cmd = [file_path.replace(".cpp", "")]

        elif self.language == "Java":
            wd = os.path.dirname(file_path)
            class_name = os.path.splitext(os.path.basename(file_path))[0]
            cmd = ["java", "-cp", wd, class_name]

        else:
            return "", f"Unsupported language: {self.language}", False

        try:
            process = subprocess.run(
                cmd,
                input=input_str,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                encoding="utf-8",
                errors="replace",
            )
            return process.stdout.strip(), process.stderr.strip(), False

        except subprocess.TimeoutExpired:
            return "", "Time Limit Exceeded", True

        except Exception as e:
            return "", str(e), False
















def match_error_subcats(err_text, lang):
    taxonomy = TAXONOMY_MAP.get(lang, {})
    matched = set()
    for subcat, patterns in taxonomy.items():
        if subcat in ("Logic_Deviation", "Logic_Breakdown"):
            continue 
        for p in patterns:
            if re.search(p, err_text, re.IGNORECASE):
                matched.add(subcat)
                break 
    return matched

def evaluate_attempt(code, lang, io_data):
    if not code.strip():
        return False, {"Logic_Breakdown"}

    executor = CodeExecutor(lang)
    temp_dir = tempfile.mkdtemp()
    try:

        try:
            file_path = executor.prepare_file(code, temp_dir)
        except ValueError:
            return False, {"Logic_Breakdown"}

        compiled, compile_err = executor.compile(file_path)

        halu_subcats = set()

        if compile_err:
            halu_subcats |= match_error_subcats(compile_err, lang)
            return False, halu_subcats


        inputs  = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])
        total   = len(inputs)

        if total == 0:
            return False, {"Logic_Breakdown"}

        passed_count        = 0
        wrong_no_error_count = 0

        for inp, expected in zip(inputs, outputs):
            actual, err, timed_out = executor.run_test_case(file_path, inp)

            if timed_out or err:
                err_text = err or "TIMEOUT"
                halu_subcats |= match_error_subcats(err_text, lang)

            elif stdout is not None:
                if actual == expected:
                    passed_count += 1
                elif actual.strip() == str(expected).strip():
                    passed_count += 1
                else:
                    wrong_no_error_count += 1

        pass_rate = passed_count / total if total > 0 else 0.0

        if pass_rate == 1.0:
            return True, set()

        if wrong_no_error_count > 0:
            if pass_rate == 0.0 and wrong_no_error_count == n:
                halu_subcats.add("Logic_Breakdown")
            elif pass_rate > 0.0 or (pass_rate == 0.0 and wrong_no_error_count < n):
                halu_subcats.add("Logic_Deviation")

        return False, halu_subcats

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def process_datapoint(job):
    model, split, lang, dp = job

    io_path = os.path.join(DATA_DIR, split, dp, "input_output.json")
    if not os.path.exists(io_path):
        return None
    try:
        with open(io_path, "r", encoding="utf-8", errors="replace") as f:
            io_data = json.load(f)
    except Exception:
        return None

    hist_path = os.path.join(RESULTS_DIR, model, split, lang, dp, "history.json")
    if not os.path.exists(hist_path):
        return None
    try:
        with open(hist_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception:
        return None

    attempts = data.get("history", [])
    pass_flags = [0] * MAX_ATTEMPTS
    subcat_flags = [{s: 0 for s in ALL_SUBCATS} for _ in range(MAX_ATTEMPTS)]

    for i, a in enumerate(attempts[:MAX_ATTEMPTS]):
        code = a.get("code", "")
        is_passed, halu_subcats = evaluate_attempt(code, lang, io_data)

        pass_flags[i] = 1 if is_passed else 0
        for s in halu_subcats:
            if s in subcat_flags[i]:
                subcat_flags[i][s] = 1

        triggered = sorted(halu_subcats) if halu_subcats else []
        status = "PASS" if is_passed else (f"HALU: {triggered}" if triggered else "FAIL (unmatched)")
        print(f"[{lang}] DP {dp} Attempt {i+1} â†’ {status}")

    return pass_flags, subcat_flags

def compute_model(model):
    print(f"\n{'='*60}")
    print(f"Starting evaluation for model: {model}")
    print(f"{'='*60}\n")

    total_hallu_rows = []
    hall_rows, pass_rows, fix_rows = [], [], []

    model_root = os.path.join(RESULTS_DIR, model)
    if not os.path.exists(model_root):
        print(f"[ERROR] Directory not found: {model_root}")
        return [], [], []

    for lang in LANGUAGES:
        print(f"\n--- Language: {lang} ---")

        jobs = []
        lang_variants = [lang]
        if lang == "C++":
            lang_variants += ["cpp", "c++"]
        elif lang == "Python":
            lang_variants += ["python"]
        elif lang == "Java":
            lang_variants += ["java"]

        for split in SPLITS:
            for variant in lang_variants:
                base = os.path.join(model_root, split, variant)
                if not os.path.isdir(base):
                    continue
                for dp in os.listdir(base):
                    if os.path.isdir(os.path.join(base, dp)):
                        jobs.append((model, split, variant, dp))
                break  

        if not jobs:
            print(f"No datapoints found for {lang}")
            continue

        print(f"Found {len(jobs)} datapoints")

        with Pool(cpu_count()) as pool:
            results = pool.map(process_datapoint, jobs)

        pass_counts = [0] * MAX_ATTEMPTS
        subcat_counts = [{s: 0 for s in ALL_SUBCATS} for _ in range(MAX_ATTEMPTS)]
        fix_success = 0
        total = 0

        for r in results:
            if r is None:
                continue
            total += 1
            p_f, s_f = r

            for i in range(MAX_ATTEMPTS):
                pass_counts[i] += p_f[i]
                for s in ALL_SUBCATS:
                    subcat_counts[i][s] += s_f[i][s]

            if p_f[0] == 0 and any(p_f[1:]):
                fix_success += 1

        if total == 0:
            print(f"No valid results for {lang}")
            continue

        print(f"\nCompleted {lang}: {total} datapoints evaluated")

        hall_row = {"Model": model, "Language": lang}
        pass_row = {"Model": model, "Language": lang}
        total_hallu_row = {"Model": model, "Language": lang}


        for i in range(MAX_ATTEMPTS):
            p_rate = round(pass_counts[i] / total, 3)
            pass_row[f"Attempt {i+1}"] = p_rate

            any_halu = sum(
                1 for r in results
                if r is not None and any(r[1][i][s] for s in ALL_SUBCATS)
            )
            hall_row[f"Attempt {i+1}"] = round(any_halu / total, 3)
            total_hallu_row[f"Attempt {i+1}"] = round(any_halu / total, 3)

            for s in ALL_SUBCATS:
                hall_row[f"Attempt {i+1}_{s}"] = round(subcat_counts[i][s] / total, 3)

            for main_cat in MAIN_CATS:
                subcats_under = [s for s in ALL_SUBCATS if subcat_to_main(s) == main_cat]
                triggered = sum(
                    1 for r in results
                    if r is not None and any(r[1][i][s] for s in subcats_under)
                )
                hall_row[f"Attempt {i+1}_{main_cat}"] = round(triggered / total, 3)

            print(f"  Attempt {i+1}: Pass={p_rate}  AnyHalu={hall_row[f'Attempt {i+1}']}")
            for s in ALL_SUBCATS:
                rate = hall_row[f"Attempt {i+1}_{s}"]
                if rate > 0:
                    print(f"    {s}: {rate}")

        fix_rate = round(fix_success / total, 3)
        print(f"  Self-Fix Success Rate: {fix_rate}")

        hall_rows.append(hall_row)
        pass_rows.append(pass_row)
        fix_rows.append({"Model": model, "Language": lang, "SelfFixSuccess": fix_rate})
        total_hallu_rows.append(total_hallu_row)

    return hall_rows, pass_rows, fix_rows, total_hallu_rows

def write_csv(filename, rows):
    if not rows:
        print(f"[INFO] No rows to write for {filename}")
        return
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Written to {filename}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        if not os.path.exists(RESULTS_DIR):
            print(f"Usage: python compute_metrics_final.py <model_name>")
            print(f"       or place models under {RESULTS_DIR}/ and run without args")
            sys.exit(1)
        target_models = [
            d for d in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, d))
        ]
        print(f"No model specified. Found models: {target_models}")
    else:
        target_models = sys.argv[1:]


    for model in target_models:
        hall_rows, pass_rows, fix_rows, total_hallu_rows = compute_model(model)
        write_csv("hallucination_rates.csv", hall_rows)
        write_csv("pass_rates.csv", pass_rows)
        write_csv("self_fix_success.csv", fix_rows)
        write_csv("total_hallucination_rate.csv", total_hallu_rows)

    print("\nDone.")
