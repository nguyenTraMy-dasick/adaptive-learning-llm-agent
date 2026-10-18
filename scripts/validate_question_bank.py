"""Script kiểm tra và xác thực tính toàn vẹn của Question Bank và Concept Graph.

Sử dụng:
    python scripts/validate_question_bank.py [--strict] [--csv PATH] [--graph PATH]

Kiểm tra:
1. Cấu trúc và tính hợp lệ của đồ thị concept (DAG, không chu trình, liên kết 2 chiều).
2. Định dạng header CSV (khớp đúng tên cột chuẩn: prerequisite_concept số ít,...).
3. Tính toàn vẹn của từng câu hỏi:
   - question_id duy nhất, không rỗng
   - concept thuộc đồ thị concept
   - prerequisite_concept khớp với concept graph hoặc alias được công nhận
   - difficulty nằm trong {Easy, Medium, Hard}
   - correct_answer nằm trong {A, B, C, D} (không dính ký tự trắng/tab như \\tA)
   - 4 phương án option_A, B, C, D không rỗng
   - common_error_type, option_error_mapping, explanation đầy đủ
4. Độ phủ (coverage) câu hỏi cho từng concept và mức độ khó.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Thêm thư mục src vào sys.path để import adaptive_learning nếu cần
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXPECTED_CSV_COLUMNS = [
    "question_id",
    "concept",
    "prerequisite_concept",
    "question",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "difficulty",
    "correct_answer",
    "common_error_type",
    "option_error_mapping",
    "explanation",
]

ALLOWED_DIFFICULTIES = {"Easy", "Medium", "Hard"}
ALLOWED_ANSWERS = {"A", "B", "C", "D"}


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict[str, Any] = {}

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_concept_graph(graph_path: Path, report: ValidationReport) -> dict[str, Any] | None:
    """Kiểm tra tính hợp lệ của file concept_graph.json."""
    if not graph_path.exists():
        report.add_error(f"Concept graph file not found: {graph_path}")
        return None

    try:
        with open(graph_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        report.add_error(f"Concept graph JSON syntax error in {graph_path}: {exc}")
        return None

    concepts = data.get("concepts")
    if not isinstance(concepts, dict) or not concepts:
        report.add_error(f"{graph_path}: Key 'concepts' must be a non-empty dictionary.")
        return None

    report.stats["graph_concept_count"] = len(concepts)

    # Kiểm tra liên kết và định hướng
    for c_name, c_data in concepts.items():
        if not isinstance(c_data, dict):
            report.add_error(f"Concept '{c_name}' metadata must be a dictionary.")
            continue

        prereqs = c_data.get("prerequisites", [])
        for p in prereqs:
            if p not in concepts:
                report.add_error(f"Concept '{c_name}' has unrecognized prerequisite: '{p}'")

        dependents = c_data.get("dependents", [])
        for d in dependents:
            if d not in concepts:
                report.add_error(f"Concept '{c_name}' has unrecognized dependent: '{d}'")

    # Kiểm tra chu trình (Cycle detection)
    visited: dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited

    def has_cycle(u: str) -> bool:
        visited[u] = 1
        deps = concepts[u].get("dependents", []) if isinstance(concepts.get(u), dict) else []
        for v in deps:
            if v not in concepts:
                continue
            if visited.get(v, 0) == 1:
                return True
            if visited.get(v, 0) == 0 and has_cycle(v):
                return True
        visited[u] = 2
        return False

    for c_name in concepts:
        if visited.get(c_name, 0) == 0:
            if has_cycle(c_name):
                report.add_error("Concept graph contains a cycle! It must be a Directed Acyclic Graph (DAG).")
                break

    # Kiểm tra edges nếu có
    edges = data.get("edges", [])
    report.stats["graph_edge_count"] = len(edges)
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in concepts:
            report.add_error(f"Edge source '{src}' does not exist in concepts dictionary.")
        if tgt not in concepts:
            report.add_error(f"Edge target '{tgt}' does not exist in concepts dictionary.")

    return data


def validate_question_bank_csv(
    csv_path: Path,
    graph_data: dict[str, Any] | None,
    report: ValidationReport,
) -> list[dict[str, str]]:
    """Kiểm tra chi tiết từng dòng trong file CSV của question bank."""
    if not csv_path.exists():
        report.add_error(f"Question bank CSV not found: {csv_path}")
        return []

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    concept_nodes = graph_data.get("concepts", {}) if graph_data else {}
    alias_map: dict[str, str] = {}
    if concept_nodes:
        for c_name, c_info in concept_nodes.items():
            alias_map[c_name.lower()] = c_name
            if isinstance(c_info, dict):
                c_id = c_info.get("id")
                if c_id:
                    alias_map[c_id.strip().lower()] = c_name
                for a in c_info.get("aliases", []):
                    if a and isinstance(a, str):
                        alias_map[a.strip().lower()] = c_name

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # 1. Kiểm tra cột CSV
        missing_cols = [c for c in EXPECTED_CSV_COLUMNS if c not in fieldnames]
        if missing_cols:
            report.add_error(f"CSV header is missing required columns: {missing_cols}")
        extra_cols = [c for c in fieldnames if c not in EXPECTED_CSV_COLUMNS]
        if extra_cols:
            report.add_warning(f"CSV header has unexpected columns: {extra_cols}")

        for line_no, raw_row in enumerate(reader, start=2):
            rows.append(raw_row)
            qid = raw_row.get("question_id", "")
            raw_qid = qid

            # Kiểm tra question_id rỗng hoặc duplicate
            clean_qid = qid.strip()
            if not clean_qid:
                report.add_error(f"Line {line_no}: Empty question_id.")
                continue
            if clean_qid in seen_ids:
                report.add_error(f"Line {line_no} ({clean_qid}): Duplicate question_id.")
            seen_ids.add(clean_qid)

            # Kiểm tra khoảng trắng thừa trong question_id
            if raw_qid != clean_qid:
                report.add_warning(f"Line {line_no} ({clean_qid}): question_id has leading/trailing whitespace.")

            # Kiểm tra concept
            raw_concept = raw_row.get("concept", "")
            clean_concept = raw_concept.strip()
            if not clean_concept:
                report.add_error(f"Line {line_no} ({clean_qid}): Missing 'concept'.")
            elif concept_nodes:
                if clean_concept in concept_nodes:
                    pass  # Khớp chính xác
                elif clean_concept.lower() in alias_map:
                    canonical = alias_map[clean_concept.lower()]
                    report.add_warning(
                        f"Line {line_no} ({clean_qid}): Concept '{clean_concept}' matches alias for canonical '{canonical}'."
                    )
                else:
                    report.add_error(
                        f"Line {line_no} ({clean_qid}): Concept '{clean_concept}' not found in concept graph."
                    )

            # Kiểm tra prerequisite_concept
            raw_prereqs = raw_row.get("prerequisite_concept", "")
            clean_prereqs = raw_prereqs.strip()
            if clean_prereqs and clean_prereqs.lower() != "none":
                items = [p.strip() for p in clean_prereqs.split(";") if p.strip()]
                for item in items:
                    if item.lower() == "none":
                        continue
                    if concept_nodes:
                        if item in concept_nodes:
                            pass
                        elif item.lower() in alias_map:
                            report.add_warning(
                                f"Line {line_no} ({clean_qid}): Prerequisite '{item}' is an alias/variant of '{alias_map[item.lower()]}'."
                            )
                        else:
                            report.add_error(
                                f"Line {line_no} ({clean_qid}): Prerequisite concept '{item}' not found in concept graph."
                            )

            # Kiểm tra difficulty
            raw_diff = raw_row.get("difficulty", "")
            clean_diff = raw_diff.strip()
            if clean_diff not in ALLOWED_DIFFICULTIES:
                report.add_error(
                    f"Line {line_no} ({clean_qid}): Invalid difficulty {raw_diff!r}. Allowed: {sorted(ALLOWED_DIFFICULTIES)}"
                )
            elif raw_diff != clean_diff:
                report.add_warning(f"Line {line_no} ({clean_qid}): Difficulty has surrounding whitespace: {raw_diff!r}")

            # Kiểm tra correct_answer
            raw_ans = raw_row.get("correct_answer", "")
            clean_ans = raw_ans.strip().upper()
            if clean_ans not in ALLOWED_ANSWERS:
                report.add_error(
                    f"Line {line_no} ({clean_qid}): Invalid correct_answer {raw_ans!r}. Allowed: {sorted(ALLOWED_ANSWERS)}"
                )
            elif raw_ans != clean_ans:
                report.add_warning(
                    f"Line {line_no} ({clean_qid}): correct_answer has whitespace or tab character: {repr(raw_ans)} -> Suggestion: '{clean_ans}'"
                )

            # Kiểm tra question text và 4 options
            q_text = raw_row.get("question", "").strip()
            if not q_text:
                report.add_error(f"Line {line_no} ({clean_qid}): 'question' text is empty.")

            for opt_key in ["option_A", "option_B", "option_C", "option_D"]:
                opt_val = raw_row.get(opt_key, "").strip()
                if not opt_val:
                    report.add_error(f"Line {line_no} ({clean_qid}): Empty {opt_key}.")

            # Kiểm tra metadata giải thích
            explanation = raw_row.get("explanation", "").strip()
            if not explanation:
                report.add_warning(f"Line {line_no} ({clean_qid}): Missing explanation.")

            err_mapping = raw_row.get("option_error_mapping", "").strip()
            if not err_mapping:
                report.add_warning(f"Line {line_no} ({clean_qid}): Missing option_error_mapping.")

    report.stats["total_questions"] = len(rows)
    return rows


def check_coverage(rows: list[dict[str, str]], graph_data: dict[str, Any] | None, report: ValidationReport) -> None:
    """Kiểm tra độ phủ câu hỏi cho từng concept và mức độ khó."""
    counts = defaultdict(lambda: defaultdict(int))
    concept_nodes = graph_data.get("concepts", {}) if graph_data else {}

    for row in rows:
        concept = row.get("concept", "").strip()
        difficulty = row.get("difficulty", "").strip()
        counts[concept][difficulty] += 1
        counts[concept]["Total"] += 1

    report.stats["concept_distribution"] = {c: dict(d) for c, d in counts.items()}

    # Kiểm tra concept trong graph có câu hỏi tương ứng không
    if concept_nodes:
        for c_name in concept_nodes:
            if c_name not in counts:
                report.add_error(f"Concept '{c_name}' in graph has NO questions in question bank.")
            else:
                for diff in ["Easy", "Medium", "Hard"]:
                    if counts[c_name].get(diff, 0) == 0:
                        report.add_warning(f"Concept '{c_name}' has 0 questions for difficulty '{diff}'.")


def print_report(report: ValidationReport, strict: bool = False) -> None:
    """Hiển thị báo cáo kết quả kiểm tra định dạng đẹp."""
    print("=" * 70)
    print("           BÁO CÁO XÁC THỰC QUESTION BANK & CONCEPT GRAPH")
    print("=" * 70)

    stats = report.stats
    print(f"Tổng số concept trong graph : {stats.get('graph_concept_count', 'N/A')}")
    print(f"Tổng số cạnh (edges) trong graph: {stats.get('graph_edge_count', 'N/A')}")
    print(f"Tổng số câu hỏi trong CSV   : {stats.get('total_questions', 'N/A')}")
    print("-" * 70)

    # In phân bố độ khó theo concept
    distribution = stats.get("concept_distribution", {})
    if distribution:
        print(f"{'Concept':<28} | {'Easy':<5} | {'Medium':<6} | {'Hard':<5} | {'Total':<5}")
        print("-" * 58)
        for c, d in sorted(distribution.items()):
            print(f"{c:<28} | {d.get('Easy', 0):<5} | {d.get('Medium', 0):<6} | {d.get('Hard', 0):<5} | {d.get('Total', 0):<5}")
        print("-" * 58)

    # In Warnings
    if report.warnings:
        print(f"\n[CẢNH BÁO / DỮ LIỆU CẦN LƯU Ý] ({len(report.warnings)} cảnh báo):")
        for i, w in enumerate(report.warnings, 1):
            print(f"  {i}. {w}")
    else:
        print("\nKhông có cảnh báo dữ liệu nào.")

    # In Errors
    if report.errors:
        print(f"\n[LỖI NGHIÊM TRỌNG] ({len(report.errors)} lỗi):")
        for i, e in enumerate(report.errors, 1):
            print(f"  {i}. {e}")
    else:
        print("\nKhông có lỗi nghiêm trọng nào.")

    print("=" * 70)
    if report.is_valid and (not strict or not report.warnings):
        print("KẾT LUẬN: HỢP ĐỒNG DỮ LIỆU ĐẠT CHUẨN (PASSED)")
    else:
        print("KẾT LUẬN: PHÁT HIỆN LỖI/CẢNH BÁO CẦN XỬ LÝ (FAILED)")
    print("=" * 70)


def fix_question_bank_csv(csv_path: Path, graph_data: dict[str, Any] | None) -> int:
    """Tự động sửa các lỗi whitespace và chuẩn hóa concept/prerequisite theo concept graph."""
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 0

    concept_nodes = graph_data.get("concepts", {}) if graph_data else {}
    alias_map: dict[str, str] = {}
    if concept_nodes:
        for c_name, c_info in concept_nodes.items():
            alias_map[c_name.lower()] = c_name
            if isinstance(c_info, dict):
                c_id = c_info.get("id")
                if c_id:
                    alias_map[c_id.strip().lower()] = c_name
                for a in c_info.get("aliases", []):
                    if a and isinstance(a, str):
                        alias_map[a.strip().lower()] = c_name

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or EXPECTED_CSV_COLUMNS
        rows = list(reader)

    fixed_count = 0
    for row in rows:
        # 1. Fix correct_answer
        raw_ans = row.get("correct_answer", "")
        clean_ans = raw_ans.strip().upper()
        if raw_ans != clean_ans:
            row["correct_answer"] = clean_ans
            fixed_count += 1

        # 2. Fix difficulty
        raw_diff = row.get("difficulty", "")
        clean_diff = raw_diff.strip()
        if raw_diff != clean_diff:
            row["difficulty"] = clean_diff
            fixed_count += 1

        # 3. Canonicalize concept if needed
        raw_concept = row.get("concept", "")
        clean_concept = raw_concept.strip()
        if clean_concept.lower() in alias_map:
            canon = alias_map[clean_concept.lower()]
            if raw_concept != canon:
                row["concept"] = canon
                fixed_count += 1

        # 4. Canonicalize prerequisite_concept
        raw_prereq = row.get("prerequisite_concept", "")
        clean_prereq = raw_prereq.strip()
        if clean_prereq and clean_prereq.lower() != "none":
            items = [p.strip() for p in clean_prereq.split(";") if p.strip()]
            fixed_items = []
            for item in items:
                if item.lower() in alias_map:
                    fixed_items.append(alias_map[item.lower()])
                else:
                    fixed_items.append(item)
            new_prereq = "; ".join(fixed_items)
            if new_prereq != raw_prereq:
                row["prerequisite_concept"] = new_prereq
                fixed_count += 1

    if fixed_count > 0:
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[AUTO-FIX] Đã chuẩn hóa {fixed_count} điểm dữ liệu trong {csv_path}.")
    else:
        print("[AUTO-FIX] Không có điểm dữ liệu nào cần sửa.")
    return fixed_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Xác thực Question Bank và Concept Graph.")
    parser.add_argument("--csv", type=str, default=str(PROJECT_ROOT / "data" / "question_bank.csv"), help="Đường dẫn file CSV")
    parser.add_argument("--graph", type=str, default=str(PROJECT_ROOT / "data" / "concept_graph.json"), help="Đường dẫn file concept graph JSON")
    parser.add_argument("--strict", action="store_true", help="Chế độ nghiêm ngặt: coi warning là lỗi.")
    parser.add_argument("--fix", action="store_true", help="Tự động sửa các lỗi whitespace và chuẩn hóa alias trong CSV.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    graph_path = Path(args.graph)

    # 1. Tự động sửa nếu có cờ --fix
    if args.fix:
        try:
            with open(graph_path, mode="r", encoding="utf-8") as f:
                g_data = json.load(f)
        except Exception:
            g_data = None
        fix_question_bank_csv(csv_path, g_data)

    report = ValidationReport()

    # 2. Kiểm tra concept graph
    graph_data = validate_concept_graph(graph_path, report)

    # 3. Kiểm tra CSV
    rows = validate_question_bank_csv(csv_path, graph_data, report)

    # 4. Kiểm tra coverage
    if rows and graph_data:
        check_coverage(rows, graph_data, report)

    # 5. In báo cáo
    print_report(report, strict=args.strict)

    if not report.is_valid:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
