"""Thành viên 1: Quản lý Question Bank và Concept Graph.

Cung cấp:
- ConceptGraph: Đọc, tra cứu và xác thực đồ thị tiên quyết (prerequisite DAG).
- QuestionBank: Đọc câu hỏi từ data/question_bank.csv cố định, hỗ trợ lọc câu hỏi
  theo concept, difficulty và loại bỏ các câu hỏi đã làm (attempted_question_ids).
"""
from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from adaptive_learning.schemas import (
    ALLOWED_DIFFICULTIES,
    Difficulty,
    Question,
    _validate_difficulty,
)

logger = logging.getLogger(__name__)

# Đặt đường dẫn mặc định tới thư mục data/ của dự án
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV_PATH = _PROJECT_ROOT / "data" / "question_bank.csv"
DEFAULT_GRAPH_PATH = _PROJECT_ROOT / "data" / "concept_graph.json"


class ConceptGraph:
    """Đồ thị tiên quyết (DAG) của các concept học phần."""

    def __init__(
        self,
        concepts: Mapping[str, Mapping[str, Any]] | None = None,
        edges: Sequence[Mapping[str, str]] | None = None,
        name: str = "Concept Graph",
        version: str = "1.0.0",
        description: str = "",
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self._concepts: dict[str, dict[str, Any]] = {}
        self._alias_map: dict[str, str] = {}
        self._edges: list[dict[str, str]] = []

        if concepts:
            for c_name, c_info in concepts.items():
                self.add_concept(c_name, c_info)
        if edges:
            for edge in edges:
                self.add_edge(edge["source"], edge["target"])

    def add_concept(self, name: str, info: Mapping[str, Any]) -> None:
        """Thêm một concept cùng metadata và aliases vào graph."""
        c_name = name.strip()
        c_dict = dict(info)
        c_dict["name"] = c_name
        c_dict.setdefault("prerequisites", [])
        c_dict.setdefault("dependents", [])
        c_dict.setdefault("aliases", [])

        self._concepts[c_name] = c_dict
        # Đăng ký chính tên canonical vào alias map
        self._alias_map[c_name.lower()] = c_name

        # Đăng ký id nếu có
        c_id = c_dict.get("id")
        if c_id:
            self._alias_map[c_id.strip().lower()] = c_name

        # Đăng ký các alias
        for alias in c_dict.get("aliases", []):
            if alias and isinstance(alias, str):
                self._alias_map[alias.strip().lower()] = c_name

    def add_edge(self, source: str, target: str) -> None:
        """Thêm cạnh định hướng: source -> target (source là tiên quyết của target)."""
        src = self.canonicalize(source)
        tgt = self.canonicalize(target)
        if src not in self._concepts:
            raise KeyError(f"Source concept '{source}' (canonical: '{src}') not found in graph.")
        if tgt not in self._concepts:
            raise KeyError(f"Target concept '{target}' (canonical: '{tgt}') not found in graph.")

        if tgt not in self._concepts[src]["dependents"]:
            self._concepts[src]["dependents"].append(tgt)
        if src not in self._concepts[tgt]["prerequisites"]:
            self._concepts[tgt]["prerequisites"].append(src)

        self._edges.append({"source": src, "target": tgt})

    def canonicalize(self, name: str) -> str:
        """Chuẩn hóa tên concept về canonical name nếu có alias hoặc khớp không phân biệt hoa thường."""
        clean = name.strip()
        lower = clean.lower()
        if lower in self._alias_map:
            return self._alias_map[lower]
        return clean

    def has_concept(self, name: str) -> bool:
        """Kiểm tra concept (hoặc alias) có tồn tại trong graph không."""
        return self.canonicalize(name) in self._concepts

    def get_concept(self, name: str) -> dict[str, Any] | None:
        """Lấy thông tin concept theo tên canonical hoặc alias."""
        canon = self.canonicalize(name)
        return self._concepts.get(canon)

    def get_prerequisites(self, name: str) -> tuple[str, ...]:
        """Lấy danh sách các concept tiên quyết trực tiếp của một concept."""
        info = self.get_concept(name)
        if not info:
            return ()
        return tuple(info.get("prerequisites", []))

    def get_dependents(self, name: str) -> tuple[str, ...]:
        """Lấy danh sách các concept kế tiếp (phụ thuộc) trực tiếp của một concept."""
        info = self.get_concept(name)
        if not info:
            return ()
        return tuple(info.get("dependents", []))

    def get_ancestors(self, name: str) -> set[str]:
        """Lấy toàn bộ các concept tiên quyết gián tiếp và trực tiếp (transitive prerequisites)."""
        visited: set[str] = set()

        def dfs(curr: str) -> None:
            for p in self.get_prerequisites(curr):
                if p not in visited:
                    visited.add(p)
                    dfs(p)

        canon = self.canonicalize(name)
        dfs(canon)
        return visited

    def is_dag(self) -> bool:
        """Kiểm tra đồ thị có chu trình hay không (phải là DAG)."""
        visited: dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited

        def has_cycle(u: str) -> bool:
            visited[u] = 1
            for v in self.get_dependents(u):
                if visited.get(v, 0) == 1:
                    return True
                if visited.get(v, 0) == 0:
                    if has_cycle(v):
                        return True
            visited[u] = 2
            return False

        for node in self._concepts:
            if visited.get(node, 0) == 0:
                if has_cycle(node):
                    return False
        return True

    def topological_sort(self) -> list[str]:
        """Trả về danh sách concept theo thứ tự topo (tiên quyết đứng trước)."""
        in_degree: dict[str, int] = {c: 0 for c in self._concepts}
        for c in self._concepts:
            for dep in self.get_dependents(c):
                in_degree[dep] += 1

        queue = [c for c, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for dep in self.get_dependents(curr):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(order) != len(self._concepts):
            raise ValueError("Graph contains a cycle; topological sort not possible.")
        return order

    def all_concepts(self) -> tuple[str, ...]:
        """Trả về tuple tất cả canonical concept names."""
        return tuple(self._concepts.keys())

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> ConceptGraph:
        """Đọc đồ thị concept từ file JSON."""
        resolved_path = Path(path) if path else DEFAULT_GRAPH_PATH
        if not resolved_path.exists():
            raise FileNotFoundError(f"Concept graph file not found at: {resolved_path}")

        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            concepts=data.get("concepts", {}),
            edges=data.get("edges", []),
            name=data.get("name", "Concept Graph"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Xuất graph ra dict."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "concepts": self._concepts,
            "edges": self._edges,
        }


class QuestionBank:
    """Nguồn câu hỏi trắc nghiệm cố định cho hệ thống Adaptive Learning."""

    def __init__(
        self,
        csv_path: str | Path | None = None,
        concept_graph: ConceptGraph | str | Path | None = None,
        auto_load: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        if isinstance(concept_graph, (str, Path)):
            self.concept_graph = ConceptGraph.from_file(concept_graph)
        elif isinstance(concept_graph, ConceptGraph):
            self.concept_graph = concept_graph
        elif DEFAULT_GRAPH_PATH.exists():
            try:
                self.concept_graph = ConceptGraph.from_file(DEFAULT_GRAPH_PATH)
            except Exception as exc:  # graph hỏng thì vẫn đọc được CSV nhưng phải báo rõ
                logger.warning(
                    "Không đọc được concept graph %s (%s); concept/prerequisite sẽ không được chuẩn hóa.",
                    DEFAULT_GRAPH_PATH, exc,
                )
                self.concept_graph = None
        else:
            self.concept_graph = None

        self._questions: dict[str, Question] = {}
        self._by_concept: dict[str, list[Question]] = defaultdict(list)
        self._by_concept_difficulty: dict[tuple[str, str], list[Question]] = defaultdict(list)
        self._by_difficulty: dict[str, list[Question]] = defaultdict(list)

        if auto_load:
            self.load()

    def _canonical(self, name: str) -> str:
        return self.concept_graph.canonicalize(name) if self.concept_graph else name.strip()

    def _canonical_prerequisites(self, raw: str) -> str:
        """Chuẩn hóa chuỗi prerequisite ('A; B') về canonical name; 'None' nếu rỗng."""
        items = [p.strip() for p in (raw or "").split(";")]
        items = [self._canonical(p) for p in items if p and p.lower() not in {"none", "null"}]
        return "; ".join(items) if items else "None"

    def load(self) -> None:
        """Đọc toàn bộ câu hỏi từ CSV vào bộ nhớ và xây dựng chỉ mục tra cứu."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Question bank CSV not found at: {self.csv_path}")

        self._questions.clear()
        self._by_concept.clear()
        self._by_concept_difficulty.clear()
        self._by_difficulty.clear()

        # newline="" là bắt buộc với csv khi ô có xuống dòng (code nhiều dòng trong câu hỏi).
        with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV file at {self.csv_path} is empty or has no header.")

            for line_no, row in enumerate(reader, start=2):
                qid = row.get("question_id", "").strip()
                if not qid:
                    logger.warning("Bỏ qua hàng %d vì thiếu question_id", line_no)
                    continue
                if qid in self._questions:
                    raise ValueError(f"Hàng {line_no}: question_id trùng lặp {qid!r}.")

                raw_concept = row.get("concept", "").strip()
                # Chuẩn hóa tên concept qua concept_graph nếu có
                concept = self._canonical(raw_concept)
                if self.concept_graph and not self.concept_graph.has_concept(concept):
                    logger.warning("Hàng %d (%s): concept %r không có trong concept graph.", line_no, qid, concept)

                # Làm sạch giá trị correct_answer và difficulty
                correct_ans = row.get("correct_answer", "").strip().upper()
                difficulty = _validate_difficulty(row.get("difficulty", "").strip())
                if difficulty is None:
                    raise ValueError(f"Hàng {line_no} ({qid}): thiếu hoặc sai difficulty {row.get('difficulty')!r}")

                question = Question(
                    question_id=qid,
                    concept=concept,
                    prerequisite_concept=self._canonical_prerequisites(row.get("prerequisite_concept", "None")),
                    question=row.get("question", "").strip(),
                    option_A=row.get("option_A", "").strip(),
                    option_B=row.get("option_B", "").strip(),
                    option_C=row.get("option_C", "").strip(),
                    option_D=row.get("option_D", "").strip(),
                    difficulty=difficulty,
                    correct_answer=correct_ans,
                    common_error_type=row.get("common_error_type", "").strip(),
                    option_error_mapping=row.get("option_error_mapping", "").strip(),
                    explanation=row.get("explanation", "").strip(),
                )

                self._questions[qid] = question
                self._by_concept[concept].append(question)
                self._by_concept_difficulty[(concept, difficulty)].append(question)
                self._by_difficulty[difficulty].append(question)

    def get_question(self, question_id: str) -> Question | None:
        """Tra cứu câu hỏi theo question_id."""
        return self._questions.get(question_id.strip())

    def has_question(self, question_id: str) -> bool:
        """Kiểm tra question_id có trong bank không."""
        return question_id.strip() in self._questions

    def has_concept(self, concept: str) -> bool:
        """Kiểm tra concept có câu hỏi nào trong bank không."""
        canon = self.concept_graph.canonicalize(concept) if self.concept_graph else concept.strip()
        return canon in self._by_concept

    def get_all_questions(self) -> list[Question]:
        """Trả về toàn bộ danh sách câu hỏi trong bank."""
        return list(self._questions.values())

    def get_all_concepts(self) -> list[str]:
        """Trả về danh sách tất cả concepts có trong question bank theo thứ tự topo hoặc sắp xếp."""
        if self.concept_graph:
            try:
                topo = self.concept_graph.topological_sort()
                return [c for c in topo if c in self._by_concept]
            except Exception:
                pass
        return sorted(self._by_concept.keys())

    def get_questions(
        self,
        concept: str | None = None,
        difficulty: str | Difficulty | None = None,
        exclude_ids: Iterable[str] | None = None,
    ) -> list[Question]:
        """Lấy danh sách câu hỏi thỏa mãn điều kiện concept/difficulty và loại trừ exclude_ids.

        Args:
            concept: Tên concept (hoặc alias). Nếu None, lấy tất cả concepts.
            difficulty: 'Easy', 'Medium', 'Hard' hoặc Difficulty enum. Nếu None, lấy mọi độ khó.
            exclude_ids: Tập question_id đã làm cần bỏ qua.

        Returns:
            Danh sách Question thỏa mãn điều kiện.
        """
        excluded = {str(x).strip() for x in exclude_ids} if exclude_ids else set()

        # Chuẩn hóa difficulty
        target_diff = _validate_difficulty(difficulty) if difficulty is not None else None

        # Chuẩn hóa concept
        target_concept: str | None = None
        if concept is not None:
            target_concept = self.concept_graph.canonicalize(concept) if self.concept_graph else concept.strip()

        # Lựa chọn danh sách câu hỏi ban đầu dựa theo chỉ mục
        if target_concept is not None and target_diff is not None:
            candidates = self._by_concept_difficulty.get((target_concept, target_diff), [])
        elif target_concept is not None:
            candidates = self._by_concept.get(target_concept, [])
        elif target_diff is not None:
            candidates = self._by_difficulty.get(target_diff, [])
        else:
            candidates = list(self._questions.values())

        if not excluded:
            return list(candidates)

        return [q for q in candidates if q.question_id not in excluded]

    def get_available_questions(
        self,
        concept: str | None = None,
        difficulty: str | Difficulty | None = None,
        attempted_ids: Iterable[str] | None = None,
    ) -> list[Question]:
        """Tiện ích tương đương get_questions phục vụ student model / policy."""
        return self.get_questions(concept=concept, difficulty=difficulty, exclude_ids=attempted_ids)

    def get_next_question(
        self,
        concept: str | None = None,
        difficulty: str | Difficulty | None = None,
        exclude_ids: Iterable[str] | None = None,
        strategy: str = "first",
        rng: random.Random | None = None,
    ) -> Question | None:
        """Lấy câu hỏi tiếp theo theo chiến lược ('first' hoặc 'random').

        Với strategy='random', truyền `rng=random.Random(seed)` để thí nghiệm tái lập được;
        nếu không truyền sẽ dùng một Random không seed.
        Trả về None nếu không còn câu hỏi nào phù hợp chưa làm.
        """
        if strategy not in {"first", "random"}:
            raise ValueError(f"Unknown strategy {strategy!r}; expected 'first' or 'random'.")
        candidates = self.get_questions(concept=concept, difficulty=difficulty, exclude_ids=exclude_ids)
        if not candidates:
            return None
        if strategy == "random":
            return (rng or random.Random()).choice(candidates)
        return candidates[0]

    def count(self, concept: str | None = None, difficulty: str | Difficulty | None = None) -> int:
        """Đếm số câu hỏi theo điều kiện."""
        return len(self.get_questions(concept=concept, difficulty=difficulty))

    def concept_summary(self) -> dict[str, dict[str, int]]:
        """Thống kê tổng số câu hỏi theo từng độ khó cho mỗi concept."""
        summary: dict[str, dict[str, int]] = {}
        for c in self.get_all_concepts():
            c_questions = self._by_concept.get(c, [])
            counts = {d: 0 for d in sorted(ALLOWED_DIFFICULTIES)}
            for q in c_questions:
                counts[q.difficulty] = counts.get(q.difficulty, 0) + 1
            counts["Total"] = len(c_questions)
            summary[c] = counts
        return summary

    def __len__(self) -> int:
        return len(self._questions)

    def __contains__(self, question_id: str) -> bool:
        return self.has_question(question_id)

    def __getitem__(self, question_id: str) -> Question:
        q = self.get_question(question_id)
        if q is None:
            raise KeyError(f"Question with ID '{question_id}' not found.")
        return q
