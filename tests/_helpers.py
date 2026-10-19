"""Tiện ích dùng chung cho test của Thành viên 3 (policy, action space, engine)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_learning.policies.base import DecisionContext  # noqa: E402
from adaptive_learning.question_bank import QuestionBank  # noqa: E402
from adaptive_learning.schemas import PolicyDecision  # noqa: E402

_QB = None


def question_bank() -> QuestionBank:
    global _QB
    if _QB is None:
        _QB = QuestionBank(
            csv_path=PROJECT_ROOT / "data" / "question_bank.csv",
            concept_graph=PROJECT_ROOT / "data" / "concept_graph.json",
        )
    return _QB


V, DT, EX, ST, BO, CO = (
    "Variables & assignment", "Data types", "Expressions & operators", "Strings",
    "Boolean expressions", "Conditionals",
)
FU, SC, RE, FO, LI = "Functions", "Scope", "Recursion", "For loops & range", "Lists"

RECURSION_INTRODUCED = (V, DT, EX, BO, CO, FU, SC, RE)


def ctx(focus=None, introduced=(), last=None, last_qid=None, last_correct=None, step=0):
    """Tạo DecisionContext ngắn gọn; last = (action_type, concept, difficulty)."""
    last_action = PolicyDecision(*last) if last else None
    last_concept = question_bank().get_question(last_qid).concept if last_qid else None
    return DecisionContext(
        focus_concept=focus,
        introduced_concepts=tuple(introduced),
        last_action=last_action,
        last_question_id=last_qid,
        last_question_concept=last_concept,
        last_correct=last_correct,
        step=step,
    )
