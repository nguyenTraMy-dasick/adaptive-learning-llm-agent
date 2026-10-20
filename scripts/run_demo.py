"""Demo một phiên học: in từng lượt (action, câu hỏi, đúng/sai, mastery).

    python scripts/run_demo.py                                 # rule-based, learner mô phỏng "novice"
    python scripts/run_demo.py --policy random --learner weak_functions --steps 25
    python scripts/run_demo.py --policy rule --interactive     # bạn tự trả lời câu hỏi
    python scripts/run_demo.py --policy llm --steps 10         # cần LLM_* trong .env
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from adaptive_learning.action_executor import ActionExecutor  # noqa: E402
from adaptive_learning.engine import LearningEngine  # noqa: E402
from adaptive_learning.policies.llm_clients import OpenAICompatibleClient, load_env_file  # noqa: E402
from adaptive_learning.question_bank import QuestionBank  # noqa: E402
from adaptive_learning.schemas import LearnerState, Question  # noqa: E402
from adaptive_learning.student_model import StudentModel  # noqa: E402
from run_learning_experiment import load_config, make_policy  # noqa: E402
from simulated_learner import SimulatedLearner, load_profiles  # noqa: E402


def ask_human(question: Question) -> str:
    print("\n" + question.question)
    for letter, text in question.options.items():
        print(f"  {letter}. {text}")
    while True:
        answer = input("Your answer (A/B/C/D): ").strip().upper()
        if answer in {"A", "B", "C", "D"}:
            return answer


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo một phiên học với engine.")
    parser.add_argument("--policy", default="rule", choices=["random", "rule", "llm"])
    parser.add_argument("--learner", default="novice", help="id trong data/simulated_learners.json")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interactive", action="store_true", help="Người dùng tự trả lời thay learner mô phỏng")
    parser.add_argument("--log", type=Path, help="Ghi log JSONL của phiên")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "configs" / "experiment_config.json")
    qb = QuestionBank()
    client = None
    if args.policy == "llm":
        load_env_file(PROJECT_ROOT / ".env")
        client = OpenAICompatibleClient.from_env()
    policy = make_policy(args.policy, args.seed, qb, config, client)
    executor = ActionExecutor(qb, StudentModel(**config.get("student_model", {})))
    engine = LearningEngine(policy, qb, executor, mastery_goal=float(config.get("mastery_goal", 0.85)))

    learner = None
    if args.interactive:
        respond = ask_human
    else:
        profiles = {p.id: p for p in load_profiles(PROJECT_ROOT / config.get("learners_path", "data/simulated_learners.json"))}
        if args.learner not in profiles:
            raise SystemExit(f"Unknown learner {args.learner!r}; choose from {', '.join(profiles)}")
        learner = SimulatedLearner(profiles[args.learner], qb, seed=args.seed)
        respond = learner.respond
        print(f"Learner mô phỏng: {args.learner} | điểm kỳ vọng ban đầu {learner.expected_test_score():.3f}")

    state = LearnerState()
    print(f"Policy: {policy.name} | tối đa {args.steps} lượt\n")
    for turn in range(1, args.steps + 1):
        if engine.goal_reached(state) or engine.questions_exhausted(state):
            break
        result = engine.run_turn(state, respond)
        if learner is not None:
            learner.observe(result)
        d = result.decision
        label = f"{d.action_type.value} {d.concept or ''} {d.difficulty or ''}".strip() if d else "<none>"
        if not result.valid:
            print(f"{turn:>3}. INVALID {label}: {result.error}")
            continue
        state = result.updated_state
        line = f"{turn:>3}. {label:<45}"
        if result.question_id:
            q = result.question
            line += f" {result.question_id} {'đúng' if result.correct else 'sai '} mastery={state.mastery[q.concept]:.2f}"
        print(line + (f"   [{d.rationale}]" if d.rationale else ""))
        if args.interactive and result.feedback:
            print(f"     {result.feedback}")

    mastered = [c for c in qb.get_all_concepts() if state.mastery.get(c, 0) >= engine.mastery_goal]
    print(f"\nĐã hỏi {len(state.attempted_question_ids)} câu; {len(mastered)}/14 concept đạt {engine.mastery_goal}.")
    if learner is not None:
        print(f"Điểm kỳ vọng sau phiên: {learner.expected_test_score():.3f}")
    if args.log:
        print(f"Log: {engine.write_log(args.log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
