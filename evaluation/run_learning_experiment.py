"""Thành viên 3: Phase 6 — so sánh hiệu quả học giữa các policy bằng learner mô phỏng.

Mỗi phiên: một policy × một hồ sơ learner (data/simulated_learners.json) × một seed.
Mọi policy dùng CÙNG QuestionBank, StudentModel, ActionExecutor (Thành viên 2), action space,
ngân sách lượt (max_steps) và cùng seed learner, chỉ khác cách chọn action.

Đo cho mỗi phiên:
- pre / post: điểm kỳ vọng trên toàn bộ question bank theo kiến thức thật của learner
  (xem evaluation/simulated_learner.py); gain = post - pre; normalized_gain.
- n_questions, efficiency = gain / n_questions, gain_per_turn = gain / số lượt.
- goal_reached và steps_to_goal (mọi concept đạt mastery_goal theo StudentModel).
- invalid_turns (quan trọng với LLM policy).

Ví dụ:
    python evaluation/run_learning_experiment.py
    python evaluation/run_learning_experiment.py --policies random,rule --repeats 30
    python evaluation/run_learning_experiment.py --policies rule,llm --repeats 2 --log-dir results/sessions

LLM: mỗi lượt là một lần gọi API (max_steps × số learner × repeats lần), nên thử repeats nhỏ trước.
Kết quả là bằng chứng trên MÔ PHỎNG, không thay cho thí nghiệm với người học thật.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

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
from adaptive_learning.policies import (  # noqa: E402
    BasePolicy,
    LLMPolicy,
    RandomPolicy,
    RuleBasedPolicy,
    RuleConfig,
)
from adaptive_learning.policies.llm_clients import OpenAICompatibleClient, load_env_file  # noqa: E402
from adaptive_learning.question_bank import QuestionBank  # noqa: E402
from adaptive_learning.schemas import LearnerState  # noqa: E402
from adaptive_learning.student_model import StudentModel  # noqa: E402
from metrics import efficiency, learning_gain, normalized_gain  # noqa: E402
from simulated_learner import LearnerProfile, SimulatedLearner, load_profiles  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment_config.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "learning_experiment.json"
POLICY_NAMES = ("random", "rule", "llm")


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_policy(name: str, seed: int, question_bank: QuestionBank, config: dict[str, Any],
                llm_client: Any = None) -> BasePolicy:
    graph = question_bank.concept_graph
    rule = RuleBasedPolicy(RuleConfig(**config.get("rule_config", {})), concept_graph=graph)
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "rule":
        return rule
    if name == "llm":
        if llm_client is None:
            raise ValueError("LLM policy needs a model client")
        llm_cfg = config.get("llm", {})
        return LLMPolicy(
            llm_client,
            concept_graph=graph,
            max_retries=int(llm_cfg.get("max_retries", 2)),
            fallback_policy=rule if llm_cfg.get("fallback_to_rule") else None,
        )
    raise ValueError(f"Unknown policy {name!r}; choose from {', '.join(POLICY_NAMES)}")


def run_session(
    policy: BasePolicy,
    profile: LearnerProfile,
    seed: int,
    question_bank: QuestionBank,
    config: dict[str, Any],
    log_path: Path | None = None,
) -> dict[str, Any]:
    learner = SimulatedLearner(profile, question_bank, seed=seed)
    executor = ActionExecutor(
        question_bank,
        StudentModel(**config.get("student_model", {})),
        question_strategy=config.get("question_strategy", "first"),
    )
    engine = LearningEngine(policy, question_bank, executor, mastery_goal=float(config.get("mastery_goal", 0.85)))

    steps_to_goal: list[int] = []

    def on_turn(result) -> None:
        learner.observe(result)
        state = result.updated_state
        if not steps_to_goal and state is not None and engine.goal_reached(state):
            steps_to_goal.append(len(engine.logs))

    pre = learner.expected_test_score()
    session = engine.run_session(
        LearnerState(),
        respond=learner.respond,
        max_steps=int(config.get("max_steps", 60)),
        max_turns_without_question=config.get("max_turns_without_question"),
        on_turn=on_turn,
    )
    post = learner.expected_test_score()
    if log_path is not None:
        engine.write_log(log_path)

    n_questions = len(session.final_state.attempted_question_ids)
    turns = len(session.logs)
    gain = learning_gain(pre, post)
    action_counts: dict[str, int] = {}
    for log in session.logs:
        if log.result.valid and log.result.decision is not None:
            key = log.result.decision.action_type.value
            action_counts[key] = action_counts.get(key, 0) + 1
    return {
        "policy": policy.name,
        "learner": profile.id,
        "seed": seed,
        "pre": pre,
        "post": post,
        "gain": gain,
        "normalized_gain": normalized_gain(pre, post),
        "n_questions": n_questions,
        "turns": turns,
        "efficiency": efficiency(gain, n_questions),
        "gain_per_turn": gain / turns if turns else None,
        "goal_reached": bool(steps_to_goal) or session.stop_reason == "goal_reached",
        "steps_to_goal": steps_to_goal[0] if steps_to_goal else None,
        "invalid_turns": sum(not log.result.valid for log in session.logs),
        "stop_reason": session.stop_reason,
        "concepts_mastered_model": sum(
            session.final_state.mastery.get(c, 0.0) >= engine.mastery_goal for c in question_bank.get_all_concepts()
        ),
        "action_counts": action_counts,
        "final_knowledge": learner.knowledge,
    }


def _mean_sd(values: list[float]) -> dict[str, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("gain", "normalized_gain", "n_questions", "efficiency", "gain_per_turn", "steps_to_goal",
              "invalid_turns", "concepts_mastered_model")
    summary: dict[str, Any] = {}
    for policy in dict.fromkeys(r["policy"] for r in records):
        rows = [r for r in records if r["policy"] == policy]
        entry = {f: _mean_sd([r[f] for r in rows]) for f in fields}
        entry["goal_reached_rate"] = sum(r["goal_reached"] for r in rows) / len(rows)
        entry["by_learner"] = {
            learner: _mean_sd([r["gain"] for r in rows if r["learner"] == learner])
            for learner in dict.fromkeys(r["learner"] for r in rows)
        }
        summary[policy] = entry
    return summary


def _fmt(stat: dict[str, Any], digits: int = 3) -> str:
    if stat["mean"] is None:
        return "-"
    return f"{stat['mean']:.{digits}f}±{stat['sd']:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6: thí nghiệm học với learner mô phỏng.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policies", help="Ghi đè config, ví dụ random,rule,llm")
    parser.add_argument("--repeats", type=int, help="Số seed cho mỗi cặp policy × learner")
    parser.add_argument("--max-steps", type=int, help="Ngân sách lượt mỗi phiên")
    parser.add_argument("--learners", help="Chỉ chạy các learner id này (cách nhau bởi dấu phẩy)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-dir", type=Path, help="Ghi log JSONL từng phiên vào thư mục này")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    policies = [p.strip() for p in (args.policies or ",".join(config.get("policies", ["random", "rule"]))).split(",")
                if p.strip()]
    for name in policies:
        if name not in POLICY_NAMES:
            raise SystemExit(f"Unknown policy {name!r}; choose from {', '.join(POLICY_NAMES)}")
    repeats = args.repeats if args.repeats is not None else int(config.get("repeats", 10))
    base_seed = int(config.get("seed", 0))

    question_bank = QuestionBank()
    profiles = load_profiles(PROJECT_ROOT / config.get("learners_path", "data/simulated_learners.json"),
                             question_bank.concept_graph)
    if args.learners:
        wanted = {x.strip() for x in args.learners.split(",")}
        profiles = [p for p in profiles if p.id in wanted]
        if not profiles:
            raise SystemExit(f"No learner matches {sorted(wanted)}")

    llm_client = None
    if "llm" in policies:
        load_env_file(PROJECT_ROOT / ".env")
        llm_client = OpenAICompatibleClient.from_env()

    records: list[dict[str, Any]] = []
    for name in policies:
        for profile in profiles:
            for i in range(repeats):
                seed = base_seed + i
                policy = make_policy(name, seed, question_bank, config, llm_client)
                log_path = args.log_dir / name / f"{profile.id}_seed{seed}.jsonl" if args.log_dir else None
                records.append(run_session(policy, profile, seed, question_bank, config, log_path))

    summary = summarize(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": {**config, "policies": policies, "repeats": repeats},
                   "summary": summary, "sessions": records}, f, ensure_ascii=False, indent=2)

    try:
        shown = args.out.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown = args.out
    print(f"{len(records)} phiên ({len(policies)} policy × {len(profiles)} learner × {repeats} seed), "
          f"max_steps={config.get('max_steps', 60)} | kết quả: {shown}")
    print("Lưu ý: kết quả trên learner MÔ PHỎNG, phụ thuộc giả định trong evaluation/simulated_learner.py.\n")
    header = f"{'policy':<11}{'gain':>14}{'norm.gain':>14}{'#questions':>13}{'gain/turn':>14}{'goal':>7}{'invalid':>9}"
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(f"{name:<11}{_fmt(s['gain']):>14}{_fmt(s['normalized_gain']):>14}{_fmt(s['n_questions'], 1):>13}"
              f"{_fmt(s['gain_per_turn'], 4):>14}{s['goal_reached_rate']:>7.0%}{_fmt(s['invalid_turns'], 1):>9}")
    print("\nGain theo learner (mean):")
    learners = list(next(iter(summary.values()))["by_learner"]) if summary else []
    print(f"{'learner':<16}" + "".join(f"{n:>12}" for n in summary))
    for learner in learners:
        row = "".join(f"{summary[n]['by_learner'][learner]['mean']:>12.3f}" for n in summary)
        print(f"{learner:<16}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
