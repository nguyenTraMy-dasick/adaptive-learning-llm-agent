"""Thành viên 3: Phase 5 — chạy các policy trên cùng bộ learner state mô phỏng.

Mỗi scenario trong evaluation/decision_scenarios.json gồm learner_state, context và
expert_labels. Với mỗi scenario, mọi policy nhận CÙNG state, CÙNG context và CÙNG
available_actions (sinh bởi action_space.build_available_actions).

Ví dụ:
    python evaluation/run_decision_evaluation.py
    python evaluation/run_decision_evaluation.py --policies random,rule --random-repeats 50
    python evaluation/run_decision_evaluation.py --policies random,rule,llm   # cần LLM_* trong môi trường

Kết quả ghi ra results/decision_eval.json (từng quyết định + tổng hợp) và in bảng tóm tắt.
Không cần StudentModel / ActionExecutor (Thành viên 2).
"""
from __future__ import annotations

import argparse
import json
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

from adaptive_learning.action_space import build_available_actions  # noqa: E402
from adaptive_learning.policies import (  # noqa: E402
    BasePolicy,
    DecisionContext,
    LLMPolicy,
    RandomPolicy,
    RuleBasedPolicy,
)
from adaptive_learning.policies.llm_clients import OpenAICompatibleClient, load_env_file  # noqa: E402
from adaptive_learning.question_bank import QuestionBank  # noqa: E402
from adaptive_learning.schemas import LearnerState  # noqa: E402
from metrics import auto_flags, scenario_agreement, summarize_decisions  # noqa: E402

DEFAULT_SCENARIOS = PROJECT_ROOT / "evaluation" / "decision_scenarios.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "decision_eval.json"


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"] if isinstance(data, dict) else data


def evaluate_policy(
    policy: BasePolicy,
    scenarios: list[dict[str, Any]],
    question_bank: QuestionBank,
    run_id: int = 0,
) -> list[dict[str, Any]]:
    records = []
    for sc in scenarios:
        state = LearnerState.from_dict(sc["learner_state"])
        context = DecisionContext.from_dict(sc.get("context"))
        actions = build_available_actions(state, question_bank, context)
        action_dicts = [a.to_dict() for a in actions]
        error = None
        try:
            decision = policy.select_action(state, actions, context)
            valid = BasePolicy.validate(decision, actions)
            decision_dict = decision.to_dict()
        except Exception as exc:  # LLM lỗi... → quyết định invalid
            valid, decision_dict, error = False, None, f"{type(exc).__name__}: {exc}"
        trace = getattr(policy, "last_trace", None)
        records.append({
            "scenario": sc["id"],
            "run": run_id,
            "policy": policy.name,
            "valid": valid,
            "decision": decision_dict,
            "error": error,
            "agreement": scenario_agreement(decision_dict if valid else None, sc.get("expert_labels", [])),
            "flags": auto_flags(decision_dict if valid else None, state.mastery, sc.get("context") or {}, action_dicts),
            "n_available_actions": len(actions),
            "llm_fallback_used": bool(trace.get("fallback_used")) if isinstance(trace, dict) and trace else None,
        })
    return records


def build_policies(names: list[str], question_bank: QuestionBank, seed: int) -> dict[str, list[BasePolicy]]:
    policies: dict[str, list[BasePolicy]] = {}
    for name in names:
        if name == "random":
            policies[name] = []  # tạo theo từng lần lặp (seed + i)
        elif name == "rule":
            policies[name] = [RuleBasedPolicy(concept_graph=question_bank.concept_graph)]
        elif name == "llm":
            load_env_file(PROJECT_ROOT / ".env")
            client = OpenAICompatibleClient.from_env()
            policies[name] = [LLMPolicy(client, concept_graph=question_bank.concept_graph)]
        else:
            raise SystemExit(f"Unknown policy {name!r}; choose from random, rule, llm")
    return policies


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5: đánh giá quyết định của các policy.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--policies", default="random,rule", help="Danh sách cách nhau bởi dấu phẩy: random,rule,llm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-repeats", type=int, default=20, help="Số lần chạy RandomPolicy (seed, seed+1, ...)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    question_bank = QuestionBank()
    scenarios = load_scenarios(args.scenarios)
    names = [n.strip() for n in args.policies.split(",") if n.strip()]
    policies = build_policies(names, question_bank, args.seed)

    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for name in names:
        if name == "random":
            records = []
            for i in range(args.random_repeats):
                records += evaluate_policy(RandomPolicy(seed=args.seed + i), scenarios, question_bank, run_id=i)
        else:
            records = evaluate_policy(policies[name][0], scenarios, question_bank)
        all_records += records
        summaries[name] = summarize_decisions(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summaries, "records": all_records}, f, ensure_ascii=False, indent=2)

    try:
        shown = args.out.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown = args.out
    print(f"Scenarios: {len(scenarios)} | kết quả: {shown}")
    for name, s in summaries.items():
        print(f"\n[{name}] n={s['n']} valid_rate={s['valid_rate']:.2f} labeled={s['labeled_scenarios']}")
        if s["labeled_scenarios"]:
            print(f"  agreement exact={s['exact_agreement']:.2f} acceptable={s['acceptable_agreement']:.2f} "
                  f"type={s['type_agreement']:.2f}")
        print("  action types: " + ", ".join(f"{k} {v:.0%}" for k, v in s["action_type_distribution"].items()))
        if s["flag_rate"]:
            print("  auto flags:   " + ", ".join(f"{k} {v:.0%}" for k, v in s["flag_rate"].items()))
    if not any(s["labeled_scenarios"] for s in summaries.values()):
        print("\nChưa có nhãn expert trong scenarios nên chưa tính agreement (xem decision_rubric.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
