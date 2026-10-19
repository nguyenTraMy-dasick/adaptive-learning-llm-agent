"""Thành viên 3: Phase 6 — so sánh hiệu quả học giữa các policy (CHƯA triển khai).

Phụ thuộc phần chưa có:
- StudentModel và ActionExecutor của Thành viên 2 (engine gọi qua interface ActionExecutor).
- Learner mô phỏng (data/simulated_learners.json) hoặc người học thật, cùng bộ pre/post-test.

Khi có đủ, kịch bản dự kiến:
    for policy in (Random, RuleBased, LLM):
        for learner in learners:
            engine = LearningEngine(policy, question_bank, executor)
            session = engine.run_session(initial_state, respond=learner.answer, max_steps=N)
            engine.write_log(f"results/sessions/{policy.name}/{learner.id}.jsonl")
            ghi pre/post score, số câu hỏi, số bước tới mastery
    → metrics.learning_gain / normalized_gain / efficiency theo từng policy.
"""
from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "run_learning_experiment chưa chạy được: cần StudentModel/ActionExecutor (Thành viên 2) "
        "và learner mô phỏng. Phase 5 dùng evaluation/run_decision_evaluation.py."
    )


if __name__ == "__main__":
    raise SystemExit(main())
