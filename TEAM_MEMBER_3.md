# Báo cáo thực hiện công việc — Thành viên 3: Policies, Engine và Đánh giá

## 1. Phạm vi

| File | Nội dung |
| :--- | :--- |
| `src/adaptive_learning/policies/base.py` | `BasePolicy`, `DecisionContext`, `validate()` |
| `src/adaptive_learning/action_space.py` | Sinh `available_actions` cho mỗi bước |
| `src/adaptive_learning/policies/random_policy.py` | Baseline A: ngẫu nhiên có seed |
| `src/adaptive_learning/policies/rule_based_policy.py` | Baseline B: luật R0–R6, ngưỡng trong `RuleConfig` |
| `src/adaptive_learning/policies/llm_policy.py` | Policy C: LLM chọn action theo số thứ tự |
| `src/adaptive_learning/policies/llm_clients.py` | Client kiểu OpenAI Chat Completions (chỉ dùng thư viện chuẩn) |
| `src/adaptive_learning/engine.py` | `LearningEngine`: validate, thực thi, log, chạy phiên |
| `evaluation/` | Phase 5 (đánh giá quyết định) và Phase 6 (thí nghiệm học mô phỏng) |
| `data/simulated_learners.json`, `configs/experiment_config.json` | Hồ sơ learner mô phỏng và cấu hình thí nghiệm |
| `scripts/run_demo.py` | Demo một phiên học |

## 2. Luồng một lượt

```text
LearnerState + DecisionContext
   → build_available_actions()                (action_space.py)
   → policy.select_action(state, actions, context)
   → BasePolicy.validate()                    (không hợp lệ: không thực thi, state giữ nguyên)
   → ActionExecutor.execute(decision, state, respond)   (Thành viên 2: chọn câu, chấm, cập nhật mastery)
   → cập nhật DecisionContext + ghi TurnLog (JSONL)
```

Policy không tạo câu hỏi, không chấm đáp án, không cập nhật mastery. Rationale của policy
chỉ để phân tích, không dùng để quyết định action có hợp lệ hay không.

## 3. Action space

| Action | Có mặt khi |
| :--- | :--- |
| `ASK(focus, d)` | focus còn câu chưa làm ở độ khó `d` |
| `HINT(focus)` | đã hỏi ít nhất một câu của focus |
| `EXPLAIN(focus)` | đã có focus |
| `REVIEW_PREREQUISITE(p)` | `p` là tiên quyết trực tiếp của focus |
| `ADVANCE(c)` | `c` khác focus, không phải tổ tiên của focus, mọi tiên quyết của `c` đã được giới thiệu, `c` còn câu |

Đầu phiên chỉ có `ADVANCE` tới concept gốc. Action space không chặn theo ngưỡng mastery,
nên policy vẫn có thể chuyển concept quá sớm; đây là một lỗi cần đo ở Phase 5.

## 4. Ba policy

- **Random**: chọn đều trên danh sách action (không theo loại action), có seed.
- **Rule-based** (ngưỡng mặc định low 0.40, high 0.75, advance 0.85, tiên quyết yếu 0.60):
  R0 đầu phiên → ADVANCE; R1 mastery ≥ 0.85 → ADVANCE; R2 vừa sai và mastery < 0.40 và có
  tiên quyết yếu → REVIEW tiên quyết yếu nhất; R3 vừa sai và mastery < 0.40 → EXPLAIN;
  R4 vừa sai và 0.40 ≤ mastery < 0.75 → HINT; R5 → ASK với độ khó theo mastery;
  R6 hết câu → ADVANCE/REVIEW/EXPLAIN. "Vừa sai" chỉ đúng khi action ngay trước là ASK,
  để tránh vòng lặp ôn tiên quyết rồi quay lại.
- **LLM**: prompt gồm graph, mastery ("not assessed" nếu chưa có), bối cảnh phiên, 5 lỗi và
  5 action gần nhất, danh sách action đánh số. Không gửi câu hỏi, đáp án hay ngưỡng của
  rule-based. Output `{"action_index", "rationale"}`; sai định dạng thì hỏi lại
  (`max_retries`), sau đó dùng `fallback_policy` hoặc báo lỗi (engine ghi lượt invalid).
  `last_trace` lưu prompt, output thô và lỗi parse.

Cấu hình LLM: copy `.env.example` thành `.env`, điền `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.

## 5. Đánh giá

### Phase 5 — chất lượng quyết định

`python evaluation/run_decision_evaluation.py [--policies random,rule,llm]`

12 learner state mô phỏng trong `evaluation/decision_scenarios.json`. Mọi policy nhận cùng
state, context và danh sách action. Cần ít nhất 2 người gán `expert_labels` độc lập theo
`evaluation/decision_rubric.md` thì mới tính được agreement.

### Phase 6 — hiệu quả học (mô phỏng)

`python evaluation/run_learning_experiment.py [--policies random,rule,llm] [--repeats 10] [--max-steps 60]`

Learner mô phỏng có kiến thức thật mà policy không thấy; policy chỉ thấy mastery của
StudentModel. Pre/post-test là điểm kỳ vọng trên toàn bộ 70 câu theo kiến thức thật.
Giả định mô hình (đoán 0.25, trượt 0.10, tiên quyết yếu làm giảm cả khả năng làm bài lẫn
tốc độ học, EXPLAIN/HINT/REVIEW/ASK có hệ số học riêng) nằm ở docstring
`evaluation/simulated_learner.py`.

Kết quả mặc định (6 hồ sơ × 10 seed, 60 lượt mỗi phiên, ngày 20/10):

| Policy | Gain | Normalized gain | Số câu đã hỏi |
| :--- | :---: | :---: | :---: |
| Random | 0.064 ± 0.031 | 0.108 ± 0.041 | 14.8 |
| Rule-based | 0.100 ± 0.050 | 0.168 ± 0.063 | 31.9 |

Đây là bằng chứng trên mô phỏng, phụ thuộc giả định của simulator; không thay cho thí
nghiệm với người học thật (project_0, Phase 6).

## 6. Kiểm thử

`python -m unittest discover -s tests` (120 test, gồm cả Task 1 và Task 2).

| File | Nội dung |
| :--- | :--- |
| `tests/test_action_space.py` | Action đầu phiên, thứ tự, hết câu, HINT, ADVANCE theo tiên quyết, alias |
| `tests/test_random_model.py` | Seed, reset, luôn hợp lệ, danh sách rỗng |
| `tests/test_rule_based_policy.py` | Từng luật R0–R6, không lặp REVIEW/ADVANCE, không sửa input |
| `tests/test_llm_policy.py` | Prompt không lộ đáp án, parse, hỏi lại, fallback, client HTTP |
| `tests/test_engine.py` | Validate, invalid, điều kiện dừng, log JSONL, tích hợp với executor thật |
| `tests/test_decision_evaluation.py` | Metric và runner Phase 5 |
| `tests/test_learning_experiment.py` | Learner mô phỏng và runner Phase 6 |

## 7. Hạn chế và việc còn lại

- Chưa chạy LLM với API thật (cần `.env`). Nên chạy Phase 5 với LLM trước, rồi Phase 6 với
  `--repeats` nhỏ vì mỗi lượt là một lần gọi API.
- `expert_labels` của Phase 5 còn trống.
- Mỗi concept chỉ có 5 câu nên learner yếu thường dùng hết câu trước khi đạt 0.85.
- Kết quả Phase 6 phụ thuộc giả định của simulator; nên báo cáo thêm phân tích độ nhạy
  (đổi các hệ số học) nếu dùng làm kết quả chính.
