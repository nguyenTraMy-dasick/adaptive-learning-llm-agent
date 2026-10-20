# Phân chia công việc — Adaptive Learning LLM Agent

Tài liệu này giúp ba thành viên làm song song mà vẫn ghép được. Trước khi bắt đầu, cả nhóm dùng chung hợp đồng trong `src/adaptive_learning/schemas.py`, đặc biệt là `LearnerState`, `Action`, `PolicyDecision` và `ActionType`.

## Thành viên 1 — Hợp đồng dữ liệu và Question Bank

**Mục tiêu:** Chốt cách biểu diễn learner state/action, chuẩn hóa nguồn dữ liệu câu hỏi và đảm bảo các concept có thể liên kết với concept graph.

**Phạm vi file:**
- `src/adaptive_learning/schemas.py`
- `src/adaptive_learning/question_bank.py`
- `data/question_bank.csv` (chỉ sửa sau khi nhóm thống nhất thay đổi schema dữ liệu)
- `data/concept_graph.json`
- `scripts/validate_question_bank.py`

**Việc cần làm:**
1. Giữ một định dạng state và action duy nhất cho cả ba policy.
2. Chốt ràng buộc mastery `[0, 1]`, difficulty (`Easy`, `Medium`, `Hard`) và action được phép.
3. Viết QuestionBank đọc CSV cố định; cung cấp thao tác lấy câu hỏi theo concept/difficulty và loại câu đã làm khi phù hợp.
4. Quyết định cách biểu diễn prerequisite: CSV hiện dùng `prerequisite_concept` (số ít). Không tự đổi tên cột nếu chưa thống nhất.
5. Xác thực concept trong CSV khớp chính xác với graph; báo lỗi dễ hiểu khi thiếu concept hoặc metadata.

**Kết quả bàn giao:** Schema được nhóm duyệt; QuestionBank đọc được file hiện tại; graph có dữ liệu hoặc có quyết định rõ về nguồn graph; script kiểm tra báo lỗi dữ liệu có thể sửa được.

**Không làm:** Không thiết kế policy, không gọi LLM, không chấm đáp án.

## Thành viên 2 — Student Model và xử lý câu hỏi

**Mục tiêu:** Xây thành phần Python xác định kết quả đáp án và cập nhật learner state; giữ thuật toán student model cố định để so sánh policy công bằng.

**Phạm vi file:**
- `src/adaptive_learning/student_model.py`
- `src/adaptive_learning/action_executor.py`
- `tests/test_student_model.py`
- `tests/test_action_executor.py`

**Việc cần làm:**
1. Chốt công thức cập nhật mastery (ví dụ baseline tỉ lệ đúng có smoothing hoặc update đơn giản) và ghi rõ tham số.
2. Chấm bằng `correct_answer` của question bank; cập nhật `recent_errors`, `recent_actions`, `attempted_question_ids` theo một quy tắc thống nhất.
3. Xác định cách chạy `ASK`, `HINT`, `EXPLAIN`, `REVIEW_PREREQUISITE`, `ADVANCE`. Python chọn nội dung từ nguồn cố định; không để policy/LLM tự tạo câu hỏi hay tự chấm.
4. Với hint/explanation, đề xuất nơi lưu nội dung và phân biệt giải thích câu trả lời với giải thích toàn concept.

**Kết quả bàn giao:** Student model có API cập nhật rõ ràng; executor nhận action đã được xác thực và trả observation/feedback; có test các trường hợp đúng, sai, mastery biên và câu hỏi lặp.

**Không làm:** Không quyết định action tiếp theo; không thay đổi Question Bank hoặc policy interface riêng lẻ.

## Thành viên 3 — Policies, Engine và tích hợp

**Mục tiêu:** So sánh ba cách chọn action trên cùng input/action hợp lệ, sau đó kết nối policy với QuestionBank và student model qua engine.

**Phạm vi file:**
- `src/adaptive_learning/policies/base.py`
- `src/adaptive_learning/policies/random_policy.py`
- `src/adaptive_learning/policies/rule_based_policy.py`
- `src/adaptive_learning/policies/llm_policy.py`
- `src/adaptive_learning/engine.py`
- `evaluation/`
- `tests/test_random_model.py`, `tests/test_rule_based_policy.py`, `tests/test_llm_policy.py`, `tests/test_engine.py`
- File thêm mới: `src/adaptive_learning/action_space.py`, `src/adaptive_learning/policies/llm_clients.py`, `evaluation/decision_scenarios.json`, `tests/test_action_space.py`, `tests/test_decision_evaluation.py`, `tests/_helpers.py`

**Việc cần làm:**
1. Ba policy có cùng `select_action(learner_state, available_actions)` và cùng kiểu trả về.
2. Mọi policy chỉ đề xuất một action trong tập hợp được engine cung cấp. Random có thể seed; rule phải ghi rõ luật; LLM chỉ nhận state/action và trả action có cấu trúc.
3. Engine kiểm tra action, gọi executor/question bank, nhận response, giao Python chấm và cập nhật student model, rồi ghi log.
4. LLM rationale chỉ là dữ liệu để phân tích, không dùng làm bằng chứng action hợp lệ hay kết quả học.
5. Đảm bảo các policy nhận cùng state và cùng action space ở cùng bước đánh giá.

**Kết quả bàn giao:** Một lượt học chạy xuyên suốt với từng policy; action sai bị từ chối hoặc xử lý theo quy tắc đã chốt; log đủ để tái lập và evaluation so sánh được ba policy.

**Phụ thuộc:** Chốt schema với thành viên 1 trước khi implement sâu; thống nhất API của executor/student model với thành viên 2 trước khi tích hợp engine.

## Thứ tự phối hợp

1. Cả nhóm duyệt `schemas.py` và quyết định nguồn/prerequisite graph.
2. Thành viên 1 và 2 triển khai theo hợp đồng; thành viên 3 có thể chuẩn bị policy stubs nhưng không tự đổi schema.
3. Ba thành viên thống nhất API QuestionBank, executor và student model.
4. Thành viên 3 ghép engine; cả nhóm chạy tests/integration và chốt các trường hợp biên.

## Các quyết định đã chốt bởi Thành viên 1

- [x] **Concept graph**: Đã chuẩn hóa và điền đầy đủ vào `data/concept_graph.json` dựa trên nguồn `project/project_1.md` và tổng hợp giáo trình (14 concepts, 21 edges, kiểm tra cấu trúc DAG không chu trình).
- [x] **Canonical concept names**: Thống nhất dùng tên concept chuẩn theo CSV: `"Functions"` (với alias `"Functions (def, params, return)"` và `"Functions (def/params/return)"`), `"Data types"` (với alias `"Data type"`). `ConceptGraph.canonicalize()` tự động map alias về canonical name.
- [x] **Cột prerequisite**: Giữ nguyên tên cột CSV `prerequisite_concept` (số ít). Trong schema `Question`, trường dữ liệu là `prerequisite_concept: str` và bổ sung property `prerequisites: tuple[str, ...]` để các thành viên 2 và 3 truy xuất dạng danh sách tiện lợi.
- [x] **Ràng buộc mastery & difficulty**: Mastery bắt buộc là số thực trong khoảng `[0.0, 1.0]`. Difficulty chuẩn hóa theo 3 mức `{"Easy", "Medium", "Hard"}` qua `Difficulty(str, Enum)` và `ALLOWED_DIFFICULTIES`.
- [x] **QuestionBank & Validation**: `QuestionBank` đã hoạt động tốt với toàn bộ 70 câu hỏi hiện tại của CSV; `scripts/validate_question_bank.py` cung cấp báo cáo chi tiết về tính toàn vẹn và hỗ trợ flag `--fix`. Đã có 30 unit tests chạy thành công tại `tests/test_question_bank.py`.
- [x] **Rà soát dữ liệu question bank (18/10)**: Đã chạy thật code của cả 70 câu và sửa: đáp án sai (FOR_002), code thụt lề hỏng (REC_001/003/004/005), option trùng (STR_005), typo (CON_003, EXP_004), prerequisite mâu thuẫn graph (VAR_002–005, TYP_005, WHI_004, SCO_005), 4 câu đặt sai concept được viết lại (LIST_002, TUP_002, DICT_002, REC_002), mapping lỗi không khớp option (VAR_004, EXP_001, EXP_004, STR_003, FUN_005). Chi tiết ở `TEAM_MEMBER_1.md` mục 2.6.
- [x] **Ràng buộc prerequisite câu hỏi ↔ graph**: Mọi prerequisite của câu hỏi phải là tổ tiên của concept trong `concept_graph.json` (không trỏ vào chính nó). `QuestionBank` chuẩn hóa prerequisite về canonical name ngay khi load, nên `q.prerequisites` dùng trực tiếp làm key cho `mastery` được. Validator báo lỗi nếu vi phạm.
- [x] **`LearnerState.mastery` chỉ đọc**: Sửa trực tiếp `state.mastery[...] = ...` sẽ báo `TypeError`; muốn cập nhật thì tạo `LearnerState` mới. Mastery phải là số thực hữu hạn trong `[0.0, 1.0]` (không nhận `bool`, `NaN`, `inf`).

## Trạng thái Thành viên 3

Chi tiết và hướng dẫn chạy: `TEAM_MEMBER_3.md`.

- [x] **Interface policy**: `select_action(learner_state, available_actions, context=None)`. Tham số `context: DecisionContext` (focus concept, các concept đã giới thiệu, action trước, câu hỏi gần nhất và đúng/sai, số bước) do engine theo dõi và truyền giống nhau cho cả ba policy. Nhờ vậy policy không phải đọc chuỗi tự do trong `recent_actions`. `BasePolicy.validate()` so khớp `(action_type, concept, difficulty)` và bỏ qua rationale.
- [x] **Action space** (`action_space.py`): ASK(focus, độ khó còn câu), HINT(focus, khi đã hỏi ít nhất một câu của focus), EXPLAIN(focus), REVIEW_PREREQUISITE(tiên quyết trực tiếp của focus), ADVANCE(concept chưa phải focus, không phải tổ tiên của focus, có mọi tiên quyết đã giới thiệu và còn câu hỏi). Đầu phiên chỉ có ADVANCE tới concept gốc. Không chặn theo ngưỡng mastery, để còn đo được lỗi "advance quá sớm".
- [x] **RandomPolicy**: chọn đều trên available_actions, có seed và `reset()`.
- [x] **RuleBasedPolicy**: luật R0–R6 ghi trong docstring; ngưỡng ở `RuleConfig` (0.40 / 0.75 / 0.85, tiên quyết yếu < 0.60); rationale ghi luật đã dùng.
- [x] **LLMPolicy**: prompt chứa graph, mastery, context, lỗi/action gần đây và danh sách action đánh số. Không gửi câu hỏi, đáp án hay ngưỡng của rule-based. LLM trả JSON `{"action_index", "rationale"}`; output sai thì hỏi lại (`max_retries`), sau đó dùng `fallback_policy` hoặc raise `LLMDecisionError`. Toàn bộ prompt và output thô được lưu ở `last_trace`. Client `OpenAICompatibleClient` chỉ dùng thư viện chuẩn, cấu hình qua `.env` (xem `.env.example`).
- [x] **Engine**: validate → thực thi qua executor → cập nhật context → ghi log JSONL. Quyết định không hợp lệ, policy lỗi, hoặc executor từ chối (`valid=False`) thì state và context giữ nguyên. `run_session` dừng khi đạt mục tiêu, hết câu hỏi, hết action, hết ngân sách lượt, nhiều lượt invalid liên tiếp, hoặc (tùy chọn) quá nhiều lượt liên tiếp không hỏi câu nào.
- [x] **Tích hợp với Thành viên 2 (20/10)**: engine chạy với `ActionExecutor`/`StudentModel` thật; có test tích hợp trong `tests/test_engine.py`.
- [x] **Đánh giá Phase 5**: `evaluation/run_decision_evaluation.py`, `metrics.py`, `decision_rubric.md`, 12 scenario mẫu. `expert_labels` để trống có chủ đích, cần gán nhãn độc lập.
- [x] **Phase 6** (`run_learning_experiment.py`): learner mô phỏng (`evaluation/simulated_learner.py`, hồ sơ trong `data/simulated_learners.json`), cấu hình ở `configs/experiment_config.json`. Kết quả là bằng chứng trên mô phỏng, không thay cho thí nghiệm với người học thật.
- [x] **Demo**: `scripts/run_demo.py` (learner mô phỏng hoặc người dùng tự trả lời với `--interactive`).
- [ ] **Chạy LLM thật**: cần điền `.env`; chưa chạy được với API thật trong môi trường phát triển.

**Interface `ActionExecutor` (đã thống nhất, Thành viên 2 implement):** `execute(decision, learner_state, respond) -> TurnResult`.

## Quyết định đã chốt khi tích hợp (20/10)

- **`learning_rate` của StudentModel = 0.40** (trước là 0.30). Với 0.30 và 5 câu/concept, mastery tối đa chỉ là 0.832 nên ngưỡng 0.85 không bao giờ đạt. Với 0.40: 4 câu đúng liên tiếp → 0.870.
- **Định dạng lịch sử**: `recent_actions` = `ASK:<question_id>:<concept>:correct|wrong` hoặc `<ACTION>:<concept>`; `recent_errors` = `common_error_type` của câu sai. Scenario Phase 5 đã đồng bộ theo định dạng này.
- **Hết câu phù hợp**: action space không đưa ra ASK cho độ khó đã hết câu; nếu executor vẫn nhận ASK như vậy thì trả `valid=False` và engine giữ nguyên state/context.
- **Điều kiện ADVANCE**: do policy quyết định trong action space ở trên (rule-based: mastery ≥ 0.85 hoặc focus hết câu).
- **Concept chưa có trong mastery**: StudentModel coi là `initial_mastery = 0.0`, rule-based dùng `unknown_mastery = 0.0` (khớp nhau); prompt LLM ghi "not assessed".

## Các quyết định còn mở

- Nội dung EXPLAIN/REVIEW hiện là mô tả ngắn của concept trong graph, HINT là `common_error_type`. Nếu thử với người học thật cần nguồn nội dung giảng dạy cố định chi tiết hơn.
- Mỗi concept chỉ có 5 câu: learner yếu dùng hết câu của một concept trước khi đạt 0.85 và buộc phải chuyển tiếp. Nên cân nhắc bổ sung câu (ít nhất cho các concept đầu) trước thí nghiệm chính thức.
- StudentModel không phân biệt độ khó (câu Hard đúng cũng cộng như câu Easy).
- Phân bố đáp án đúng đang lệch (A 9, B 30, C 25, D 6). Cần quyết định: trộn thứ tự option khi hiển thị (Thành viên 2, executor) hay cân lại thứ tự ngay trong CSV.
