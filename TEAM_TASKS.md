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

## Các quyết định còn mở cần ghi lại

- `data/concept_graph.json` hiện rỗng; chọn graph trong `project/project_1.md` làm nguồn để điền hay chọn một nguồn chuẩn khác.
- CSV hiện có cột `prerequisite_concept` dạng số ít; giữ tên này trong code trừ khi nhóm thống nhất migration.
- CSV có `explanation` cho câu hỏi nhưng chưa có nguồn riêng cho hint/giải thích concept.
- Chốt khi nào cho phép hỏi lại, hành vi khi hết câu phù hợp, và điều kiện `ADVANCE`.
- `project_1.md` có một số concept dạng đầy đủ hơn tên trong CSV (ví dụ `Functions (def, params, return)` và `Functions`); phải chọn canonical concept ID/name trước khi kiểm tra graph.
