# Báo cáo thực hiện công việc — Thành viên 2: Student Model & Action Executor

Tài liệu này mô tả phần triển khai của **Thành viên 2** theo phân công trong
[TEAM_TASKS.md](TEAM_TASKS.md). Phần này sử dụng trực tiếp các hợp đồng dữ liệu
đã chốt trong `schemas.py` và API của `QuestionBank` do Thành viên 1 bàn giao.

---

## 1. Mục tiêu và phạm vi thực hiện

Mục tiêu:

1. Xây dựng một student model xác định, có cùng công thức cho mọi policy.
2. Chấm đáp án bằng answer key cố định trong QuestionBank.
3. Cập nhật learner state mà không làm mutate state cũ.
4. Cung cấp executor tương thích với interface mà `LearningEngine` yêu cầu.
5. Kiểm thử các trường hợp đúng, sai, mastery biên và câu hỏi lặp.

Các file triển khai:

- `src/adaptive_learning/student_model.py`
- `src/adaptive_learning/action_executor.py`
- `tests/test_student_model.py`
- `tests/test_action_executor.py`

Ngoài ra, file báo cáo này được thêm để bàn giao quyết định thiết kế và cách
sử dụng cho Thành viên 3.

---

## 2. StudentModel

### 2.1. Công thức cập nhật mastery

Student model dùng exponential moving average với tham số mặc định:

- `learning_rate = 0.30`
- `initial_mastery = 0.0`
- `history_limit = 10`

Công thức:

```text
outcome = 1.0 nếu đúng, 0.0 nếu sai
new_mastery = old_mastery + learning_rate * (outcome - old_mastery)
```

Ví dụ:

- Mastery `0.0`, trả lời đúng: `0.0 + 0.3 * (1.0 - 0.0) = 0.3`.
- Mastery `1.0`, trả lời sai: `1.0 + 0.3 * (0.0 - 1.0) = 0.7`.
- Mastery luôn được giới hạn trong `[0.0, 1.0]`.

Concept chưa có trong `LearnerState.mastery` được khởi tạo bằng
`initial_mastery`.

### 2.2. Tính bất biến

`StudentModel` không sửa trực tiếp `LearnerState` hiện tại. Phương thức
`update_state()` tạo một `LearnerState` mới, phù hợp với contract immutable của
`schemas.py`.

Các cấu hình không hợp lệ đều bị từ chối:

- `learning_rate` phải thuộc `(0, 1]`.
- `initial_mastery` phải thuộc `[0, 1]`.
- Hai giá trị trên phải hữu hạn và không được là `bool`.
- `history_limit` phải là số nguyên dương.

---

## 3. Chấm đáp án và cập nhật state

### 3.1. Chấm đáp án

`StudentModel.grade_answer(question, answer)`:

- So sánh trực tiếp với `question.correct_answer`.
- Bỏ qua khoảng trắng đầu/cuối.
- Không phân biệt chữ hoa/chữ thường.
- `None` hoặc đáp án không khớp được xem là sai.
- Không để policy hoặc LLM tự chấm đáp án.

### 3.2. Cập nhật các trường lịch sử

Sau một câu hỏi:

- `mastery[question.concept]` được cập nhật theo công thức EMA.
- `question_id` được thêm vào `attempted_question_ids`.
- Nếu question ID đã có, không thêm trùng.
- `recent_actions` ghi theo định dạng:

```text
ASK:<question_id>:<concept>:correct
ASK:<question_id>:<concept>:wrong
```

- Nếu trả lời sai, `question.common_error_type` được thêm vào `recent_errors`.
- Nếu `common_error_type` rỗng, dùng giá trị `incorrect_answer`.
- `recent_actions` và `recent_errors` được giới hạn bởi `history_limit`.

---

## 4. ActionExecutor

### 4.1. Interface

Executor cung cấp method:

```python
execute(
    decision: PolicyDecision,
    learner_state: LearnerState,
    respond: Callable[[Question], Any] | None = None,
) -> TurnResult
```

Với action `ASK`, `respond` phải được truyền vào. Nếu thiếu responder, executor
trả `valid=False`, giữ nguyên learner state và không dùng answer key làm câu trả
lời giả định. `respond=None` chỉ có ý nghĩa khi action không cần câu trả lời.

Đây là interface tương thích với `LearningEngine`.

Có thể khởi tạo:

```python
from adaptive_learning.action_executor import ActionExecutor
from adaptive_learning.question_bank import QuestionBank

executor = ActionExecutor(QuestionBank())
```

### 4.2. ASK

Với `ASK`:

1. Lấy câu hỏi từ QuestionBank theo `concept` và `difficulty`.
2. Loại bỏ các câu đã có trong `attempted_question_ids`.
3. Dùng chiến lược `first` mặc định để tái lập kết quả.
4. Có thể dùng `question_strategy="random"` và truyền `random.Random(seed)`.
5. Gọi `respond(question)` để nhận đáp án.
6. Chấm bằng `question.correct_answer`.
7. Gọi StudentModel để tạo state mới.
8. Trả về `TurnResult` gồm:
   - `question_id`
   - `question`
   - `correct`
   - `feedback`
   - `updated_state`

Nếu không còn câu phù hợp, executor trả về `valid=False`, giữ nguyên state và
điền thông báo lỗi rõ ràng.

### 4.3. HINT, EXPLAIN, REVIEW_PREREQUISITE và ADVANCE

- `HINT`: lấy câu hỏi gần nhất của concept từ lịch sử và tạo feedback dựa trên
  `common_error_type`; không tiết lộ answer key.
- `EXPLAIN`: lấy mô tả concept cố định từ ConceptGraph.
- `REVIEW_PREREQUISITE`: lấy mô tả prerequisite từ ConceptGraph.
- `ADVANCE`: ghi nhận action chuyển concept, không thay đổi mastery.

Các action không phải `ASK` không tự tạo câu hỏi và không tự chấm đáp án.
Chúng chỉ cập nhật `recent_actions`, bảo toàn mastery và danh sách câu đã làm.

### 4.4. Validation

Executor kiểm tra trực tiếp:

- `decision` phải là `PolicyDecision`.
- Mọi action phải có `concept`.
- `ASK` phải có `difficulty`.

Engine vẫn là nơi chịu trách nhiệm chính để kiểm tra decision có thuộc
`available_actions` hay không trước khi gọi executor.

---

## 5. Kiểm thử

### 5.1. Test StudentModel

Đã kiểm thử:

1. Chấm đáp án đúng/sai và chuẩn hóa hoa thường.
2. Công thức EMA.
3. Mastery tại biên `0.0` và `1.0`.
4. Cập nhật mastery sau câu đúng.
5. Ghi nhận lỗi sau câu sai.
6. Giới hạn lịch sử.
7. Không thêm trùng câu hỏi.
8. Từ chối cấu hình không hợp lệ.

### 5.2. Test ActionExecutor

Đã kiểm thử:

1. Lấy câu hỏi từ QuestionBank và chấm đúng.
2. Chấm sai dựa trên answer key.
3. Loại trừ câu hỏi đã làm.
4. Xử lý khi hết câu hỏi phù hợp.
5. Từ chối `ASK` thiếu responder mà không giả định learner trả lời đúng hoặc đổi state.
6. Bảo toàn mastery với action `EXPLAIN`.
7. Ghi nhận action hỗ trợ vào lịch sử.
8. Từ chối `ASK` thiếu difficulty.

### 5.3. Kết quả chạy test

Đã chạy toàn bộ test suite bằng:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Kết quả:

```text
Ran 99 tests
OK
```

Đã chạy thêm compile check:

```bash
python -m compileall -q src tests
```

Kết quả compile thành công.

---

## 6. Hướng dẫn tích hợp cho Thành viên 3

Engine có thể dùng executor thật như sau:

```python
from adaptive_learning.action_executor import ActionExecutor
from adaptive_learning.engine import LearningEngine
from adaptive_learning.question_bank import QuestionBank

question_bank = QuestionBank()
executor = ActionExecutor(question_bank)
engine = LearningEngine(policy, question_bank, executor)

result = engine.run_turn(
    learner_state,
    respond=lambda question: learner.answer(question),
)
```

Responder chỉ cần trả về lựa chọn của learner, ví dụ `"A"`, `"B"`, `"C"` hoặc
`"D"`. Việc chọn câu, chấm đáp án và cập nhật mastery do Python executor và
StudentModel đảm nhiệm.

---

## 7. Trạng thái bàn giao

- [x] Chốt công thức mastery và tham số mặc định.
- [x] Chấm đáp án bằng `correct_answer` của QuestionBank.
- [x] Cập nhật `mastery`.
- [x] Cập nhật `recent_errors`.
- [x] Cập nhật `recent_actions`.
- [x] Cập nhật `attempted_question_ids` và loại trừ câu hỏi lặp.
- [x] Triển khai `ASK`, `HINT`, `EXPLAIN`, `REVIEW_PREREQUISITE`, `ADVANCE`.
- [x] Trả về `TurnResult` tương thích với engine.
- [x] Viết test cho các trường hợp đúng, sai, biên mastery và câu hỏi lặp.
- [x] Chạy toàn bộ 99 test thành công sau rà soát.

Các quyết định còn cần nhóm cân nhắc ở giai đoạn tích hợp:

- Có muốn thay `history_limit=10` bằng một giá trị cấu hình trong
  `experiment_config.json` hay không.
- Có muốn sử dụng `question_strategy="random"` cho thực nghiệm hay giữ
  `"first"` để dễ tái lập.
- Nội dung hint hiện dùng `common_error_type`; nếu cần hint chi tiết hơn, nhóm
  nên bổ sung nguồn nội dung cố định riêng thay vì để policy hoặc LLM tự tạo.

### 8.1. Đối chiếu yêu cầu và điểm cần nhóm tích hợp

- **Đã đáp ứng trong code:** công thức EMA và tham số được cố định; answer key
  chỉ dùng để chấm; state mới được tạo bất biến; lịch sử có giới hạn; câu đã làm
  bị loại khỏi lần chọn tiếp theo; action hỗ trợ không thay đổi mastery.
- **Hợp đồng `recent_actions`:** câu hỏi được ghi là
  `ASK:<question_id>:<concept>:correct|wrong`; action không hỏi ghi
  `<ACTION>:<concept>`. `recent_errors` chỉ thêm `common_error_type` khi trả lời
  sai. Các fixture trong `evaluation/decision_scenarios.json` vẫn mô tả format
  provisional khác; Thành viên 3 cần đồng bộ fixture/tài liệu của mình nếu các
  phần đó dùng trường lịch sử này để đánh giá.
- **Nội dung:** `ASK` trả explanation của câu sau khi đã chấm; `EXPLAIN` và
  `REVIEW_PREREQUISITE` đọc description cố định từ concept graph; `HINT` dùng
  `common_error_type` của câu gần nhất thuộc concept và không đọc answer key.
  Graph chỉ có mô tả khái quát, vì vậy explanation concept có thể cần nguồn nội
  dung giảng dạy chi tiết hơn nếu sản phẩm yêu cầu.
- **Không tự trộn lựa chọn:** executor giữ nguyên nhãn A–D trong CSV để
  `correct_answer` và `option_error_mapping` luôn trỏ đúng lựa chọn. Phân bố đáp
  án đúng lệch trong dữ liệu vẫn là quyết định cần nhóm thống nhất; nếu muốn
  xáo trộn, phải tạo Question hiển thị mới đồng thời remap answer key và mapping.
- **Phase 6:** ghi chú đầu
  `evaluation/run_learning_experiment.py` còn nói StudentModel/ActionExecutor
  đang thiếu. Hai thành phần đã được triển khai; phần learner mô phỏng và
  pre/post-test vẫn chưa có. Thành viên 3 nên cập nhật trạng thái phụ thuộc trong
  file thuộc phạm vi của họ trước khi chạy experiment.

### 8.2. Xác minh sau khi rà soát

Đã chạy lại toàn bộ test hiện có sau khi bổ sung kiểm tra responder:

```text
Ran 99 tests
OK
```
