# Báo cáo thực hiện công việc — Thành viên 1: Hợp đồng dữ liệu & Question Bank

Tài liệu này giải thích chi tiết toàn bộ công việc, quyết định thiết kế và mã nguồn đã hoàn thành bởi **Thành viên 1** theo phân công trong `TEAM_TASKS.md`. Tài liệu này đóng vai trò là bản hướng dẫn hợp đồng dữ liệu và API cho **Thành viên 2** (Student Model, Action Executor) và **Thành viên 3** (Policies, Engine).

---

## 1. Mục tiêu và Phạm vi thực hiện

- **Mục tiêu:** 
  1. Chốt định dạng chuẩn cho `LearnerState`, `Action`, `PolicyDecision` dùng chung cho toàn bộ hệ thống.
  2. Xây dựng và chuẩn hóa nguồn tri thức: đồ thị tiên quyết `concept_graph.json` và ngân hàng câu hỏi `QuestionBank`.
  3. Cung cấp công cụ xác thực dữ liệu tự động `validate_question_bank.py` và bộ kiểm thử đơn vị `test_question_bank.py`.

- **Phạm vi file phụ trách:**
  - `src/adaptive_learning/schemas.py`
  - `data/concept_graph.json`
  - `src/adaptive_learning/question_bank.py`
  - `scripts/validate_question_bank.py`
  - `tests/test_question_bank.py`
  - `data/question_bank.csv` (đảm bảo tương thích tuyệt đối, không tự ý đổi schema)

---

## 2. Chi tiết công việc đã triển khai

### 2.1. Chuẩn hóa Hợp đồng dữ liệu (`src/adaptive_learning/schemas.py`)

Hợp đồng dữ liệu đã được hoàn thiện với các ràng buộc nghiêm ngặt và tính bất biến (immutability) để tránh side-effect khi các policy xử lý state:

1. **Ràng buộc Mastery `[0.0, 1.0]`**:
   - `LearnerState` kiểm tra từng concept trong trường `mastery`.
   - Bắt buộc là số thực hữu hạn (`float` hoặc `int`, không nhận `bool`). Nếu nằm ngoài đoạn `[0.0, 1.0]` hoặc là `NaN`/`inf`, chương trình sẽ ngay lập tức raise `ValueError`. Nếu sai kiểu dữ liệu sẽ raise `TypeError`.
   - `mastery` được lưu dưới dạng `FrozenMastery` (dict chỉ đọc): `state.mastery["Lists"] = 0.9` sẽ raise `TypeError`. Muốn cập nhật thì tạo `LearnerState` mới (xem ví dụ ở mục 3). Vẫn so sánh, `json.dumps`, `copy.deepcopy`, `pickle` bình thường.
2. **Xử lý danh sách trong State**:
   - Các trường `recent_errors`, `recent_actions`, `attempted_question_ids` trong `LearnerState` được tự động chuyển đổi thành `tuple[str, ...]` (immutable).
   - Cung cấp phương thức `to_dict()` (chuyển tuple thành list để serialize JSON) và `from_dict()` (deserialization từ JSON).
3. **Chuẩn hóa Độ khó (Difficulty)**:
   - Khai báo Enum `Difficulty(str, Enum)` với 3 mức: `Easy`, `Medium`, `Hard`.
   - Tập hợp hợp lệ: `ALLOWED_DIFFICULTIES = frozenset({"Easy", "Medium", "Hard"})`. Bất kỳ giá trị nào ngoài 3 mức này đều bị từ chối.
4. **Chuẩn hóa Action Space**:
   - Enum `ActionType` gồm 5 hành động: `ASK`, `HINT`, `EXPLAIN`, `REVIEW_PREREQUISITE`, `ADVANCE`.
   - `Action` và `PolicyDecision` đều validate `action_type` và `difficulty`.
5. **Mô hình câu hỏi `Question`**:
   - Khớp chính xác 13 cột của `data/question_bank.csv`.
   - **Quyết định về prerequisite**: Giữ nguyên tên trường `prerequisite_concept: str` (dạng số ít khớp với header CSV) để tránh phá vỡ tương thích.
   - Bổ sung property tiện ích:
     - `q.prerequisites`: Tự động parse chuỗi ngăn cách bằng `;` thành `tuple[str, ...]`, loại bỏ giá trị `'None'`. Khi đọc qua `QuestionBank`, các tên này đã được chuẩn hóa về canonical name của graph (ví dụ `"Functions (def, params, return)"` → `"Functions"`), nên dùng trực tiếp làm key tra `mastery` được.
     - `q.options`: Trả về dictionary `{"A": ..., "B": ..., "C": ..., "D": ...}`.

---

### 2.2. Xây dựng Đồ thị Concept (`data/concept_graph.json`)

Được trích xuất và chuẩn hóa từ bảng tổng hợp chương trình học và biểu đồ trong `project/project_1.md`:

- **Quy mô**: Gồm **14 concepts** cốt lõi của môn học Python cơ bản và **21 cạnh có hướng (prerequisite edges)**.
- **Tính chất đồ thị**: Đảm bảo là một Đồ thị có hướng không chu trình (**DAG** - Directed Acyclic Graph) với nút gốc bắt đầu là `Variables & assignment`.
- **Thứ tự topo chuẩn hóa (Topological Order)**:
  `Variables & assignment` $\rightarrow$ `Data types` $\rightarrow$ `Expressions & operators` $\rightarrow$ `Strings` $\rightarrow$ `Boolean expressions` $\rightarrow$ `Functions` $\rightarrow$ `For loops & range` $\rightarrow$ `Conditionals` $\rightarrow$ `While loops` $\rightarrow$ `Scope` $\rightarrow$ `Lists` $\rightarrow$ `Recursion` $\rightarrow$ `Dictionaries` $\rightarrow$ `Tuples & unpacking`.
- **Metadata của từng Concept**:
  - `id`: Mã viết tắt (VAR, DT, EXPR, STR, BOOL, COND, WHILE, FOR, LIST, DICT, TUP, FUNC, SCOPE, REC).
  - `name`: Tên canonical chuẩn.
  - `aliases`: Danh sách tên gọi tương đương trong các giáo trình (ví dụ: `"Functions (def, params, return)"`, `"Data type"`).
  - `prerequisites`: Danh sách concept tiên quyết trực tiếp.
  - `dependents`: Danh sách concept kế tiếp trực tiếp.
  - `assessment_evidence`: Bằng chứng đánh giá (từ tài liệu nghiên cứu).
  - `description`: Mô tả phạm vi kiến thức.

---

### 2.3. Hiện thực QuestionBank và ConceptGraph (`src/adaptive_learning/question_bank.py`)

Cung cấp hai lớp xử lý chính:

#### A. Lớp `ConceptGraph`
- `ConceptGraph.from_file(path)`: Tự động nạp và kiểm tra tính hợp lệ của file JSON.
- `canonicalize(name)`: Ánh xạ tên nhập vào (kể cả case-insensitive hay alias) về đúng canonical concept name.
- `get_prerequisites(concept)`: Trả về tuple các concept tiên quyết trực tiếp.
- `get_ancestors(concept)`: Trả về set toàn bộ các concept tiên quyết cả trực tiếp lẫn bắc cầu (transitive prerequisites).
- `get_dependents(concept)`: Trả về tuple các concept phụ thuộc trực tiếp.
- `is_dag()`: Kiểm tra đồ thị không có chu trình.
- `topological_sort()`: Trả về danh sách concept đã sắp xếp theo thứ tự học hợp lý.

#### B. Lớp `QuestionBank`
- Đọc file cố định `data/question_bank.csv` (70 câu hỏi, mỗi concept có đúng 5 câu gồm 2 Easy, 2 Medium, 1 Hard).
- **Làm sạch dữ liệu tự động**: Tự động loại bỏ ký tự khoảng trắng thừa và ký tự tab trong `correct_answer` (phòng trường hợp dán dữ liệu từ Google Sheets) để đảm bảo không bị lỗi runtime.
- **Chuẩn hóa prerequisite khi load**: `prerequisite_concept` của mỗi câu được map về canonical name qua `ConceptGraph.canonicalize()`.
- **Báo lỗi dữ liệu rõ ràng**: `question_id` trùng lặp → `ValueError`; concept không có trong graph → log cảnh báo; không đọc được graph → log cảnh báo (thay vì im lặng).
- **Tạo chỉ mục tra cứu**: Lập index theo `question_id`, `concept`, `difficulty`, và bộ đôi `(concept, difficulty)` giúp việc tra cứu đạt độ phức tạp $O(1)$.
- **Các hàm API phục vụ Thành viên 2 và 3**:
  - `get_question(question_id) -> Question | None`: Lấy chi tiết câu hỏi theo ID.
  - `get_questions(concept=None, difficulty=None, exclude_ids=None) -> list[Question]`: Lọc câu hỏi theo concept (hỗ trợ cả alias), theo độ khó và tự động loại bỏ các câu hỏi đã làm (`exclude_ids`).
  - `get_available_questions(concept=None, difficulty=None, attempted_ids=None)`: Hàm tiện ích bọc cho `get_questions`.
  - `get_next_question(concept, difficulty, exclude_ids, strategy="first" | "random", rng=None)`: Lấy câu hỏi tiếp theo chưa làm. Với `strategy="random"`, truyền `rng=random.Random(seed)` để thí nghiệm tái lập được.
  - `concept_summary() -> dict`: Thống kê tổng số câu hỏi theo từng độ khó cho từng concept.

---

### 2.4. Công cụ Xác thực Dữ liệu (`scripts/validate_question_bank.py`)

Script này chạy kiểm tra tự động trước khi chạy thực nghiệm:
- **Kiểm tra đồ thị**: Xác thực định dạng JSON, kiểm tra tính chất DAG (báo lỗi nếu có chu trình), kiểm tra liên kết 2 chiều giữa prerequisites và dependents.
- **Kiểm tra CSV**:
  - Kiểm tra đầy đủ 13 cột header.
  - Kiểm tra tính duy nhất của `question_id`.
  - Kiểm tra `concept` và `prerequisite_concept` phải thuộc danh sách concept hoặc alias hợp lệ.
  - Kiểm tra `difficulty` $\in$ `{Easy, Medium, Hard}`.
  - Kiểm tra `correct_answer` $\in$ `{A, B, C, D}`.
  - Kiểm tra không để trống câu hỏi, 4 phương án và phần giải thích; 4 phương án không được trùng nhau.
  - Kiểm tra prerequisite của câu hỏi là tổ tiên của concept trong graph và không trỏ vào chính concept đó.
  - Compile đoạn code trong câu hỏi (phần sau dòng đầu) để bắt lỗi thụt lề/cú pháp.
- **Báo cáo trực quan**: Bảng phân bố câu hỏi Easy/Medium/Hard/Total cho 14 concept.
- **Cờ tính năng**:
  - `--strict`: Coi warning như lỗi (trả về exit code 1 nếu có warning).
  - `--fix`: Tự động chuẩn hóa dữ liệu whitespace và alias trong CSV mà không làm thay đổi cấu trúc bảng.

---

### 2.5. Kiểm thử Đơn vị (`tests/test_question_bank.py`)

Toàn bộ logic đã được kiểm thử với 30 test cases. 15 test ban đầu gồm:
1. `test_learner_state_valid`: Kiểm tra tạo state hợp lệ.
2. `test_learner_state_mastery_bounds`: Kiểm tra chặn mastery $< 0.0$, $> 1.0$ hoặc sai type.
3. `test_learner_state_serialization`: Kiểm tra tính toàn vẹn khi serialize và deserialize sang JSON dictionary.
4. `test_action_and_policy_decision`: Kiểm tra tạo và xác thực Action, PolicyDecision.
5. `test_question_schema`: Kiểm tra tạo Question, thuộc tính `options`, tách `prerequisites`.
6. `test_graph_loaded_correctly`: Kiểm tra nạp đủ 14 concept và tính chất DAG.
7. `test_canonicalize_aliases`: Kiểm tra chuyển đổi các alias về canonical concept name.
8. `test_prerequisites_and_dependents`: Kiểm tra các cạnh phụ thuộc cụ thể (như Recursion phụ thuộc Conditionals, Functions, Scope).
9. `test_topological_sort`: Kiểm tra thứ tự sắp xếp topo của toàn bộ đồ thị.
10. `test_total_questions_and_concepts`: Kiểm tra QuestionBank nạp đủ 70 câu và 14 concepts.
11. `test_difficulty_distribution_per_concept`: Kiểm tra mỗi concept có đúng 2 Easy, 2 Medium, 1 Hard.
12. `test_get_question_by_id`: Kiểm tra lấy câu hỏi theo ID và làm sạch tab `\tA`.
13. `test_filter_by_concept_and_difficulty`: Kiểm tra lọc theo concept và difficulty.
14. `test_exclude_attempted_questions`: Kiểm tra cơ chế loại trừ câu hỏi đã làm.
15. `test_get_next_question`: Kiểm tra lấy câu hỏi tiếp theo chưa làm.

15 test bổ sung trong đợt rà soát:
- `TestLearnerStateImmutability` (4 test): mastery chỉ đọc, cập nhật bằng state mới, chặn `NaN`/`inf`/`bool`, copy/pickle/hash.
- `TestQuestionBankDataIntegrity` (5 test): prerequisite là canonical name và là tổ tiên trong graph, 4 option khác nhau, code trong mọi câu compile được, chọn ngẫu nhiên tái lập được với seed.
- `TestValidatorDetectsDataErrors` (5 test): validator bắt được option trùng, prerequisite tự trỏ, prerequisite không phải tổ tiên, code thụt lề hỏng.
- `TestQuestionBankLoadErrors` (1 test): `question_id` trùng lặp bị báo lỗi.

### 2.6. Rà soát và sửa dữ liệu `data/question_bank.csv` (18/10)

Chạy thật đoạn code của cả 70 câu và đối chiếu output với đáp án. Các sửa đổi:

| Nhóm lỗi | Câu | Sửa |
| :--- | :--- | :--- |
| Sai đáp án | FOR_002 | `range(2, 8, 2)` lặp 3 lần → đáp án B; đổi option D thành 6 cho khớp mapping |
| Code không chạy (thụt lề) | REC_001, REC_003, REC_004, REC_005 | Sửa thụt lề; chuẩn hóa thụt lề 4 khoảng trắng cho LIST/TUP/DICT |
| Option trùng | STR_005 | Excel ẩn dấu `'` đầu ô → đổi `repr(slice1)` thành `len(slice1)` |
| Typo | CON_003, EXP_004 | `TRUE` → `True`; `Data type` → `Data types` |
| Prerequisite mâu thuẫn graph | VAR_002–VAR_005, TYP_005, WHI_004, SCO_005 | Sửa prerequisite; WHI_004 và SCO_005 viết lại code bỏ `if` |
| Câu đặt sai concept | LIST_002, TUP_002, DICT_002, REC_002 | Viết lại để thật sự kiểm tra Lists / Tuples / Dictionaries / Recursion |
| Mapping không khớp option | VAR_004, EXP_001, EXP_004, STR_003, FUN_005 | Đổi distractor hoặc mapping cho khớp lỗi mô tả |
| Đáp án có hai cách hiểu | SCO_005 | `UnboundLocalError` là lớp con của `NameError` → bỏ option NameError |
| Option đúng dài bất thường | SCO_002, REC_001, REC_005 | Rút gọn option đúng |
| Kiến thức | TYP_001 | `//` là floor (làm tròn về −∞), không phải truncation |
| Định dạng | VAR_004, TYP_004 | Xóa tab trong `correct_answer`; prerequisite dùng canonical name |

---

## 3. Hướng dẫn sử dụng API cho Thành viên 2 và Thành viên 3

### Dành cho Thành viên 2 (Student Model & Action Executor):
```python
from adaptive_learning.schemas import LearnerState, Action, ActionType, Difficulty
from adaptive_learning.question_bank import QuestionBank

qb = QuestionBank()

# 1. Khi Executor nhận action ASK, lấy câu hỏi tương ứng:
available = qb.get_available_questions(
    concept="Variables & assignment",
    difficulty="Easy",
    attempted_ids=learner_state.attempted_question_ids
)
if available:
    question = available[0]
    # Lấy đáp án đúng để chấm điểm: question.correct_answer (ví dụ: 'B')
    # Lấy giải thích: question.explanation
    # Lấy loại lỗi thường gặp: question.common_error_type

# 2. Khi cần review kiến thức tiên quyết (REVIEW_PREREQUISITE):
prereqs = qb.concept_graph.get_prerequisites("Recursion")
# -> ('Conditionals', 'Functions', 'Scope')

# 3. Khi cập nhật LearnerState:
new_state = LearnerState(
    mastery={**learner_state.mastery, "Variables & assignment": 0.85},
    recent_errors=tuple(updated_errors),
    recent_actions=tuple(updated_actions),
    attempted_question_ids=tuple(list(learner_state.attempted_question_ids) + [question.question_id]),
)
```

### Dành cho Thành viên 3 (Policies & Learning Engine):
```python
from adaptive_learning.schemas import Action, ActionType, PolicyDecision, LearnerState
from adaptive_learning.question_bank import QuestionBank

qb = QuestionBank()

# 1. Engine tạo tập hợp available_actions hợp lệ gửi cho Policy:
available_actions = []
for concept in qb.get_all_concepts():
    for diff in ["Easy", "Medium", "Hard"]:
        if qb.get_available_questions(concept=concept, difficulty=diff, attempted_ids=state.attempted_question_ids):
            available_actions.append(Action(action_type=ActionType.ASK, concept=concept, difficulty=diff))

# 2. Policy chọn 1 action và trả về PolicyDecision:
# decision = policy.select_action(state, tuple(available_actions))
# Engine kiểm tra tính hợp lệ:
is_valid = any(
    a.action_type == decision.action_type and a.concept == decision.concept and a.difficulty == decision.difficulty
    for a in available_actions
)
```

---

## 4. Các lệnh chạy xác thực và kiểm thử

1. **Chạy kiểm thử đơn vị:**
   ```bash
   python -m unittest tests/test_question_bank.py
   ```
   *Kết quả kỳ vọng:* `Ran 30 tests ... OK`

2. **Chạy script kiểm tra Question Bank và Concept Graph:**
   ```bash
   python scripts/validate_question_bank.py
   ```
   *Kết quả kỳ vọng:* Báo cáo bảng 14 concepts với 70 câu hỏi và kết luận `HỢP ĐỒNG DỮ LIỆU ĐẠT CHUẨN (PASSED)`. Chạy thêm `--strict` cũng phải PASSED (không còn cảnh báo).

3. **Chạy tự động chuẩn hóa dữ liệu CSV (nếu muốn làm sạch khoảng trắng):**
   ```bash
   python scripts/validate_question_bank.py --fix
   ```

---

## 5. Tổng kết trạng thái bàn giao

Dưới đây là danh sách đối chiếu toàn bộ các đầu việc đã hoàn thành so với yêu cầu ban đầu trong `TEAM_TASKS.md`:

- [x] **1. Giữ một định dạng state và action duy nhất cho cả ba policy**
  - *Trạng thái:* Đã xong
  - *Ghi chú:* Hoàn thành trong `schemas.py` (`LearnerState`, `Action`, `PolicyDecision`).

- [x] **2. Chốt ràng buộc mastery [0, 1], difficulty (Easy, Medium, Hard)**
  - *Trạng thái:* Đã xong
  - *Ghi chú:* Validate tự động bằng Enum và post-init checks trong `schemas.py`.

- [x] **3. Viết QuestionBank đọc CSV cố định, hỗ trợ lọc theo concept/difficulty và loại trừ câu đã làm**
  - *Trạng thái:* Đã xong
  - *Ghi chú:* Hoàn thành trong `question_bank.py` (`QuestionBank`).

- [x] **4. Quyết định biểu diễn prerequisite (`prerequisite_concept`)**
  - *Trạng thái:* Đã xong
  - *Ghi chú:* Giữ nguyên tên cột CSV, bổ sung property `prerequisites` (tuple) trong `schemas.py`.

- [x] **5. Đồ thị Concept và kiểm tra khớp nối với CSV**
  - *Trạng thái:* Đã xong
  - *Ghi chú:* Hoàn thành `concept_graph.json` (14 concepts, 21 edges) & class `ConceptGraph`.

- [x] **6. Script kiểm tra báo lỗi dữ liệu có thể sửa được**
  - *Trạng thái:* Đã xong
  - *Ghi chú:* Hoàn thành `validate_question_bank.py` hỗ trợ cả cờ tự sửa `--fix`.

- [x] **7. Bộ unit tests kiểm chứng hệ thống**
  - *Trạng thái:* Đã xong
  - *Ghi chú:* Hoàn thành `test_question_bank.py` với 30/30 tests PASS.
