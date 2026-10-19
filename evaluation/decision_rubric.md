# Rubric đánh giá quyết định sư phạm (Phase 5)

Mục tiêu: đo **chất lượng quyết định** của Random, Rule-based và LLM policy trên cùng
một bộ learner state mô phỏng (`evaluation/decision_scenarios.json`), độc lập với
student model và người học thật.

## 1. Quy trình

1. Mỗi scenario gồm `learner_state`, `context` (focus concept, câu hỏi gần nhất, action trước)
   và danh sách available actions sinh bởi `action_space.build_available_actions`.
   Xem nhanh danh sách action của từng scenario bằng cách chạy
   `python evaluation/run_decision_evaluation.py --policies rule` rồi mở `results/decision_eval.json`.
2. **Ít nhất 2 annotator gán nhãn độc lập**, không xem quyết định của các policy trước khi gán.
   Người viết luật của rule-based policy không nên là annotator duy nhất, nếu không
   rule-based sẽ được lợi một cách giả tạo.
3. Mỗi annotator thêm một phần tử vào `expert_labels` của scenario:

```json
{
  "annotator": "A1",
  "preferred_actions": [
    {"action_type": "REVIEW_PREREQUISITE", "concept": "Functions", "difficulty": null}
  ],
  "acceptable_actions": [
    {"action_type": "EXPLAIN", "concept": "Recursion", "difficulty": null}
  ],
  "scores": {"remediation": 2, "weak_concept": 2, "difficulty": null, "support_vs_ask": 2, "advance_timing": 2},
  "notes": "Functions 0.45 là nguyên nhân khả dĩ; ôn tiên quyết trước khi hỏi tiếp."
}
```

   - `preferred_actions`: action tốt nhất (có thể nhiều hơn một nếu ngang nhau).
   - `acceptable_actions`: chấp nhận được nhưng không tối ưu.
   - Action phải lấy đúng từ danh sách available actions của scenario đó.
4. Chạy lại `run_decision_evaluation.py`: script tự tính agreement khi có nhãn.
5. Báo cáo thêm độ đồng thuận giữa các annotator (ví dụ Cohen's kappa trên `action_type`)
   trước khi diễn giải kết quả policy.

## 2. Tiêu chí (theo project_0, Phase 5)

Annotator chấm thêm từng tiêu chí cho **quyết định của mỗi policy** (khi đã che tên policy),
thang 0 = sai, 1 = chấp nhận được, 2 = tốt; `null` nếu tiêu chí không áp dụng.

| Mã | Câu hỏi | Gợi ý chấm |
| :--- | :--- | :--- |
| `remediation` | Có cần can thiệp (giải thích, gợi ý, ôn) không, và policy có làm đúng lúc không? | Sau câu sai ở concept yếu mà vẫn hỏi tiếp câu khó → 0 |
| `weak_concept` | Có chọn đúng concept yếu (concept đang học hay tiên quyết)? | Lỗi do tiên quyết yếu mà chỉ giải thích concept hiện tại → 1 |
| `difficulty` | Độ khó có phù hợp với mastery? | Hard khi mastery rất thấp, Easy khi đã thành thạo → 0 |
| `support_vs_ask` | Nên giải thích/gợi ý hay hỏi thêm câu? | EXPLAIN lặp lại ngay sau EXPLAIN → 0 |
| `advance_timing` | Chuyển concept có quá sớm (hoặc quá muộn)? | ADVANCE khi mastery còn thấp và còn câu hỏi → 0 |
| `no_giveaway` | Có làm lộ đáp án không? | Chỉ áp dụng khi chấm nội dung HINT/EXPLAIN của executor (Thành viên 2) |

## 3. Metric do script tính

| Metric | Định nghĩa |
| :--- | :--- |
| `valid_rate` | Tỉ lệ quyết định nằm trong available actions |
| `exact_agreement` | Quyết định trùng một `preferred_action` (trung bình qua annotator) |
| `acceptable_agreement` | Trùng `preferred` hoặc `acceptable` |
| `type_agreement` | Cùng loại action với một `preferred_action` |
| `action_type_distribution` | Phân bố loại action mà policy chọn |
| `flag_rate` | Tỉ lệ cờ heuristic tự động (bên dưới) |

Random policy được chạy nhiều lần (`--random-repeats`, mặc định 20 seed) để ước lượng mức
agreement ngẫu nhiên làm mốc so sánh.

### Cờ heuristic tự động (chỉ để tham khảo)

Các cờ này mang giả định gần với rule-based policy, nên **không dùng làm kết luận chính**.
Ngưỡng ở `evaluation/metrics.py` (`FlagConfig`), cố ý khác ngưỡng của `RuleConfig`.

| Cờ | Điều kiện |
| :--- | :--- |
| `premature_advance` | ADVANCE khi mastery focus < 0.6 và focus còn câu hỏi |
| `too_hard` | ASK Hard khi mastery focus < 0.4 |
| `too_easy` | ASK Easy khi mastery focus ≥ 0.85 |
| `repeated_support` | HINT/EXPLAIN trùng hệt action ngay trước |
| `no_decision` | Policy không đưa ra được quyết định hợp lệ |

## 4. Bộ scenario hiện có

12 scenario trong `decision_scenarios.json`, phủ các tình huống: bắt đầu phiên, concept mới
chưa có dữ liệu, concept yếu do tiên quyết yếu, concept yếu khi tiên quyết vững, mastery trung
bình vừa sai, mastery khá vừa đúng, đã thành thạo, sai lặp cùng một kiểu lỗi, hết câu hỏi,
vừa giải thích xong, đúng liên tiếp nhưng mastery còn thấp, quay lại sau khi ôn tiên quyết.
`expert_labels` đang để trống có chủ đích. Nên bổ sung thêm scenario (khoảng 30–50) trước khi
báo cáo kết quả chính thức.
