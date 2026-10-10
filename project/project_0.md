# Project 0 — Adaptive Learning System using LLM Agent

## 1. Nội dung đề xuất cho đề tài

### Định hướng thu hẹp đề tài

Đề xuất ban đầu có khá nhiều thành phần có thể mở rộng rất nhanh:

- Knowledge tracing
- Question generation
- Error detection
- Difficulty estimation
- LLM agent
- Personalized explanation
- Learning experiment

Nếu triển khai đồng thời tất cả các phần trên, project rất dễ trở thành một software project rộng nhưng research nông.

Vì vậy, core research problem nên được thu hẹp thành:

\[
\textbf{Student Modeling}
\rightarrow
\textbf{Pedagogical Decision}
\rightarrow
\textbf{Learning Outcome}
\]

Trong đó, phần được thay đổi chính là **Pedagogical Decision** bằng LLM Agent, còn các thành phần khác nên được giữ cố định để tạo một controlled experiment rõ ràng:

- Student model: cố định
- Question bank: cố định
- Domain: cố định
- Action space: cố định

Sau đó so sánh ba chiến lược:

\[
\text{Random}
\quad vs \quad
\text{Rule-based}
\quad vs \quad
\text{LLM Agent}
\]

Mục tiêu không phải chỉ là chứng minh rằng LLM có thể làm một adaptive quiz, mà là kiểm tra xem LLM Agent có thực sự đưa ra **pedagogical decisions tốt hơn** các chiến lược truyền thống hay không.

### Thesis statement đề xuất

> **This project investigates whether an LLM agent can improve adaptive learning by making sequential pedagogical decisions based on an explicit model of a learner's evolving knowledge state.**

### Research Question chính

> **Given the same learner state and available pedagogical actions, can an LLM-based policy outperform random and rule-based adaptive strategies in selecting the next instructional action?**

### Research Question phụ

> **Does the resulting policy lead to higher learning gains or faster mastery for real learners?**

Ngay cả khi Research Question phụ cho kết quả negative, project vẫn có scientific value nếu nhóm có thể phân tích:

- Khi nào LLM policy tốt hơn rule-based policy
- Khi nào LLM policy kém hơn
- Những loại pedagogical errors mà LLM thường mắc
- Liệu sự linh hoạt của LLM có thực sự cần thiết hay một số simple rules đã đủ tốt

---

## 2. Lộ trình đề xuất

### Phase 1 — Define the learning domain

Đây là việc cần làm trước khi xây dựng LLM Agent.

Không nên làm một hệ thống “học bất cứ thứ gì”. Nhóm nên chọn một domain nhỏ nhưng có cấu trúc rõ ràng, ví dụ:

- Introductory Python
- Propositional logic
- Basic SQL
- Elementary algebra
- Probability fundamentals

Ví dụ với Python, các concept có thể có prerequisite relationship:

```text
Variables
   ↓
Expressions
   ↓
Conditionals
   ↓
Loops
   ↓
Functions
   ↓
Recursion
```

Có thể biểu diễn các concept thành một **knowledge graph / prerequisite graph**.

Điều này quan trọng vì Agent cần phân biệt liệu learner yếu ở một concept trực tiếp hay do prerequisite chưa vững.

---

### Phase 2 — Build question bank

Mỗi question nên có metadata tối thiểu:

```text
question_id
concept
prerequisite_concepts
difficulty
correct_answer
common_error_type
explanation
```

Trong giai đoạn đầu, không cần dùng LLM để generate questions.

Nên bắt đầu bằng một fixed question bank để đảm bảo experiment có tính reproducible.

Sau đó, nếu còn thời gian, nhóm có thể mở rộng bằng LLM-generated variants.

---

### Phase 3 — Build the simplest Student Model

Ở version đầu tiên, nhóm có thể sử dụng một mô hình đơn giản để ước lượng mastery, ví dụ:

\[
M_k =
\frac{\text{weighted correct answers}}
{\text{weighted attempts}}
\]

hoặc exponential update.

Sau đó có thể chuyển sang **Bayesian Knowledge Tracing (BKT)**.

Ví dụ learner state:

\[
S_t =
[0.91,0.83,0.34,0.68,...]
\]

tương ứng:

```text
variables      0.91
conditionals   0.83
loops          0.34
functions      0.68
```

BKT phù hợp với project này vì interpretable.

Nếu còn thời gian, nhóm mới thử **Deep Knowledge Tracing (DKT)**.

DKT có thể mạnh hơn nhưng cũng làm project trở thành một bài toán ML lớn riêng và khiến state khó giải thích hơn cho Agent.

---

### Phase 4 — Implement baseline policies

Cần có ít nhất ba policy để so sánh.

#### Baseline A — Random / Fixed Quiz

Không có personalization.

#### Baseline B — Rule-based Adaptive

Ví dụ:

```text
mastery < 0.4
→ easier question / explanation

0.4 ≤ mastery < 0.75
→ same concept

mastery ≥ 0.75
→ harder question

mastery ≥ 0.85
→ advance
```

Đây là baseline quan trọng vì nó đại diện cho adaptive learning bằng các rule đơn giản và dễ diễn giải.

#### Policy C — LLM Agent

LLM Agent nhận vào learner state có cấu trúc rõ ràng, ví dụ:

```text
Learning goal:
Master all 12 concepts.

Current learner state:
Loops:             0.82
Functions:         0.74
Recursion:         0.31
Recursive tracing: 0.22

Recent mistakes:
- confused recursive return with print
- forgot base case twice

Recent actions:
ASK recursion difficulty=2 → wrong
EXPLAIN base case
ASK recursion difficulty=1 → correct

Available actions:
ASK
HINT
EXPLAIN
REVIEW_PREREQUISITE
ADVANCE
```

Agent nên output structured action, ví dụ:

```json
{
  "action": "REVIEW_PREREQUISITE",
  "concept": "function_return",
  "difficulty": 2
}
```

Trong architecture này, Agent là **decision-maker**, không phải là một chatbot tự làm tất cả mọi thứ.

---

### Phase 5 — Evaluate Agent decisions

Không nên chỉ đánh giá LLM bằng việc nó trả lời đúng hay sai.

Cần đánh giá chất lượng pedagogical decision.

Một số tiêu chí có thể dùng:

```text
Was remediation needed?
Did it select the correct weak concept?
Was difficulty appropriate?
Should it explain or ask another question?
Did it advance too early?
Did it give away the answer?
```

Nhóm có thể tạo một dataset gồm các simulated learner states rồi nhờ expert annotate preferred action.

Đây có thể trở thành một experiment độc lập để đánh giá chất lượng decision policy của Agent.

---

### Phase 6 — Evaluate learning effectiveness

Cuối cùng, nếu có đủ nguồn lực, tiến hành human experiment:

```text
Pre-test
   ↓
Random/fixed quiz  ─────────┐
Rule adaptive     ──────────┼──→ Post-test
LLM adaptive      ──────────┘
```

Primary metric:

\[
LearningGain = PostTest - PreTest
\]

Có thể bổ sung:

\[
Efficiency =
\frac{LearningGain}{\#Questions}
\]

và các metric khác như:

```text
time to mastery
number of questions to mastery
error recurrence
retention test after N days
```

Lưu ý: historical datasets có thể giúp đánh giá knowledge tracing và response prediction, nhưng không thể tự chứng minh rằng một adaptive policy mới sẽ giúp real learners học tốt hơn.

Nếu không đủ điều kiện làm human experiment, claim của project nên giới hạn ở:

> **LLM Agent produces more appropriate adaptive instructional decisions.**

thay vì khẳng định:

> **LLM Agent improves learning.**

---

## 3. Gợi ý dataset có thể dùng

### ASSISTments

ASSISTments là một trong những dataset phổ biến nhất trong Knowledge Tracing, đặc biệt trong domain mathematics.

Có thể dùng dataset này để:

- Thử nghiệm Bayesian Knowledge Tracing
- Thử nghiệm Deep Knowledge Tracing
- Predict learner responses
- Estimate learner mastery state
- Benchmark student modeling approaches

Các phiên bản phổ biến gồm:

- ASSISTments 2009
- ASSISTments 2012

Dataset này phù hợp khi nhóm muốn bắt đầu từ một bài toán Knowledge Tracing tương đối chuẩn.

---

### EdNet

EdNet là một educational interaction dataset rất lớn, được xây dựng từ nền tảng TOEIC tutoring Santa.

Dataset chứa lượng lớn learner interactions và nhiều dạng learner activity hơn so với các dataset Knowledge Tracing truyền thống.

Có thể dùng EdNet để:

- Train / evaluate Knowledge Tracing models
- Study learner behavior sequences
- Analyze long-term interaction history
- Develop richer student-state representations

EdNet đặc biệt hữu ích nếu nhóm muốn thử những model phức tạp hơn như DKT hoặc sequence-based student modeling.

---

### Cách sử dụng dataset trong project

ASSISTments và EdNet nên chủ yếu được dùng để:

\[
\boxed{\text{learn and validate Knowledge Tracing}}
\]

Không nhất thiết phải xây toàn bộ application dựa trên domain của dataset.

Nhóm hoàn toàn có thể:

1. Dùng ASSISTments / EdNet để validate student-modeling component.
2. Sau đó xây một domain nhỏ riêng cho adaptive learning application và LLM Agent experiment.

---

## 4. Gợi ý hướng và tài liệu nghiên cứu có liên quan

Có thể nhìn literature của project theo knowledge tree:

```text
                    Adaptive Learning
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
   Student Model     Decision Policy      Pedagogy
         │                 │                  │
 Knowledge Tracing     RL / Agent       Scaffolding
 BKT / DKT / IRT       Planning         Mastery
         │                 │                  │
         └─────────────────┼──────────────────┘
                           │
                       LLM Tutor
```

### Hướng 1 — Knowledge Tracing / Student Modeling

#### Corbett & Anderson (1995)
**Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge**

Đây là tài liệu nền tảng về Bayesian Knowledge Tracing.

Nên đọc để hiểu:

- Learner knowledge được biểu diễn như thế nào
- Latent knowledge state
- Mastery probability
- Update learner state từ correct / incorrect answers

---

#### Piech et al. (2015)
**Deep Knowledge Tracing**

Đây là công trình nổi tiếng đưa deep learning vào Knowledge Tracing.

Nên đọc để hiểu:

- Modeling learner interactions bằng sequence models
- RNN-based student modeling
- Sự khác biệt giữa explicit probabilistic state và learned latent representation

---

#### Knowledge Tracing: A Survey — ACM Computing Surveys

Survey này phù hợp để nhóm có cái nhìn tổng quan về:

- Classical Knowledge Tracing
- Bayesian Knowledge Tracing
- Deep Knowledge Tracing
- Transformer-based Knowledge Tracing
- Datasets
- Evaluation metrics

Đây là tài liệu rất nên đọc ở giai đoạn literature review.

---

### Hướng 2 — Adaptive Learning as Sequential Decision Making

#### Reinforcement Learning in Education: A Systematic Literature Review

Survey này rất quan trọng đối với phần Agent.

Nên đọc để hiểu các bài toán như:

- Curriculum sequencing
- Adaptive difficulty
- Personalized feedback
- Learning-path recommendation
- Sequential instructional decisions

Project có thể được nhìn dưới góc độ một sequential decision-making problem:

```text
Learner state
    ↓
Agent selects pedagogical action
    ↓
Learner responds
    ↓
New learner state
```

---

### Hướng 3 — LLM for Education / LLM Tutor

#### LearnLM: Improving Gemini for Learning

Tài liệu này đáng đọc để hiểu rằng:

> Một model trả lời đúng chưa chắc là một tutor tốt.

Nhóm nên quan tâm tới pedagogical behavior thay vì chỉ content correctness.

Các khía cạnh liên quan gồm:

- Scaffolding
- Guiding instead of giving answers
- Adapting explanations
- Supporting active learning
- Pedagogical instruction following

---

#### Maurya et al. — Unifying AI Tutor Evaluation

Tài liệu này hữu ích cho việc xây evaluation framework.

Nó giúp nhóm suy nghĩ về việc đánh giá AI Tutor theo nhiều chiều thay vì chỉ dùng accuracy.

Có thể tham khảo để thiết kế rubric cho Agent decisions.

---

#### MathTutorBench

Benchmark này đánh giá LLM trong vai trò tutor.

Điểm quan trọng cần rút ra:

- Subject-matter expertise và pedagogical ability là hai capability khác nhau.
- Một model giỏi giải toán chưa chắc giỏi dạy toán.
- Long multi-turn tutoring dialogue có thể làm chất lượng pedagogical behavior giảm.

Điều này trực tiếp hỗ trợ motivation cho project.

---

### Hướng 4 — Agentic AI in Education

#### Agentic AI in Education: State of the Art and Future Directions

Có thể dùng tài liệu này để hiểu broader landscape của AI Agent trong giáo dục.

Các chủ đề liên quan:

- Autonomous educational agents
- Long-term interaction
- Memory
- Planning
- Learner modeling
- Action selection
- Multi-step tutoring

---

#### Systematic review về LLM educational agents

Một hướng tổng quan đáng đọc là các nghiên cứu phân tích educational agent theo các thành phần:

```text
Profile
Memory
Planning
Action
```

Framework này khá gần với architecture của project:

- Profile ≈ learner state
- Memory ≈ interaction history
- Planning ≈ pedagogical decision
- Action ≈ question / hint / explanation / review / advance

---

## Tóm tắt hướng nghiên cứu

Project nên đặt trọng tâm vào câu hỏi:

> **Can an LLM Agent make better pedagogical decisions than conventional adaptive policies when given the same explicit student knowledge state?**

Do đó, các nhóm kiến thức chính cần tìm hiểu gồm:

1. **Knowledge Tracing / Student Modeling**
2. **Adaptive Learning / Intelligent Tutoring Systems**
3. **Reinforcement Learning / Sequential Decision Making in Education**
4. **Pedagogy and Mastery Learning**
5. **LLM Tutor Evaluation**
6. **Agentic AI in Education**

Cấu trúc này giúp project có một research story rõ ràng:

\[
\text{Estimate Learner State}
\rightarrow
\text{Choose Pedagogical Action}
\rightarrow
\text{Measure Learning Effectiveness}
\]

và cho phép kiểm tra trực tiếp giá trị thực sự của LLM Agent trong một adaptive learning system.
