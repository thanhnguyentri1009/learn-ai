# Prompt Engineering

## Prompt Engineering là gì?

Prompt Engineering là nghệ thuật và khoa học thiết kế các câu lệnh (prompts) để hướng dẫn AI model tạo ra output mong muốn. Một prompt tốt có thể là sự khác biệt giữa kết quả vô nghĩa và kết quả chuyên nghiệp.

> "Chất lượng của prompt quyết định chất lượng của output. Garbage in, garbage out."

## Zero-shot Prompting

Zero-shot là khi bạn yêu cầu model thực hiện task mà không cung cấp ví dụ nào.

```python
from openai import OpenAI

client = OpenAI()

# Zero-shot: Không có ví dụ
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Phân loại cảm xúc của câu sau: 'Tôi rất thất vọng với dịch vụ này.'"
    }]
)
# Output: "Tiêu cực" hoặc "Negative"
```

**Khi nào dùng zero-shot:**
- Task đơn giản, rõ ràng
- Model đã được huấn luyện trên nhiều data tương tự
- Nhanh và tiết kiệm tokens

## Few-shot Prompting

Cung cấp vài ví dụ (examples) để model học theo pattern.

```python
few_shot_prompt = """Phân loại cảm xúc của các câu sau thành: Tích cực, Tiêu cực, hoặc Trung lập.

Câu: "Sản phẩm này tuyệt vời, tôi rất hài lòng!"
Cảm xúc: Tích cực

Câu: "Giao hàng chậm, chất lượng kém."
Cảm xúc: Tiêu cực

Câu: "Đơn hàng đã được xác nhận."
Cảm xúc: Trung lập

Câu: "Tôi sẽ không mua lại sản phẩm này nữa."
Cảm xúc:"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": few_shot_prompt}]
)
# Output: "Tiêu cực"
```

**Khi nào dùng few-shot:**
- Format output cụ thể
- Task không phổ biến
- Cần tính nhất quán cao
- Classification tasks

## Chain-of-Thought (CoT) Prompting

Yêu cầu model "suy nghĩ từng bước" trước khi đưa ra câu trả lời cuối cùng.

```python
# Không có CoT - thường sai với toán phức tạp
simple_prompt = "Roger có 5 quả bóng tennis. Anh ấy mua thêm 2 hộp bóng, mỗi hộp có 3 quả. Hỏi Roger có bao nhiêu quả bóng?"

# Với CoT - chính xác hơn nhiều
cot_prompt = """Roger có 5 quả bóng tennis. Anh ấy mua thêm 2 hộp bóng, mỗi hộp có 3 quả.
Hỏi Roger có bao nhiêu quả bóng?

Hãy suy nghĩ từng bước:"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": cot_prompt}],
    temperature=0,  # Temperature = 0 cho reasoning tasks
)
```

**Output mẫu với CoT:**
```
Bước 1: Roger ban đầu có 5 quả bóng tennis.
Bước 2: Anh ấy mua 2 hộp bóng, mỗi hộp có 3 quả.
Bước 3: Số bóng mua thêm = 2 × 3 = 6 quả.
Bước 4: Tổng số bóng = 5 + 6 = 11 quả.
Vậy Roger có 11 quả bóng tennis.
```

### Zero-shot CoT

```python
# Chỉ cần thêm "Let's think step by step" hoặc "Hãy suy nghĩ từng bước"
zero_shot_cot = """Giải bài toán sau:
Một cửa hàng bán áo với giá 200,000đ/chiếc. Trong tháng 1, họ bán được 150 chiếc.
Tháng 2, doanh số tăng 20% so với tháng 1. Tính tổng doanh thu 2 tháng.

Hãy suy nghĩ từng bước trước khi trả lời."""
```

## Role Prompting

Gán role (vai trò) cụ thể cho model để cải thiện chất lượng.

```python
# Không có role
basic_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Review đoạn code Python này: [code]"
    }]
)

# Với role trong system prompt
role_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": """Bạn là một Senior Python Developer với 10 năm kinh nghiệm.
Chuyên môn: clean code, performance optimization, security.
Khi review code, hãy:
1. Chỉ ra bugs và security issues (ưu tiên cao)
2. Đề xuất cải thiện performance
3. Gợi ý best practices
4. Giải thích lý do cho mỗi gợi ý
Dùng markdown với headers và code blocks."""
        },
        {
            "role": "user",
            "content": f"Review đoạn code Python này:\n```python\n{code}\n```"
        }
    ],
    temperature=0.3,
)
```

### Các role hữu ích

```python
roles = {
    "code_reviewer": "Senior Software Engineer chuyên review code, tìm bugs và đề xuất improvements.",
    "teacher": "Giáo viên kiên nhẫn giải thích mọi khái niệm từ cơ bản, dùng ví dụ thực tế.",
    "translator": "Dịch thuật chuyên nghiệp, giữ nguyên tone và ý nghĩa, không thêm bớt.",
    "analyst": "Data Analyst phân tích số liệu, tìm insights, và đưa ra recommendations dựa trên data.",
    "writer": "Content writer viết ngắn gọn, hấp dẫn, SEO-friendly, phù hợp với target audience.",
}
```

## Output Formatting

### JSON Output

```python
import json

def extract_info_as_json(text: str) -> dict:
    """Trích xuất thông tin từ text và trả về JSON."""

    prompt = f"""Phân tích đoạn text sau và trích xuất thông tin vào JSON.
Chỉ trả về JSON, không thêm text khác.

Format JSON:
{{
  "person": {{
    "name": "string",
    "age": "number hoặc null",
    "occupation": "string hoặc null"
  }},
  "location": "string hoặc null",
  "date": "string hoặc null",
  "key_facts": ["array of strings"]
}}

Text:
{text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},  # Force JSON output
    )

    return json.loads(response.choices[0].message.content)

# Test
text = "Ngày 15/3/2024, ông Nguyễn Văn An (45 tuổi), giám đốc công ty ABC tại Hà Nội, đã ký kết hợp đồng trị giá 5 tỷ đồng."
info = extract_info_as_json(text)
print(json.dumps(info, ensure_ascii=False, indent=2))
```

### Structured Output với Pydantic

```python
from pydantic import BaseModel
from typing import Optional, List
from openai import OpenAI

client = OpenAI()

class ProductReview(BaseModel):
    sentiment: str  # "positive", "negative", "neutral"
    score: int      # 1-5
    pros: List[str]
    cons: List[str]
    summary: str

def analyze_review(review_text: str) -> ProductReview:
    """Phân tích review sản phẩm và trả về structured data."""

    prompt = f"""Phân tích review sản phẩm sau:
"{review_text}"

Trả về JSON với format:
- sentiment: "positive", "negative", hoặc "neutral"
- score: điểm từ 1-5
- pros: danh sách điểm tích cực (có thể rỗng)
- cons: danh sách điểm tiêu cực (có thể rỗng)
- summary: tóm tắt 1 câu"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    data = json.loads(response.choices[0].message.content)
    return ProductReview(**data)
```

## System Prompts hiệu quả

```python
# Cấu trúc system prompt tốt
system_prompt_template = """# Vai trò
{role_description}

# Nhiệm vụ
{task_description}

# Quy tắc
{rules}

# Format output
{output_format}

# Ví dụ (nếu cần)
{examples}"""

# Ví dụ thực tế: Customer support bot
customer_support_system = """# Vai trò
Bạn là nhân viên hỗ trợ khách hàng của TechShop - cửa hàng thiết bị điện tử.

# Nhiệm vụ
Giúp khách hàng giải quyết vấn đề về sản phẩm, đơn hàng, và chính sách.

# Quy tắc
- Luôn lịch sự và chuyên nghiệp
- Xin lỗi khi khách hàng gặp vấn đề
- Cung cấp giải pháp cụ thể, không trả lời chung chung
- Nếu không biết, nói "Tôi sẽ chuyển vấn đề này đến bộ phận kỹ thuật"
- KHÔNG tiết lộ thông tin nội bộ của công ty
- KHÔNG hứa hẹn những gì không thể thực hiện

# Format output
- Ngắn gọn, tối đa 3-4 câu
- Kết thúc bằng câu hỏi "Tôi có thể giúp gì thêm cho bạn không?"

# Thông tin sản phẩm
- Bảo hành: 12 tháng cho tất cả sản phẩm
- Đổi trả: trong 30 ngày nếu sản phẩm lỗi
- Hotline: 1800-xxxx (8:00-22:00)"""
```

## Prompt Templates và Reusability

```python
from string import Template

class PromptTemplate:
    def __init__(self, template: str):
        self.template = Template(template)

    def format(self, **kwargs) -> str:
        return self.template.safe_substitute(**kwargs)

# Tạo các template tái sử dụng
SUMMARIZE_TEMPLATE = PromptTemplate("""
Tóm tắt đoạn văn sau thành $num_sentences câu.
Tập trung vào: $focus_area
Tone: $tone

Văn bản:
$text
""")

CLASSIFY_TEMPLATE = PromptTemplate("""
Phân loại văn bản sau vào một trong các danh mục: $categories

Văn bản: $text

Chỉ trả về tên danh mục, không giải thích.
""")

TRANSLATE_TEMPLATE = PromptTemplate("""
Dịch đoạn văn sau từ $source_lang sang $target_lang.
Giữ nguyên tone và phong cách của văn bản gốc.

Văn bản:
$text
""")

# Sử dụng
summary_prompt = SUMMARIZE_TEMPLATE.format(
    num_sentences=3,
    focus_area="điểm chính và kết luận",
    tone="chuyên nghiệp",
    text="[văn bản của bạn ở đây]"
)
```

## Kỹ thuật nâng cao: Self-Consistency

```python
def self_consistency_answer(question: str, num_samples: int = 5) -> str:
    """
    Gọi model nhiều lần với temperature cao, sau đó lấy câu trả lời phổ biến nhất.
    Cải thiện độ chính xác cho reasoning tasks.
    """
    answers = []

    for i in range(num_samples):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"{question}\n\nHãy suy nghĩ từng bước và đưa ra câu trả lời cuối cùng trên dòng riêng bắt đầu bằng 'Đáp án:'"
                }
            ],
            temperature=0.7,  # Temperature cao để có đa dạng
        )
        text = response.choices[0].message.content
        # Lấy phần đáp án cuối
        if "Đáp án:" in text:
            answer = text.split("Đáp án:")[-1].strip()
        else:
            answer = text.strip()
        answers.append(answer)

    # Tìm câu trả lời xuất hiện nhiều nhất
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0]
    return most_common[0]
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: Email classifier với few-shot

Xây dựng một email classifier sử dụng few-shot prompting để phân loại email vào các danh mục:
- **Spam**: Email quảng cáo không mong muốn
- **Quan trọng**: Email cần hành động ngay
- **Thông thường**: Email thông tin, không cần hành động gấp

Yêu cầu:
- Cung cấp ít nhất 2 ví dụ cho mỗi danh mục
- Trả về kết quả dạng JSON: `{"category": "...", "confidence": 0-1, "reason": "..."}`
- Test với 5 email mẫu khác nhau

### Bài tập 2: Code explainer với chain-of-thought

Viết một hàm `explain_code()` nhận vào một đoạn code Python và:
1. Dùng CoT để phân tích từng phần của code
2. Giải thích theo 3 level:
   - **Beginner**: Giải thích đơn giản, tránh jargon
   - **Intermediate**: Giải thích logic và pattern
   - **Advanced**: Phân tích performance, edge cases, improvements
3. Trả về JSON với 3 level này

### Bài tập 3: Dynamic prompt builder

Xây dựng một class `PromptBuilder` có các tính năng:
- Method `add_context(context: str)`: Thêm context
- Method `add_examples(*examples)`: Thêm few-shot examples
- Method `add_constraint(*constraints)`: Thêm ràng buộc
- Method `set_output_format(format: str)`: Đặt format output
- Method `build() -> str`: Tạo prompt hoàn chỉnh
- Method `run(user_input: str) -> str`: Gọi API và trả về kết quả

---SOLUTION---

## Lời giải

### Giải bài tập 1: Email classifier

```python
import json
from openai import OpenAI

client = OpenAI()

def classify_email(email_subject: str, email_body: str) -> dict:
    """
    Phân loại email sử dụng few-shot prompting.
    """
    prompt = f"""Phân loại email sau vào một trong 3 danh mục: Spam, Quan trọng, Thông thường.

=== VÍ DỤ SPAM ===
Subject: Chúc mừng! Bạn đã trúng thưởng iPhone 15 Pro!
Body: Click vào link dưới để nhận quà: bit.ly/claim-prize. Ưu đãi chỉ trong 24h!
Kết quả: {{"category": "Spam", "confidence": 0.98, "reason": "Quảng cáo trúng thưởng giả mạo với link đáng ngờ"}}

Subject: Giảm giá 90% tất cả sản phẩm - Chỉ hôm nay!
Body: Mua ngay để không bỏ lỡ cơ hội vàng. Nhiều sản phẩm hot đang chờ bạn...
Kết quả: {{"category": "Spam", "confidence": 0.95, "reason": "Email marketing quảng cáo với lời mời hấp dẫn không thực tế"}}

=== VÍ DỤ QUAN TRỌNG ===
Subject: [URGENT] Server production đang down - cần xử lý ngay
Body: Server của chúng ta đang offline từ 14:30. Khách hàng không truy cập được. Team cần họp khẩn lúc 15:00.
Kết quả: {{"category": "Quan trọng", "confidence": 0.99, "reason": "Sự cố production nghiêm trọng cần xử lý ngay lập tức"}}

Subject: Deadline nộp báo cáo: ngày mai 9:00
Body: Nhắc nhở: Báo cáo Q4 cần nộp cho sếp trước 9:00 sáng mai. Vui lòng kiểm tra và xác nhận.
Kết quả: {{"category": "Quan trọng", "confidence": 0.92, "reason": "Deadline gấp cần hành động ngay"}}

=== VÍ DỤ THÔNG THƯỜNG ===
Subject: Newsletter tháng 10 - Tin tức công ty
Body: Chào các bạn, dưới đây là các tin tức nổi bật trong tháng vừa qua...
Kết quả: {{"category": "Thông thường", "confidence": 0.88, "reason": "Newsletter thông tin định kỳ, không cần hành động gấp"}}

Subject: Kết quả họp team tuần này
Body: Đây là tổng hợp các điểm đã thảo luận trong buổi họp hôm nay để tham khảo.
Kết quả: {{"category": "Thông thường", "confidence": 0.85, "reason": "Email tổng hợp thông tin, chỉ để tham khảo"}}

=== EMAIL CẦN PHÂN LOẠI ===
Subject: {email_subject}
Body: {email_body}
Kết quả (chỉ trả về JSON):"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)

# Test với 5 email mẫu
test_emails = [
    {
        "subject": "Cuộc họp review sprint ngày mai",
        "body": "Nhắc nhở team: Sprint review lúc 10:00 sáng mai. Mọi người chuẩn bị demo phần mình đã làm."
    },
    {
        "subject": "Earn $5000/week working from home!!!",
        "body": "No experience needed! Click here to start your journey to financial freedom..."
    },
    {
        "subject": "Password reset request",
        "body": "Ai đó đã yêu cầu reset password cho tài khoản của bạn. Nếu không phải bạn, hãy liên hệ ngay bộ phận bảo mật."
    },
    {
        "subject": "Báo cáo traffic tháng 9",
        "body": "Dưới đây là tổng hợp số liệu traffic website trong tháng 9/2024."
    },
    {
        "subject": "Hệ thống thanh toán đang gặp lỗi",
        "body": "ALERT: 30% giao dịch đang bị fail từ 16:00. Ảnh hưởng trực tiếp đến doanh thu. Cần fix ngay!"
    },
]

for email in test_emails:
    result = classify_email(email["subject"], email["body"])
    print(f"Subject: {email['subject'][:50]}...")
    print(f"  => {result['category']} (confidence: {result['confidence']:.0%})")
    print(f"  Lý do: {result['reason']}\n")
```

### Giải bài tập 2: Code explainer với CoT

```python
import json
from openai import OpenAI

client = OpenAI()

def explain_code(code: str) -> dict:
    """
    Giải thích code Python theo 3 levels sử dụng Chain-of-Thought.
    """
    prompt = f"""Bạn là giáo viên lập trình xuất sắc. Hãy giải thích đoạn code Python sau.

BƯỚC 1 - PHÂN TÍCH: Đọc code và xác định:
- Mục đích tổng thể là gì?
- Các thành phần chính (variables, functions, loops, etc.)
- Logic flow

BƯỚC 2 - GIẢI THÍCH 3 LEVELS:
Sau khi phân tích, tạo 3 giải thích cho 3 đối tượng khác nhau.

Code cần giải thích:
```python
{code}
```

Trả về JSON với format sau:
{{
  "purpose": "Mục đích của code trong 1 câu",
  "analysis": "Phân tích ngắn gọn các thành phần",
  "beginner": {{
    "explanation": "Giải thích đơn giản, không jargon kỹ thuật",
    "analogy": "Ví dụ so sánh với cuộc sống thực"
  }},
  "intermediate": {{
    "explanation": "Giải thích logic và pattern sử dụng",
    "key_concepts": ["concept1", "concept2"]
  }},
  "advanced": {{
    "explanation": "Phân tích sâu về implementation",
    "performance": "Phân tích time/space complexity",
    "improvements": ["cải tiến 1", "cải tiến 2"],
    "edge_cases": ["edge case 1", "edge case 2"]
  }}
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)

# Test
sample_code = """
def find_duplicates(lst):
    seen = set()
    duplicates = []
    for item in lst:
        if item in seen:
            if item not in duplicates:
                duplicates.append(item)
        else:
            seen.add(item)
    return duplicates
"""

result = explain_code(sample_code)
print(f"Mục đích: {result['purpose']}\n")
print("=== BEGINNER ===")
print(result['beginner']['explanation'])
print(f"Ví dụ: {result['beginner']['analogy']}\n")
print("=== INTERMEDIATE ===")
print(result['intermediate']['explanation'])
print(f"Concepts: {', '.join(result['intermediate']['key_concepts'])}\n")
print("=== ADVANCED ===")
print(result['advanced']['explanation'])
print(f"Performance: {result['advanced']['performance']}")
print(f"Cải tiến: {result['advanced']['improvements']}")
```

### Giải bài tập 3: Dynamic PromptBuilder

```python
from openai import OpenAI
from typing import List, Optional

client = OpenAI()

class PromptBuilder:
    """
    Builder pattern để tạo prompts có cấu trúc và tái sử dụng được.
    """

    def __init__(self, task: str):
        """
        Args:
            task: Mô tả task chính cần thực hiện
        """
        self.task = task
        self._context: List[str] = []
        self._examples: List[dict] = []
        self._constraints: List[str] = []
        self._output_format: Optional[str] = None

    def add_context(self, context: str) -> 'PromptBuilder':
        """Thêm thông tin context. Hỗ trợ method chaining."""
        self._context.append(context)
        return self

    def add_examples(self, *examples) -> 'PromptBuilder':
        """
        Thêm few-shot examples.
        Mỗi example là dict với keys 'input' và 'output'.
        """
        for example in examples:
            if isinstance(example, dict) and 'input' in example and 'output' in example:
                self._examples.append(example)
        return self

    def add_constraint(self, *constraints) -> 'PromptBuilder':
        """Thêm ràng buộc cho output."""
        self._constraints.extend(constraints)
        return self

    def set_output_format(self, format_desc: str) -> 'PromptBuilder':
        """Đặt mô tả format output mong muốn."""
        self._output_format = format_desc
        return self

    def build(self) -> str:
        """Tổng hợp tất cả thành prompt hoàn chỉnh."""
        parts = [f"Task: {self.task}"]

        if self._context:
            parts.append("\nContext:")
            for ctx in self._context:
                parts.append(f"- {ctx}")

        if self._examples:
            parts.append("\nExamples:")
            for i, ex in enumerate(self._examples, 1):
                parts.append(f"\nExample {i}:")
                parts.append(f"Input: {ex['input']}")
                parts.append(f"Output: {ex['output']}")

        if self._constraints:
            parts.append("\nConstraints:")
            for constraint in self._constraints:
                parts.append(f"- {constraint}")

        if self._output_format:
            parts.append(f"\nOutput format: {self._output_format}")

        parts.append("\nNow process the following input:")
        return "\n".join(parts)

    def run(self, user_input: str, temperature: float = 0.7) -> str:
        """Build prompt và gọi API."""
        system_prompt = self.build()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

# Ví dụ sử dụng
sentiment_builder = (
    PromptBuilder("Phân tích cảm xúc của text và phân loại")
    .add_context("Đây là review sản phẩm từ khách hàng Việt Nam")
    .add_context("Phân tích cả text và emoji")
    .add_examples(
        {"input": "Sản phẩm tuyệt vời! 5 sao ⭐⭐⭐⭐⭐", "output": "Tích cực"},
        {"input": "Chất lượng kém, không như quảng cáo 😡", "output": "Tiêu cực"},
        {"input": "Bình thường, tạm được", "output": "Trung lập"},
    )
    .add_constraint("Chỉ trả về một trong: Tích cực, Tiêu cực, Trung lập")
    .add_constraint("Không giải thích, chỉ trả về nhãn")
    .set_output_format("Một từ hoặc cụm từ duy nhất")
)

# Test
reviews = [
    "Giao hàng siêu nhanh, đóng gói cẩn thận! Sẽ mua lại 🥰",
    "Sản phẩm không giống ảnh, thất vọng quá",
    "OK, dùng được. Không có gì đặc biệt.",
]

for review in reviews:
    result = sentiment_builder.run(review, temperature=0)
    print(f"Review: {review[:50]}...")
    print(f"=> {result}\n")
```
