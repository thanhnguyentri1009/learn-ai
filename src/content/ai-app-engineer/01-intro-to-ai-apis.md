# Giới thiệu AI APIs

## AI APIs là gì?

AI APIs (Application Programming Interfaces) cho phép các nhà phát triển tích hợp khả năng trí tuệ nhân tạo vào ứng dụng của mình mà không cần tự xây dựng và huấn luyện model. Thay vì phải có hàng chục GPU và hàng tháng huấn luyện, bạn chỉ cần gửi một HTTP request và nhận về kết quả ngay lập tức.

### Các nhà cung cấp AI API phổ biến

| Nhà cung cấp | Model nổi bật | Điểm mạnh |
|---|---|---|
| **OpenAI** | GPT-4o, GPT-4 Turbo | Ecosystem lớn, nhiều tính năng |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus | An toàn, lý luận tốt |
| **Google** | Gemini 1.5 Pro, Gemini Flash | Context window lớn, multimodal |
| **Mistral** | Mistral Large, Mixtral | Open-source options, giá rẻ |
| **Cohere** | Command R+ | Tối ưu cho RAG và search |

## Cách hoạt động

### Mô hình Request-Response

```
Ứng dụng của bạn
      │
      │  HTTP POST (prompt + parameters)
      ▼
  AI Provider API
      │
      │  JSON Response (generated text + metadata)
      ▼
Ứng dụng của bạn
```

### Các khái niệm quan trọng

**Tokens**: Đơn vị cơ bản mà model xử lý. Một token thường tương đương ~4 ký tự tiếng Anh. Câu "Hello, how are you?" có khoảng 6 tokens.

**Context Window**: Số token tối đa model có thể "nhìn thấy" cùng lúc. GPT-4o hỗ trợ 128K tokens, Gemini 1.5 Pro hỗ trợ 1M tokens.

**Temperature**: Kiểm soát độ ngẫu nhiên của output:
- `temperature=0`: Deterministic, luôn cho kết quả giống nhau
- `temperature=1`: Sáng tạo hơn, đa dạng hơn
- `temperature=2`: Rất ngẫu nhiên, có thể không coherent

**System / User / Assistant roles**:
- `system`: Hướng dẫn tổng quát cho model (personality, rules, context)
- `user`: Input từ người dùng
- `assistant`: Phản hồi của model (dùng trong conversation history)

## Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt thư viện
pip install openai anthropic python-dotenv
```

Tạo file `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Gọi OpenAI API

### Ví dụ cơ bản

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key từ file .env
load_dotenv()

# Khởi tạo client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def simple_completion(prompt: str) -> str:
    """Gọi OpenAI API với một prompt đơn giản."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Model rẻ hơn để test
        messages=[
            {
                "role": "system",
                "content": "Bạn là một trợ lý AI hữu ích, trả lời bằng tiếng Việt."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,      # Độ sáng tạo vừa phải
        max_tokens=1000,      # Giới hạn độ dài response
    )

    # Lấy text từ response
    return response.choices[0].message.content

# Test
result = simple_completion("Giải thích AI API là gì trong 3 câu.")
print(result)
```

### Conversation với lịch sử

```python
from openai import OpenAI

client = OpenAI()

class Chatbot:
    def __init__(self, system_prompt: str):
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def chat(self, user_message: str) -> str:
        # Thêm tin nhắn của user vào lịch sử
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # Gọi API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            temperature=0.7,
        )

        # Lấy phản hồi của assistant
        assistant_message = response.choices[0].message.content

        # Lưu phản hồi vào lịch sử
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

# Sử dụng
bot = Chatbot("Bạn là chuyên gia tư vấn Python. Trả lời ngắn gọn, chính xác.")
print(bot.chat("List comprehension là gì?"))
print(bot.chat("Cho tôi ví dụ thực tế hơn"))
print(bot.chat("So sánh với vòng lặp for thông thường"))
```

## Gọi Anthropic (Claude) API

```python
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def claude_completion(prompt: str, system: str = "") -> str:
    """Gọi Claude API."""
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system if system else "Bạn là trợ lý AI hữu ích.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text

# Lưu ý: Anthropic dùng cú pháp khác OpenAI
# - system prompt là tham số riêng, không nằm trong messages
# - Response là message.content[0].text
# - Model name khác: "claude-3-5-sonnet-20241022"

result = claude_completion(
    prompt="Giải thích sự khác biệt giữa supervised và unsupervised learning",
    system="Bạn là giáo viên dạy AI. Giải thích đơn giản, dùng ví dụ thực tế."
)
print(result)
```

## Xử lý lỗi và Rate Limiting

```python
import time
import random
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

client = OpenAI()

def robust_completion(
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> str:
    """
    Gọi API với retry logic và exponential backoff.
    Handles: rate limits, timeouts, temporary errors.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=30,  # Timeout sau 30 giây
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            # Rate limit: chờ và thử lại
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Chờ {wait_time:.1f}s trước khi thử lại...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Đã thử {max_retries} lần nhưng vẫn bị rate limit") from e

        except APITimeoutError:
            # Timeout: thử lại ngay
            if attempt < max_retries - 1:
                print(f"Timeout. Đang thử lại (lần {attempt + 2}/{max_retries})...")
            else:
                raise Exception("API timeout sau nhiều lần thử")

        except APIError as e:
            # Lỗi API khác (5xx server errors)
            if e.status_code and e.status_code >= 500 and attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)
                print(f"Server error {e.status_code}. Chờ {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise nếu không thể retry

    return ""  # Fallback (không bao giờ đến đây nếu raise đúng cách
```

## Quản lý chi phí

```python
def estimate_cost(prompt: str, model: str = "gpt-4o-mini") -> dict:
    """
    Ước tính chi phí trước khi gọi API.
    Giá tháng 10/2024 (có thể thay đổi):
    - gpt-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens
    - gpt-4o: $5/1M input tokens, $15/1M output tokens
    """
    # Ước tính số tokens (1 token ~ 4 ký tự)
    estimated_input_tokens = len(prompt) / 4

    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    }

    if model not in pricing:
        return {"error": "Unknown model"}

    price = pricing[model]
    input_cost = (estimated_input_tokens / 1_000_000) * price["input"]

    return {
        "estimated_input_tokens": int(estimated_input_tokens),
        "estimated_input_cost_usd": round(input_cost, 6),
        "note": "Output cost depends on response length"
    }

# Track actual usage từ response
def track_usage(response) -> dict:
    usage = response.usage
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
```

## Streaming Response

```python
from openai import OpenAI

client = OpenAI()

def stream_completion(prompt: str):
    """
    Streaming cho phép hiển thị text ngay khi model generate,
    thay vì chờ toàn bộ response.
    """
    print("Assistant: ", end="", flush=True)

    with client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # Bật streaming
    ) as stream:
        for chunk in stream:
            # Mỗi chunk chứa một phần nhỏ của response
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)

    print()  # Newline sau khi xong

stream_completion("Kể tên 5 ngôn ngữ lập trình phổ biến nhất năm 2024")
```

## Best Practices

### 1. Luôn dùng environment variables cho API keys
```python
# KHÔNG BAO GIỜ làm thế này:
client = OpenAI(api_key="sk-abc123...")  # Hardcode API key!

# Làm đúng:
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### 2. Đặt max_tokens hợp lý
```python
# Không đặt quá cao nếu không cần, tiết kiệm chi phí
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    max_tokens=500,  # Phù hợp cho câu trả lời ngắn
)
```

### 3. Cache responses khi có thể
```python
import hashlib
import json

cache = {}

def cached_completion(prompt: str) -> str:
    # Tạo cache key từ prompt
    cache_key = hashlib.md5(prompt.encode()).hexdigest()

    if cache_key in cache:
        print("(từ cache)")
        return cache[cache_key]

    # Gọi API nếu không có trong cache
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    result = response.choices[0].message.content

    # Lưu vào cache
    cache[cache_key] = result
    return result
```

### 4. Log usage để monitor chi phí
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_completion(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    # Log token usage
    usage = response.usage
    logger.info(
        f"API call: {usage.prompt_tokens} input + "
        f"{usage.completion_tokens} output = "
        f"{usage.total_tokens} total tokens"
    )

    return response.choices[0].message.content
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: Script tóm tắt văn bản

Viết một script Python nhận vào một đoạn văn bản dài và sử dụng OpenAI API để tóm tắt thành 3-5 câu. Script cần:

- Đọc văn bản từ file hoặc stdin
- Gọi API với prompt phù hợp
- In ra bản tóm tắt
- Hiển thị số tokens đã dùng

**Đoạn văn mẫu để test:**
```
Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta làm việc và sống. Từ các ứng dụng y tế giúp chẩn đoán bệnh sớm hơn, đến các hệ thống tự động hóa trong sản xuất, AI đang thâm nhập vào mọi ngành nghề. Các mô hình ngôn ngữ lớn như GPT-4 và Claude có khả năng viết code, phân tích dữ liệu, và giải đáp câu hỏi phức tạp. Tuy nhiên, AI cũng đặt ra nhiều thách thức về đạo đức, việc làm, và bảo mật dữ liệu. Việc hiểu và làm chủ công nghệ AI không còn là lựa chọn mà là yêu cầu bắt buộc cho các developer trong thập kỷ tới.
```

### Bài tập 2: Chatbot với conversation history

Xây dựng một chatbot terminal đơn giản:
- Giữ lịch sử hội thoại qua các lượt chat
- Cho phép user nhập `quit` để thoát
- Hiển thị số tokens đã dùng trong session
- Có system prompt tùy chỉnh (ví dụ: chatbot về Python)

### Bài tập 3: Retry logic nâng cao

Implement một hàm `smart_completion()` với các tính năng:
- Exponential backoff khi gặp rate limit (1s, 2s, 4s, 8s...)
- Jitter ngẫu nhiên để tránh thundering herd
- Maximum retry count có thể cấu hình
- Logging rõ ràng mỗi lần retry
- Raise exception với thông tin hữu ích sau khi hết retry

---SOLUTION---

## Lời giải

### Giải bài tập 1: Script tóm tắt văn bản

```python
# summarize.py
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(text: str, max_sentences: int = 5) -> dict:
    """
    Tóm tắt văn bản sử dụng OpenAI API.

    Args:
        text: Văn bản cần tóm tắt
        max_sentences: Số câu tối đa trong bản tóm tắt

    Returns:
        dict chứa summary và usage info
    """
    prompt = f"""Hãy tóm tắt đoạn văn bản sau thành {max_sentences} câu súc tích.
Chỉ trả về bản tóm tắt, không thêm phần mở đầu hay kết luận.

Văn bản:
{text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Bạn là chuyên gia tóm tắt văn bản. Tóm tắt ngắn gọn, đầy đủ ý chính."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,  # Thấp để kết quả nhất quán
        max_tokens=300,
    )

    summary = response.choices[0].message.content
    usage = response.usage

    return {
        "summary": summary,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }

def main():
    # Đọc input
    if len(sys.argv) > 1:
        # Đọc từ file
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Dùng văn bản mẫu
        text = """Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta làm việc và sống.
Từ các ứng dụng y tế giúp chẩn đoán bệnh sớm hơn, đến các hệ thống tự động hóa
trong sản xuất, AI đang thâm nhập vào mọi ngành nghề. Các mô hình ngôn ngữ lớn
như GPT-4 và Claude có khả năng viết code, phân tích dữ liệu, và giải đáp câu hỏi
phức tạp. Tuy nhiên, AI cũng đặt ra nhiều thách thức về đạo đức, việc làm, và
bảo mật dữ liệu. Việc hiểu và làm chủ công nghệ AI không còn là lựa chọn mà là
yêu cầu bắt buộc cho các developer trong thập kỷ tới."""

    print("Đang tóm tắt...\n")
    result = summarize_text(text)

    print("=" * 50)
    print("BẢN TÓM TẮT:")
    print("=" * 50)
    print(result["summary"])
    print()
    print(f"Tokens đã dùng: {result['input_tokens']} input + "
          f"{result['output_tokens']} output = "
          f"{result['total_tokens']} tổng")

if __name__ == "__main__":
    main()
```

### Giải bài tập 2: Chatbot với conversation history

```python
# chatbot.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TerminalChatbot:
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.model = model
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
        self.total_tokens = 0
        self.turn_count = 0

    def chat(self, user_input: str) -> str:
        """Gửi tin nhắn và nhận phản hồi."""
        # Thêm tin nhắn user
        self.messages.append({"role": "user", "content": user_input})

        # Gọi API
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
            max_tokens=800,
        )

        # Lấy phản hồi
        assistant_reply = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # Cập nhật tracking
        self.messages.append({"role": "assistant", "content": assistant_reply})
        self.total_tokens += tokens_used
        self.turn_count += 1

        return assistant_reply, tokens_used

    def get_stats(self) -> dict:
        return {
            "turns": self.turn_count,
            "total_tokens": self.total_tokens,
            "message_count": len(self.messages),
        }

def main():
    print("=" * 60)
    print("Python Expert Chatbot")
    print("Gõ 'quit' để thoát, 'stats' để xem thống kê")
    print("=" * 60)

    # System prompt tùy chỉnh cho chatbot Python
    system_prompt = """Bạn là chuyên gia Python với 10 năm kinh nghiệm.
Hãy trả lời các câu hỏi về Python một cách rõ ràng, kèm theo ví dụ code khi cần.
Luôn giải thích tại sao, không chỉ giải thích cái gì."""

    bot = TerminalChatbot(system_prompt)

    while True:
        try:
            user_input = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            stats = bot.get_stats()
            print(f"\nTổng kết session:")
            print(f"  Số lượt chat: {stats['turns']}")
            print(f"  Tổng tokens: {stats['total_tokens']}")
            print("Tạm biệt!")
            break

        if user_input.lower() == 'stats':
            stats = bot.get_stats()
            print(f"\nThống kê session:")
            print(f"  Lượt chat: {stats['turns']}")
            print(f"  Tổng tokens: {stats['total_tokens']}")
            print(f"  Tin nhắn trong memory: {stats['message_count']}")
            continue

        reply, tokens = bot.chat(user_input)
        print(f"\nAssistant: {reply}")
        print(f"[Tokens lượt này: {tokens}]")

if __name__ == "__main__":
    main()
```

### Giải bài tập 3: Retry logic nâng cao

```python
# smart_completion.py
import os
import time
import random
import logging
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class APICallError(Exception):
    """Custom exception cho API call failures."""
    def __init__(self, message: str, attempts: int, last_error: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error

def smart_completion(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_range: float = 0.5,
    timeout: float = 30.0,
) -> str:
    """
    Gọi OpenAI API với retry logic nâng cao.

    Args:
        prompt: Câu hỏi hoặc nội dung cần xử lý
        model: Model ID
        max_retries: Số lần retry tối đa
        base_delay: Thời gian chờ cơ bản (giây)
        max_delay: Thời gian chờ tối đa (giây)
        jitter_range: Khoảng ngẫu nhiên thêm vào delay (giây)
        timeout: Timeout cho mỗi API call (giây)

    Returns:
        Generated text từ API

    Raises:
        APICallError: Nếu tất cả retry đều thất bại
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempting API call (attempt {attempt}/{max_retries})")

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )

            result = response.choices[0].message.content
            logger.info(f"Success on attempt {attempt}. "
                       f"Tokens: {response.usage.total_tokens}")
            return result

        except RateLimitError as e:
            last_exception = e
            if attempt == max_retries:
                break

            # Exponential backoff với jitter
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(-jitter_range, jitter_range)
            actual_delay = max(0, delay + jitter)

            logger.warning(
                f"Rate limit hit (attempt {attempt}). "
                f"Waiting {actual_delay:.2f}s before retry..."
            )
            time.sleep(actual_delay)

        except APITimeoutError as e:
            last_exception = e
            if attempt == max_retries:
                break

            # Timeout: retry nhanh hơn vì có thể là tạm thời
            delay = min(base_delay * attempt, 10.0)
            logger.warning(
                f"Request timeout (attempt {attempt}). "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

        except APIError as e:
            last_exception = e
            # Chỉ retry với 5xx server errors
            if not (e.status_code and e.status_code >= 500):
                logger.error(f"Non-retryable API error: {e}")
                raise APICallError(
                    f"Non-retryable error: {e}",
                    attempts=attempt,
                    last_error=e
                ) from e

            if attempt == max_retries:
                break

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                f"Server error {e.status_code} (attempt {attempt}). "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

        except Exception as e:
            # Unexpected errors: không retry
            logger.error(f"Unexpected error: {e}")
            raise APICallError(
                f"Unexpected error: {e}",
                attempts=attempt,
                last_error=e
            ) from e

    # Hết retry, raise lỗi với thông tin đầy đủ
    raise APICallError(
        f"API call failed after {max_retries} attempts. "
        f"Last error: {last_exception}",
        attempts=max_retries,
        last_error=last_exception
    ) from last_exception

# Test
if __name__ == "__main__":
    try:
        result = smart_completion(
            "Giải thích khái niệm exponential backoff trong 2 câu.",
            max_retries=3
        )
        print(f"Result: {result}")
    except APICallError as e:
        print(f"Failed after {e.attempts} attempts: {e}")
```
