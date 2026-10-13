# LangChain Framework

## LangChain là gì?

**LangChain** là framework mã nguồn mở giúp xây dựng ứng dụng AI phức tạp bằng cách kết nối các components lại với nhau: LLMs, tools, memory, và data sources.

> LangChain = "LEGO bricks for AI applications"

### Tại sao dùng LangChain?

- **Abstractions**: Chuẩn hóa interface giữa các LLM providers khác nhau
- **Chains**: Kết nối nhiều bước xử lý thành pipeline
- **Agents**: LLM tự quyết định dùng tools nào
- **Memory**: Quản lý conversation history
- **Integrations**: Kết nối với 100+ tools và databases

## Cài đặt

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

## Cơ bản: LLM và ChatModel

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Khởi tạo model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
)

# Gọi model trực tiếp
messages = [
    SystemMessage(content="Bạn là chuyên gia Python."),
    HumanMessage(content="Giải thích list comprehension?"),
]

response = llm.invoke(messages)
print(response.content)  # AIMessage

# Cách đơn giản hơn
response = llm.invoke("Giải thích list comprehension trong Python?")
print(response.content)
```

## Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Simple prompt template
template = PromptTemplate(
    input_variables=["language", "concept"],
    template="Giải thích {concept} trong {language} bằng ví dụ đơn giản."
)

formatted = template.format(language="Python", concept="decorator")
print(formatted)

# Chat prompt template (cho ChatModel)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "Bạn là giáo viên {subject}. Giải thích ngắn gọn, dùng ví dụ thực tế."),
    ("human", "{question}"),
])

messages = chat_template.format_messages(
    subject="lập trình Python",
    question="Generator là gì?"
)

response = llm.invoke(messages)
print(response.content)
```

## Chains

### LCEL (LangChain Expression Language)

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

# Chain đơn giản với pipe operator
prompt = ChatPromptTemplate.from_template(
    "Dịch câu sau sang tiếng Anh: {text}"
)
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"text": "Xin chào, bạn có khỏe không?"})
print(result)  # "Hello, how are you?"
```

### Sequential Chain

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Bước 1: Phân tích yêu cầu
analyze_prompt = ChatPromptTemplate.from_template("""
Phân tích yêu cầu code sau và liệt kê:
1. Chức năng cần implement
2. Input/Output
3. Edge cases cần xử lý

Yêu cầu: {requirement}
""")

# Bước 2: Viết code
code_prompt = ChatPromptTemplate.from_template("""
Dựa trên phân tích sau, viết code Python hoàn chỉnh:

{analysis}

Yêu cầu gốc: {requirement}

Code Python:
""")

# Bước 3: Review code
review_prompt = ChatPromptTemplate.from_template("""
Review đoạn code sau và chỉ ra:
- Bugs (nếu có)
- Cải tiến performance
- Best practices chưa áp dụng

Code:
{code}
""")

# Tạo pipeline
analyze_chain = analyze_prompt | llm | StrOutputParser()
code_chain = code_prompt | llm | StrOutputParser()
review_chain = review_prompt | llm | StrOutputParser()

def code_pipeline(requirement: str) -> dict:
    """Pipeline đầy đủ: analyze -> code -> review."""
    print("1. Đang phân tích yêu cầu...")
    analysis = analyze_chain.invoke({"requirement": requirement})

    print("2. Đang viết code...")
    code = code_chain.invoke({
        "analysis": analysis,
        "requirement": requirement
    })

    print("3. Đang review code...")
    review = review_chain.invoke({"code": code})

    return {
        "analysis": analysis,
        "code": code,
        "review": review,
    }

# Test
result = code_pipeline(
    "Viết hàm tính số Fibonacci với memoization"
)
print("\n=== CODE ===")
print(result["code"])
print("\n=== REVIEW ===")
print(result["review"])
```

## Memory

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Buffer Memory: Giữ toàn bộ lịch sử
buffer_memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=buffer_memory, verbose=True)

conversation.predict(input="Tên tôi là Thanh")
conversation.predict(input="Tôi đang học về AI")
response = conversation.predict(input="Bạn còn nhớ tên tôi không?")
print(response)  # Sẽ nhớ "Thanh"

# Summary Memory: Tóm tắt lịch sử để tiết kiệm tokens
summary_memory = ConversationSummaryMemory(llm=llm)
long_conversation = ConversationChain(llm=llm, memory=summary_memory)

# Phù hợp cho conversation dài
for i in range(10):
    long_conversation.predict(input=f"Tin nhắn thứ {i+1}: [nội dung dài]")
```

## Tools và Agents

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

# Định nghĩa custom tools
@tool
def calculator(expression: str) -> str:
    """
    Tính toán biểu thức toán học.
    Input: biểu thức như "2 + 3 * 4" hoặc "sqrt(16)"
    """
    try:
        import math
        # Safe eval với chỉ các phép tính toán
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Lỗi: {str(e)}"

@tool
def get_current_time() -> str:
    """Lấy thời gian hiện tại."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def web_search(query: str) -> str:
    """
    Tìm kiếm thông tin trên web.
    Dùng khi cần thông tin cập nhật hoặc không biết.
    """
    # Demo: trong thực tế dùng API như Tavily, SerpAPI
    return f"Kết quả tìm kiếm cho '{query}': [Demo - cần tích hợp search API]"

# Tạo agent với tools
tools = [calculator, get_current_time, web_search]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI thông minh. Sử dụng các tools khi cần.
Luôn giải thích bạn đang làm gì và tại sao."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Test agent
result = executor.invoke({
    "input": "Tính 2^10 + sqrt(144) và cho biết mấy giờ rồi?"
})
print(result["output"])
```

## Retrieval Chain (RAG với LangChain)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def build_rag_chain(documents: list[str]) -> RetrievalQA:
    """
    Xây dựng RAG chain từ danh sách documents.
    """
    # Chunk documents
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc)
        all_chunks.extend(chunks)

    # Tạo vector store
    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        persist_directory="./langchain_chroma"
    )

    # Custom prompt
    rag_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Sử dụng thông tin từ tài liệu để trả lời câu hỏi.
Nếu không có thông tin, nói "Tôi không tìm thấy thông tin về điều này."

Tài liệu:
{context}

Câu hỏi: {question}
Câu trả lời:"""
    )

    # Tạo chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True,
    )

    return chain

# Sử dụng
sample_docs = [
    "Python được tạo bởi Guido van Rossum năm 1991...",
    "LangChain là framework để xây dựng LLM applications...",
]

chain = build_rag_chain(sample_docs)
result = chain.invoke({"query": "LangChain được dùng để làm gì?"})
print(result["result"])
```

## Streaming với LangChain

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Streaming LLM
streaming_llm = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

prompt = ChatPromptTemplate.from_template(
    "Viết một đoạn văn ngắn về: {topic}"
)

chain = prompt | streaming_llm

# Text sẽ được in ra từng từ khi model generate
print("Đang generate: ")
chain.invoke({"topic": "tầm quan trọng của AI trong giáo dục"})
print()  # Newline sau khi xong
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: Custom tool agent

Xây dựng một agent có các tools:
- `weather_tool(city)`: Trả về thời tiết giả (mock data)
- `currency_converter(amount, from_currency, to_currency)`: Chuyển đổi tiền tệ (mock rates)
- `translate(text, target_language)`: Dịch văn bản sử dụng LLM

Agent phải:
- Tự quyết định dùng tool nào dựa trên câu hỏi
- Kết hợp nhiều tools trong một câu trả lời
- Giải thích quá trình suy nghĩ

### Bài tập 2: Document Q&A Chain

Xây dựng một chain hoàn chỉnh:
1. Load documents từ folder
2. Chunk và index vào Chroma
3. Xây dựng RAG chain với custom prompt
4. Thêm conversation memory để hỏi follow-up
5. Support nhiều ngôn ngữ (hỏi bằng tiếng Việt, tài liệu tiếng Anh)

### Bài tập 3: Code Review Pipeline

Tạo một pipeline tự động review code:
1. Tool `analyze_complexity`: Phân tích độ phức tạp (cyclomatic complexity)
2. Tool `check_style`: Kiểm tra PEP8 (dùng `pycodestyle` library)
3. LLM review: Đưa ra suggestions
4. Final report: Tổng hợp tất cả findings thành Markdown report

---SOLUTION---

## Lời giải

### Giải bài tập 1: Custom tool agent

```python
# agent_with_tools.py
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

# Mock data
WEATHER_DATA = {
    "hanoi": {"temp": 28, "condition": "Nắng", "humidity": 75},
    "hcmc": {"temp": 32, "condition": "Có mây", "humidity": 80},
    "danang": {"temp": 30, "condition": "Mưa nhẹ", "humidity": 85},
    "default": {"temp": 25, "condition": "Không rõ", "humidity": 70},
}

EXCHANGE_RATES = {
    "USD": 1.0,
    "VND": 24500.0,
    "EUR": 0.92,
    "JPY": 150.0,
    "SGD": 1.35,
}

@tool
def weather_tool(city: str) -> str:
    """
    Lấy thông tin thời tiết của một thành phố.
    Input: tên thành phố (ví dụ: hanoi, hcmc, danang)
    Output: thông tin thời tiết hiện tại
    """
    city_lower = city.lower().replace(" ", "").replace("hồchíminh", "hcmc")
    city_lower = city_lower.replace("hànội", "hanoi").replace("đànẵng", "danang")

    data = WEATHER_DATA.get(city_lower, WEATHER_DATA["default"])

    return f"""Thời tiết {city}:
- Nhiệt độ: {data['temp']}°C
- Tình trạng: {data['condition']}
- Độ ẩm: {data['humidity']}%"""

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Chuyển đổi tiền tệ.
    Input: số tiền, đơn vị gốc, đơn vị đích
    Ví dụ: amount=100, from_currency="USD", to_currency="VND"
    """
    from_curr = from_currency.upper()
    to_curr = to_currency.upper()

    if from_curr not in EXCHANGE_RATES or to_curr not in EXCHANGE_RATES:
        return f"Không hỗ trợ cặp tiền tệ {from_curr}/{to_curr}"

    # Convert to USD first, then to target
    usd_amount = amount / EXCHANGE_RATES[from_curr]
    result = usd_amount * EXCHANGE_RATES[to_curr]

    return f"{amount:,.2f} {from_curr} = {result:,.2f} {to_curr}"

@tool
def translate(text: str, target_language: str) -> str:
    """
    Dịch văn bản sang ngôn ngữ đích.
    Input: văn bản cần dịch, ngôn ngữ đích (ví dụ: English, Vietnamese, Japanese)
    """
    from langchain_openai import ChatOpenAI
    translator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = translator.invoke(
        f"Dịch sang {target_language}, chỉ trả về bản dịch: {text}"
    )
    return response.content

# Tạo agent
tools = [weather_tool, currency_converter, translate]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý thông minh có thể kiểm tra thời tiết,
chuyển đổi tiền tệ, và dịch văn bản. Sử dụng tools phù hợp để trả lời.
Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# Test cases
test_questions = [
    "Thời tiết Hà Nội hôm nay thế nào?",
    "Đổi 500 USD sang VND giúp tôi",
    "Dịch 'Good morning, have a nice day!' sang tiếng Việt",
    "Thời tiết ở Đà Nẵng và đổi 1000 EUR sang VND",
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    result = executor.invoke({"input": question})
    print(f"A: {result['output']}")
```

### Giải bài tập 2: Document Q&A Chain với Memory

```python
# doc_qa_chain.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory

class DocumentQAChain:
    def __init__(self, language: str = "vi"):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        self.language = language

    def load_documents(self, texts: list[str], sources: list[str] = None):
        """Load và index documents."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        all_chunks = []
        all_metadatas = []

        for i, text in enumerate(texts):
            chunks = splitter.split_text(text)
            source = sources[i] if sources else f"doc_{i}"
            all_chunks.extend(chunks)
            all_metadatas.extend([{"source": source}] * len(chunks))

        self.vectorstore = Chroma.from_texts(
            texts=all_chunks,
            embedding=self.embeddings,
            metadatas=all_metadatas,
        )
        print(f"Indexed {len(all_chunks)} chunks")

    def _get_retriever(self):
        return self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def ask(self, question: str) -> str:
        """Trả lời câu hỏi với memory và multilingual support."""
        if not self.vectorstore:
            return "Chưa có tài liệu nào được load."

        # Lấy conversation history
        history = self.memory.load_memory_variables({})
        history_text = ""
        if history.get("history"):
            history_text = "\n".join([
                f"{'User' if msg.type == 'human' else 'AI'}: {msg.content}"
                for msg in history["history"][-4:]
            ])

        # Tìm relevant docs
        docs = self._get_retriever().invoke(question)
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "unknown") for d in docs]))

        # Build prompt
        prompt_text = f"""Trả lời câu hỏi bằng tiếng Việt dựa trên tài liệu.
Nếu tài liệu không có thông tin, hãy nói rõ.

Lịch sử hội thoại:
{history_text if history_text else "Không có"}

Tài liệu tham khảo:
{context}

Câu hỏi: {question}
Câu trả lời (bằng tiếng Việt):"""

        response = self.llm.invoke(prompt_text)
        answer = response.content

        # Lưu vào memory
        self.memory.save_context(
            {"input": question},
            {"output": answer}
        )

        return f"{answer}\n\n*Nguồn: {', '.join(sources)}*"

# Demo
if __name__ == "__main__":
    qa = DocumentQAChain()

    # Sample English documents
    qa.load_documents(
        texts=[
            "Python is a high-level programming language known for its simplicity...",
            "LangChain is a framework for building LLM-powered applications...",
        ],
        sources=["python_intro.txt", "langchain_docs.txt"]
    )

    # Hỏi bằng tiếng Việt
    print(qa.ask("Python là gì?"))
    print(qa.ask("Tại sao lại dùng LangChain?"))
    print(qa.ask("Nói thêm về điều đó"))  # Follow-up
```

### Giải bài tập 3: Code Review Pipeline

```python
# code_review_pipeline.py
import ast
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

@tool
def analyze_complexity(code: str) -> str:
    """
    Phân tích độ phức tạp của code Python.
    Đếm số nhánh điều kiện để tính cyclomatic complexity.
    """
    try:
        tree = ast.parse(code)

        # Đếm các nhánh phức tạp
        complexity = 1  # Base complexity
        branch_nodes = (
            ast.If, ast.While, ast.For, ast.ExceptHandler,
            ast.With, ast.Assert, ast.comprehension,
        )

        for node in ast.walk(tree):
            if isinstance(node, branch_nodes):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        # Đếm functions
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        level = "Thấp (tốt)" if complexity <= 5 else \
                "Trung bình" if complexity <= 10 else \
                "Cao (cần refactor)"

        return f"""Phân tích complexity:
- Cyclomatic complexity: {complexity} ({level})
- Số functions: {len(functions)}
- Khuyến nghị: {"OK" if complexity <= 10 else "Cần chia nhỏ hàm"}"""

    except SyntaxError as e:
        return f"Lỗi syntax: {e}"

@tool
def check_style(code: str) -> str:
    """
    Kiểm tra style code theo PEP8.
    Kiểm tra các quy tắc cơ bản không cần thư viện ngoài.
    """
    issues = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        # Kiểm tra line length
        if len(line) > 79:
            issues.append(f"Dòng {i}: Quá dài ({len(line)} ký tự, max 79)")

        # Kiểm tra trailing whitespace
        if line.rstrip() != line and line.strip():
            issues.append(f"Dòng {i}: Có khoảng trắng thừa ở cuối")

        # Kiểm tra tab vs space
        if '\t' in line:
            issues.append(f"Dòng {i}: Dùng tab thay vì 4 spaces")

    if not issues:
        return "Không có style issues! Code tuân theo PEP8."

    return f"Style issues ({len(issues)}):\n" + "\n".join(issues[:10])

# Agent setup
tools = [analyze_complexity, check_style]
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là Senior Code Reviewer. Khi review code:
1. Dùng analyze_complexity để kiểm tra độ phức tạp
2. Dùng check_style để kiểm tra style
3. Tự phân tích logic, naming, documentation
4. Tổng hợp thành Markdown report với sections:
   ## Summary, ## Issues (Critical/Major/Minor), ## Suggestions"""),
    ("human", "Review code này:\n```python\n{code}\n```"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

def review_code(code: str) -> str:
    """Review code và trả về Markdown report."""
    result = executor.invoke({"code": code})
    return result["output"]

# Test
sample_code = """
def process_data(data,flag,debug=False):
    result = []
    for item in data:
        if item > 0:
            if flag == True:
                if debug == True:
                    print(f"Processing: {item}")
                result.append(item * 2)
            else:
                result.append(item)
        else:
            if flag:
                result.append(0)
    return result
"""

report = review_code(sample_code)
print(report)
```
