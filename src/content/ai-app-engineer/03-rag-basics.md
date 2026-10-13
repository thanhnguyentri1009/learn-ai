# RAG - Retrieval Augmented Generation

## RAG là gì?

**RAG (Retrieval Augmented Generation)** là kỹ thuật kết hợp khả năng sinh văn bản của LLM với việc truy xuất thông tin từ nguồn dữ liệu bên ngoài. Thay vì chỉ dựa vào kiến thức được huấn luyện, model có thể tra cứu tài liệu cụ thể và trả lời dựa trên thông tin đó.

### Tại sao cần RAG?

**Vấn đề với LLM thuần túy:**
- **Knowledge cutoff**: Model không biết thông tin sau ngày huấn luyện
- **Hallucination**: Model có thể bịa ra thông tin sai
- **Private data**: Model không biết về tài liệu nội bộ của công ty bạn
- **Context limit**: Không thể nhét cả database vào prompt

**RAG giải quyết bằng cách:**
1. Lưu tài liệu của bạn vào vector database
2. Khi có câu hỏi, tìm các đoạn tài liệu liên quan
3. Đưa các đoạn đó vào context của LLM
4. LLM trả lời dựa trên tài liệu thực tế

```
Câu hỏi user
    │
    ▼
[Embedding Model]
    │ Chuyển thành vector
    ▼
[Vector Database]
    │ Tìm top-k tài liệu liên quan
    ▼
[Context Builder]
    │ Ghép tài liệu + câu hỏi thành prompt
    ▼
[LLM]
    │ Generate câu trả lời dựa trên context
    ▼
Câu trả lời
```

## Các thành phần của RAG

### 1. Document Loader

```python
# Cài đặt thư viện
# pip install openai chromadb pypdf sentence-transformers tiktoken

import os
from pathlib import Path

def load_text_file(file_path: str) -> str:
    """Load text từ file .txt."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_pdf(file_path: str) -> str:
    """Load text từ file PDF."""
    import pypdf
    text = ""
    with open(file_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_documents_from_folder(folder_path: str) -> list[dict]:
    """Load tất cả documents từ một folder."""
    documents = []
    folder = Path(folder_path)

    for file_path in folder.rglob("*"):
        if file_path.suffix == '.txt':
            content = load_text_file(str(file_path))
            documents.append({
                "content": content,
                "metadata": {"source": str(file_path), "type": "text"}
            })
        elif file_path.suffix == '.pdf':
            content = load_pdf(str(file_path))
            documents.append({
                "content": content,
                "metadata": {"source": str(file_path), "type": "pdf"}
            })

    return documents
```

### 2. Text Chunking

```python
import re
from typing import Generator

def chunk_by_tokens(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Chia text thành các chunks dựa trên số từ.
    overlap giúp giữ context ở ranh giới giữa các chunks.
    """
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        # Lấy chunk_size từ, bắt đầu từ vị trí i
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        # Di chuyển tới chunk tiếp theo, có overlap
        i += chunk_size - overlap

    return chunks

def chunk_by_paragraph(text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Chia text theo đoạn văn (paragraph).
    Hiệu quả hơn vì giữ nguyên context của mỗi đoạn.
    """
    # Tách theo dòng trống
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Nếu thêm paragraph này vẫn trong giới hạn, thêm vào
        if len(current_chunk) + len(paragraph) < max_chunk_size:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        else:
            # Lưu chunk hiện tại và bắt đầu chunk mới
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> list[str]:
    """Chia text theo câu, mỗi chunk có số câu cố định."""
    # Tách câu đơn giản (có thể dùng nltk cho chính xác hơn)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk)

    return chunks
```

### 3. Embedding và Vector Database với Chroma

```python
import chromadb
from openai import OpenAI

client = OpenAI()

# Khởi tạo ChromaDB (lưu locally)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_embedding(text: str) -> list[float]:
    """
    Tạo embedding vector cho text.
    Ada-002 có 1536 dimensions, text-embedding-3-small có 1536 dimensions.
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",  # Rẻ hơn, đủ tốt cho hầu hết tasks
    )
    return response.data[0].embedding

def create_vector_store(collection_name: str):
    """Tạo hoặc lấy collection từ ChromaDB."""
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Dùng cosine similarity
    )
    return collection

def add_documents_to_store(
    collection,
    documents: list[str],
    metadatas: list[dict] = None,
    ids: list[str] = None
):
    """
    Thêm documents vào vector store.
    ChromaDB tự động tạo embedding nếu dùng embedding function,
    hoặc ta có thể tự tạo embedding.
    """
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]

    if metadatas is None:
        metadatas = [{}] * len(documents)

    # Tạo embeddings cho tất cả documents
    print(f"Đang tạo embeddings cho {len(documents)} chunks...")
    embeddings = []
    for i, doc in enumerate(documents):
        embedding = get_embedding(doc)
        embeddings.append(embedding)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(documents)} done")

    # Thêm vào ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Đã thêm {len(documents)} chunks vào vector store!")

def search_documents(
    collection,
    query: str,
    n_results: int = 5
) -> list[dict]:
    """
    Tìm kiếm documents liên quan đến query.
    Trả về top-k kết quả theo cosine similarity.
    """
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format kết quả
    formatted_results = []
    for i in range(len(results["documents"][0])):
        formatted_results.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
        })

    return formatted_results
```

### 4. RAG Pipeline hoàn chỉnh

```python
class RAGSystem:
    """
    Hệ thống RAG hoàn chỉnh cho Q&A dựa trên tài liệu.
    """

    def __init__(self, collection_name: str = "my_documents"):
        self.llm_client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_document(self, text: str, source: str = "unknown"):
        """Xử lý và index một tài liệu vào vector store."""
        # 1. Chunk document
        chunks = chunk_by_paragraph(text, max_chunk_size=800)
        print(f"Đã chia thành {len(chunks)} chunks")

        # 2. Chuẩn bị metadata
        metadatas = [{"source": source, "chunk_index": i}
                     for i in range(len(chunks))]

        # 3. Tạo unique IDs
        import hashlib
        ids = [f"{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
               for chunk in chunks]

        # 4. Thêm vào vector store
        add_documents_to_store(self.collection, chunks, metadatas, ids)

    def answer_question(
        self,
        question: str,
        n_context: int = 4,
        temperature: float = 0.1
    ) -> dict:
        """
        Trả lời câu hỏi dựa trên tài liệu đã index.

        Returns:
            dict với keys: answer, sources, context_used
        """
        # 1. Tìm tài liệu liên quan
        relevant_docs = search_documents(self.collection, question, n_results=n_context)

        if not relevant_docs:
            return {
                "answer": "Tôi không tìm thấy thông tin liên quan trong tài liệu.",
                "sources": [],
                "context_used": ""
            }

        # 2. Build context từ các tài liệu
        context_parts = []
        sources = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"[Tài liệu {i}]\n{doc['content']}")
            source = doc['metadata'].get('source', 'unknown')
            if source not in sources:
                sources.append(source)

        context = "\n\n---\n\n".join(context_parts)

        # 3. Tạo prompt với context
        prompt = f"""Hãy trả lời câu hỏi dựa CHÍNH XÁC vào các tài liệu được cung cấp.
Nếu tài liệu không có thông tin để trả lời, hãy nói "Tài liệu không có thông tin về điều này."
KHÔNG suy đoán hoặc thêm thông tin không có trong tài liệu.

=== TÀI LIỆU THAM KHẢO ===
{context}

=== CÂU HỎI ===
{question}

=== CÂU TRẢ LỜI ==="""

        # 4. Gọi LLM
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là trợ lý Q&A chuyên nghiệp. Chỉ trả lời dựa trên tài liệu được cung cấp."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "docs_retrieved": len(relevant_docs),
        }
```

### 5. Sử dụng RAG System

```python
def demo_rag():
    """Demo đầy đủ của RAG system."""

    # Khởi tạo
    rag = RAGSystem(collection_name="company_docs")

    # Sample documents (thực tế sẽ load từ file)
    company_policy = """
    Chính sách làm việc của Công ty ABC

    1. Giờ làm việc
    Nhân viên làm việc từ 8:30 đến 17:30, thứ Hai đến thứ Sáu.
    Giờ nghỉ trưa: 12:00 - 13:00.
    Làm việc từ xa (remote) được phép tối đa 3 ngày/tuần sau khi được approval.

    2. Nghỉ phép
    Nhân viên được 12 ngày nghỉ phép/năm.
    Nghỉ bệnh: tối đa 10 ngày/năm, cần giấy tờ từ bác sĩ từ ngày thứ 3.
    Nghỉ phép phải đăng ký trước ít nhất 3 ngày làm việc.

    3. Chế độ đãi ngộ
    Lương tháng 13 vào tháng 12 hàng năm.
    Thưởng hiệu suất theo quý: 0-20% lương tháng.
    Bảo hiểm sức khỏe: đóng 100% premium cho nhân viên.

    4. Quy định IT
    Mật khẩu phải thay đổi mỗi 90 ngày.
    Không được cài phần mềm chưa được phê duyệt.
    Báo cáo ngay khi phát hiện sự cố bảo mật.
    """

    product_guide = """
    Hướng dẫn sản phẩm TechDevice Pro

    Cài đặt:
    1. Sạc đầy pin trước khi sử dụng lần đầu (khoảng 2 tiếng).
    2. Bật thiết bị bằng cách giữ nút nguồn 3 giây.
    3. Kết nối WiFi trong Settings > Network > WiFi.
    4. Cập nhật firmware: Settings > System > Software Update.

    Xử lý sự cố:
    - Thiết bị không bật: kiểm tra pin, sạc ít nhất 30 phút trước khi thử lại.
    - WiFi không kết nối: restart router, quên network rồi kết nối lại.
    - Thiết bị chạy chậm: Settings > Storage > Clear Cache.
    - Màn hình đen: giữ nút nguồn + volume down 10 giây để reset.

    Bảo hành:
    - Bảo hành chính hãng 24 tháng kể từ ngày mua.
    - Không bảo hành: rơi vỡ, ngấm nước, tự sửa chữa.
    - Liên hệ: support@techdevice.com hoặc 1800-888-999.
    """

    # Index documents
    print("Đang index tài liệu...")
    rag.index_document(company_policy, source="company_policy.txt")
    rag.index_document(product_guide, source="product_guide.txt")

    # Test Q&A
    questions = [
        "Nhân viên được nghỉ phép bao nhiêu ngày mỗi năm?",
        "Làm thế nào để cập nhật firmware của TechDevice Pro?",
        "Chính sách remote work như thế nào?",
        "Thiết bị bị màn hình đen phải làm gì?",
        "Công ty có chính sách thưởng không?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = rag.answer_question(question)
        print(f"A: {result['answer']}")
        print(f"Nguồn: {', '.join(result['sources'])}")

if __name__ == "__main__":
    demo_rag()
```

## Advanced: Reranking

```python
def rerank_results(query: str, documents: list[str], top_k: int = 3) -> list[str]:
    """
    Rerank documents sử dụng LLM để chọn ra những đoạn liên quan nhất.
    Cross-encoder approach: chính xác hơn nhưng chậm hơn.
    """
    # Format documents
    docs_text = "\n\n".join([
        f"[Doc {i+1}]: {doc[:300]}..."
        for i, doc in enumerate(documents)
    ])

    prompt = f"""Xếp hạng các đoạn văn sau theo mức độ liên quan đến câu hỏi.
Chỉ trả về số thứ tự của top {top_k} đoạn liên quan nhất, cách nhau bởi dấu phẩy.
Ví dụ: "3,1,5"

Câu hỏi: {query}

Các đoạn văn:
{docs_text}

Top {top_k} liên quan nhất (chỉ trả về số):"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )

    # Parse kết quả
    indices_str = response.choices[0].message.content.strip()
    try:
        indices = [int(i.strip()) - 1 for i in indices_str.split(",")]
        return [documents[i] for i in indices if 0 <= i < len(documents)]
    except:
        return documents[:top_k]
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: Q&A system đơn giản

Xây dựng một RAG system đơn giản (không cần vector DB thực sự, dùng brute-force search):
- Load text từ một file `.txt` bất kỳ (ví dụ: README, documentation)
- Chunk text thành các đoạn nhỏ
- Khi user hỏi, tìm đoạn liên quan nhất bằng cách dùng LLM để score
- Trả lời dựa trên đoạn tìm được
- Yêu cầu: KHÔNG dùng vector database, chỉ dùng string matching đơn giản hoặc LLM scoring

### Bài tập 2: Multi-document Q&A với nguồn tham khảo

Nâng cấp từ bài 1:
- Hỗ trợ nhiều files (txt, có thể thêm PDF)
- Mỗi câu trả lời phải kèm theo nguồn tài liệu (tên file, số trang/đoạn)
- Nếu không tìm thấy thông tin, phải nói rõ không có trong tài liệu
- Có confidence score cho câu trả lời

### Bài tập 3: Conversational RAG

Xây dựng RAG system hỗ trợ multi-turn conversation:
- Giữ lịch sử hội thoại
- Có thể hỏi follow-up questions ("Nói thêm về điểm thứ 2 đó")
- Tham chiếu câu trả lời trước trong context
- Có lệnh `/reset` để bắt đầu conversation mới

---SOLUTION---

## Lời giải

### Giải bài tập 1: Q&A system đơn giản

```python
# simple_rag.py
import os
from openai import OpenAI

client = OpenAI()

def load_and_chunk_text(file_path: str, chunk_size: int = 300) -> list[dict]:
    """Load file và chia thành chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Chia theo paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Merge paragraph nhỏ, giữ paragraph lớn
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size * 5:  # ~5 chars per word
            current += "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return [{"id": i, "content": c} for i, c in enumerate(chunks)]

def find_relevant_chunks(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Tìm chunks liên quan sử dụng LLM scoring (không cần vector DB).
    Đây là brute-force approach, phù hợp cho small datasets.
    """
    if len(chunks) <= top_k:
        return chunks

    # Score từng chunk
    chunks_text = "\n\n".join([
        f"Chunk {c['id']}: {c['content'][:200]}..."
        for c in chunks
    ])

    prompt = f"""Dưới đây là {len(chunks)} đoạn văn được đánh số.
Câu hỏi: "{query}"

Hãy liệt kê ID của {top_k} đoạn liên quan NHẤT đến câu hỏi.
Chỉ trả về các số ID cách nhau bằng dấu phẩy. Ví dụ: "2,5,1"

Các đoạn:
{chunks_text}

IDs liên quan nhất:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )

    # Parse IDs
    try:
        ids_str = response.choices[0].message.content.strip()
        ids = [int(x.strip()) for x in ids_str.split(",")]
        chunk_map = {c["id"]: c for c in chunks}
        return [chunk_map[i] for i in ids if i in chunk_map]
    except:
        return chunks[:top_k]

def answer_from_chunks(question: str, relevant_chunks: list[dict]) -> str:
    """Trả lời câu hỏi dựa trên các chunks liên quan."""
    context = "\n\n---\n\n".join([c["content"] for c in relevant_chunks])

    prompt = f"""Trả lời câu hỏi dựa trên tài liệu sau:

{context}

Câu hỏi: {question}

Nếu tài liệu không đủ thông tin, hãy nói rõ điều đó."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Chỉ trả lời dựa trên tài liệu được cung cấp."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

class SimpleRAG:
    def __init__(self, file_path: str):
        print(f"Loading {file_path}...")
        self.chunks = load_and_chunk_text(file_path)
        self.source = os.path.basename(file_path)
        print(f"Loaded {len(self.chunks)} chunks from {self.source}")

    def ask(self, question: str) -> str:
        print(f"Tìm thông tin liên quan...")
        relevant = find_relevant_chunks(question, self.chunks)
        print(f"Tìm thấy {len(relevant)} chunks liên quan")
        return answer_from_chunks(question, relevant)

# Sử dụng
# Tạo file test.txt với nội dung bất kỳ
with open("test.txt", "w", encoding="utf-8") as f:
    f.write("""Python là ngôn ngữ lập trình bậc cao...
    [Nội dung tài liệu của bạn ở đây]""")

rag = SimpleRAG("test.txt")
print(rag.ask("Python được dùng để làm gì?"))
```

### Giải bài tập 2: Multi-document Q&A

```python
# multi_doc_rag.py
import os
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()

class MultiDocRAG:
    def __init__(self):
        self.documents = []  # List of {content, source, chunk_id}

    def add_document(self, file_path: str):
        """Thêm một document vào hệ thống."""
        path = Path(file_path)
        source = path.name

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Chunk
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]

        for i, para in enumerate(paragraphs):
            self.documents.append({
                "content": para,
                "source": source,
                "chunk_id": i,
                "file_path": file_path
            })

        print(f"Đã thêm {len(paragraphs)} chunks từ {source}")

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """Tìm kiếm các chunks liên quan."""
        if not self.documents:
            return []

        # Score tất cả documents
        scored = []
        for doc in self.documents:
            score_prompt = f"""Mức độ liên quan của đoạn này với câu hỏi (0-10)?
Câu hỏi: {query}
Đoạn: {doc['content'][:300]}
Chỉ trả về con số:"""

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": score_prompt}],
                temperature=0,
                max_tokens=3,
            )
            try:
                score = float(resp.choices[0].message.content.strip())
            except:
                score = 0

            scored.append((score, doc))

        # Sort by score và lấy top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 3]

    def answer(self, question: str) -> dict:
        """Trả lời câu hỏi với nguồn tham khảo."""
        relevant = self.search(question)

        if not relevant:
            return {
                "answer": "Không tìm thấy thông tin liên quan trong tài liệu.",
                "sources": [],
                "confidence": 0.0
            }

        # Build context với source labels
        context_parts = []
        sources = []
        for doc in relevant:
            label = f"[{doc['source']}, đoạn {doc['chunk_id']+1}]"
            context_parts.append(f"{label}\n{doc['content']}")
            source_info = f"{doc['source']} (đoạn {doc['chunk_id']+1})"
            if source_info not in sources:
                sources.append(source_info)

        context = "\n\n".join(context_parts)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Trả lời dựa trên tài liệu. Trích dẫn nguồn khi cần. Nếu không đủ thông tin, nói rõ."
                },
                {
                    "role": "user",
                    "content": f"Tài liệu:\n{context}\n\nCâu hỏi: {question}"
                }
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content

        # Tính confidence dựa trên số lượng chunks tìm được
        confidence = min(len(relevant) / 4, 1.0)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "chunks_used": len(relevant)
        }

# Demo
rag = MultiDocRAG()
# rag.add_document("doc1.txt")
# rag.add_document("doc2.txt")
# result = rag.answer("Câu hỏi của bạn?")
# print(result["answer"])
# print(f"Nguồn: {result['sources']}")
# print(f"Confidence: {result['confidence']:.0%}")
```

### Giải bài tập 3: Conversational RAG

```python
# conversational_rag.py
from openai import OpenAI
from simple_rag import load_and_chunk_text, find_relevant_chunks

client = OpenAI()

class ConversationalRAG:
    def __init__(self, file_paths: list[str]):
        # Load tất cả documents
        self.chunks = []
        for path in file_paths:
            chunks = load_and_chunk_text(path)
            for chunk in chunks:
                chunk["source"] = path
            self.chunks.extend(chunks)

        # Conversation history
        self.history = []
        self.system_prompt = """Bạn là trợ lý Q&A. Trả lời dựa trên tài liệu được cung cấp.
Nếu câu hỏi liên quan đến cuộc hội thoại trước, hãy sử dụng context đó.
Trích dẫn cụ thể từ tài liệu khi có thể."""

    def _reformulate_query(self, question: str) -> str:
        """
        Nếu câu hỏi là follow-up, expand it với context từ history.
        Ví dụ: "Nói thêm về điều đó" -> "Nói thêm về [topic từ câu trước]"
        """
        if not self.history:
            return question

        # Chỉ dùng 2 lượt gần nhất
        recent_history = self.history[-4:]
        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}"
            for msg in recent_history
        ])

        reformulate_prompt = f"""Dựa vào lịch sử hội thoại, reformulate câu hỏi thành câu đầy đủ.
Nếu câu hỏi đã rõ ràng, giữ nguyên.

Lịch sử:
{history_text}

Câu hỏi hiện tại: {question}

Câu hỏi đầy đủ (chỉ trả về câu hỏi):"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": reformulate_prompt}],
            temperature=0,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()

    def chat(self, user_message: str) -> str:
        """Xử lý tin nhắn và trả lời."""
        # Lệnh reset
        if user_message.strip().lower() == '/reset':
            self.history = []
            return "Đã reset hội thoại. Bạn có thể bắt đầu câu hỏi mới."

        # Reformulate query nếu là follow-up
        search_query = self._reformulate_query(user_message)

        # Tìm relevant chunks
        relevant = find_relevant_chunks(search_query, self.chunks)

        # Build context
        context = "\n\n---\n\n".join([c["content"] for c in relevant])

        # Thêm user message vào history
        self.history.append({"role": "user", "content": user_message})

        # Gọi API với full history
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Tài liệu tham khảo:\n{context}"},
        ] + self.history

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        # Lưu vào history
        self.history.append({"role": "assistant", "content": answer})

        return answer

# Terminal interface
def run_conversational_rag(file_paths: list[str]):
    rag = ConversationalRAG(file_paths)
    print("Conversational RAG sẵn sàng! Gõ /reset để bắt đầu lại, /quit để thoát.")

    while True:
        user_input = input("\nBạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() == '/quit':
            break

        response = rag.chat(user_input)
        print(f"\nAssistant: {response}")
```
