# Vector Databases

## Embeddings là gì?

**Embedding** là một vector số học biểu diễn ý nghĩa ngữ nghĩa của văn bản trong không gian nhiều chiều. Các văn bản có ý nghĩa tương tự sẽ có vector gần nhau trong không gian đó.

```
"Con mèo ngồi trên thảm"  →  [0.23, -0.45, 0.12, ..., 0.67]  (1536 chiều)
"The cat sat on the mat"   →  [0.24, -0.44, 0.11, ..., 0.65]  (rất gần nhau!)
"Giá cổ phiếu hôm nay"    →  [-0.78, 0.34, -0.21, ..., 0.03] (rất khác)
```

### Tại sao cần Vector Database?

Khi bạn có hàng triệu documents, bạn cần tìm nhanh những document "gần nhất" với câu hỏi. Việc tính similarity cho tất cả documents là rất tốn kém. Vector databases dùng các thuật toán đặc biệt (HNSW, IVF) để tìm kiếm gần đúng nhưng nhanh hơn nhiều.

### So sánh Vector Databases phổ biến

| Database | Open Source | Cloud | Best For |
|---|---|---|---|
| **Chroma** | Có | Không | Development, prototype |
| **Pinecone** | Không | Có | Production, fully managed |
| **Weaviate** | Có | Có | Production, schema-based |
| **Qdrant** | Có | Có | Performance, filtering |
| **Milvus** | Có | Có | Large scale, enterprise |
| **pgvector** | Có | Có | Đã dùng PostgreSQL |

## Similarity Metrics

```python
import numpy as np

def cosine_similarity(v1: list, v2: list) -> float:
    """
    Đo góc giữa 2 vectors. Phổ biến nhất cho text.
    Range: -1 (ngược chiều) đến 1 (cùng chiều)
    """
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1: list, v2: list) -> float:
    """
    Khoảng cách Euclidean. Nhỏ hơn = gần hơn.
    Dùng khi magnitude của vector quan trọng.
    """
    v1, v2 = np.array(v1), np.array(v2)
    return np.linalg.norm(v1 - v2)

def dot_product(v1: list, v2: list) -> float:
    """
    Tích vô hướng. Dùng khi vectors đã được normalize.
    Nhanh hơn cosine vì không cần tính norm.
    """
    return np.dot(v1, v2)

# Demo
v1 = [0.23, -0.45, 0.12, 0.67]
v2 = [0.24, -0.44, 0.11, 0.65]  # Rất gần v1
v3 = [-0.78, 0.34, -0.21, 0.03]  # Rất khác

print(f"cos(v1, v2) = {cosine_similarity(v1, v2):.4f}")  # ~0.999
print(f"cos(v1, v3) = {cosine_similarity(v1, v3):.4f}")  # ~-0.2
```

## ChromaDB - Getting Started

```python
# pip install chromadb openai

import chromadb
from openai import OpenAI

openai_client = OpenAI()

# 1. Persistent client (lưu vào disk)
chroma_client = chromadb.PersistentClient(path="./my_vector_db")

# 2. In-memory client (chỉ dùng cho testing)
# chroma_client = chromadb.Client()

# Tạo collection
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={
        "hnsw:space": "cosine",        # Similarity metric
        "hnsw:construction_ef": 100,   # Build quality (cao hơn = chính xác hơn, chậm hơn)
        "hnsw:search_ef": 50,          # Search quality
    }
)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo embeddings cho danh sách texts."""
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]

# Thêm documents
documents = [
    "Python là ngôn ngữ lập trình bậc cao, dễ học và mạnh mẽ.",
    "Machine Learning là tập con của AI, cho phép máy tính học từ data.",
    "Deep Learning sử dụng neural networks nhiều lớp để học features phức tạp.",
    "LangChain giúp xây dựng ứng dụng AI với LLM một cách dễ dàng.",
    "Vector database lưu trữ và tìm kiếm embeddings hiệu quả.",
]

metadatas = [
    {"category": "programming", "difficulty": "beginner"},
    {"category": "ai", "difficulty": "beginner"},
    {"category": "ai", "difficulty": "intermediate"},
    {"category": "framework", "difficulty": "intermediate"},
    {"category": "database", "difficulty": "intermediate"},
]

ids = [f"doc_{i}" for i in range(len(documents))]

# Tạo embeddings
print("Đang tạo embeddings...")
embeddings = embed_texts(documents)

# Thêm vào Chroma
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids,
)
print(f"Đã thêm {len(documents)} documents!")
```

## Tìm kiếm trong Vector Database

```python
def semantic_search(
    query: str,
    collection,
    n_results: int = 3,
    filter_metadata: dict = None
) -> list[dict]:
    """
    Tìm kiếm semantic search.
    Trả về các documents tương tự nhất với query.
    """
    # Tạo embedding cho query
    query_embedding = embed_texts([query])[0]

    # Tìm kiếm
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }

    # Filter theo metadata nếu có
    if filter_metadata:
        kwargs["where"] = filter_metadata

    results = collection.query(**kwargs)

    # Format output
    formatted = []
    for i in range(len(results["documents"][0])):
        similarity = 1 - results["distances"][0][i]
        formatted.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity": round(similarity, 4),
        })

    return formatted

# Test searches
queries = [
    "Cách học lập trình",
    "AI và machine learning khác nhau thế nào?",
    "Framework nào dùng để build AI app?",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = semantic_search(query, collection, n_results=2)
    for r in results:
        print(f"  [{r['similarity']:.3f}] {r['document'][:60]}...")

# Search với filter
print("\n\nChỉ tìm trong category 'ai':")
ai_results = semantic_search(
    "Giải thích neural networks",
    collection,
    n_results=2,
    filter_metadata={"category": "ai"}
)
for r in ai_results:
    print(f"  [{r['similarity']:.3f}] {r['document']}")
```

## CRUD Operations

```python
# CREATE: Đã thấy ở trên với collection.add()

# READ: Lấy document theo ID
doc = collection.get(ids=["doc_0"])
print(doc["documents"][0])

# UPDATE: Cập nhật document
new_embedding = embed_texts(["Python là ngôn ngữ phổ biến nhất thế giới hiện nay."])[0]
collection.update(
    ids=["doc_0"],
    documents=["Python là ngôn ngữ phổ biến nhất thế giới hiện nay."],
    embeddings=[new_embedding],
    metadatas=[{"category": "programming", "difficulty": "beginner", "updated": True}]
)

# DELETE: Xóa document theo ID
collection.delete(ids=["doc_4"])

# DELETE theo filter
collection.delete(where={"difficulty": "advanced"})

# COUNT
count = collection.count()
print(f"Số documents: {count}")
```

## Pinecone - Production Vector Database

```python
# pip install pinecone-client

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

openai_client = OpenAI()
pc = Pinecone(api_key="your-pinecone-api-key")

# Tạo index
INDEX_NAME = "semantic-search"
EMBEDDING_DIM = 1536  # text-embedding-3-small

# Tạo index nếu chưa tồn tại
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

def upsert_documents(documents: list[dict]):
    """
    Upsert documents vào Pinecone.
    Mỗi document cần có: id, text, metadata (optional)
    """
    # Batch embedding (hiệu quả hơn gọi từng cái)
    texts = [doc["text"] for doc in documents]
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    embeddings = [item.embedding for item in response.data]

    # Upsert theo batch
    vectors = [
        {
            "id": doc["id"],
            "values": embeddings[i],
            "metadata": {
                "text": doc["text"],
                **doc.get("metadata", {})
            }
        }
        for i, doc in enumerate(documents)
    ]

    # Pinecone khuyến nghị batch size 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"Upserted {len(vectors)} vectors")

def search_pinecone(query: str, top_k: int = 5, filter: dict = None) -> list[dict]:
    """Tìm kiếm trong Pinecone."""
    query_embedding = openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter,
    )

    return [
        {
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
        }
        for match in results.matches
    ]
```

## Semantic Search System hoàn chỉnh

```python
class SemanticSearchEngine:
    """
    Engine tìm kiếm semantic sử dụng Chroma + OpenAI embeddings.
    Hỗ trợ: add, search, filter, delete.
    """

    def __init__(self, collection_name: str = "search_engine"):
        import chromadb
        from openai import OpenAI

        self.llm = OpenAI()
        self.client = chromadb.PersistentClient(path=f"./{collection_name}_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self.llm.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
        )
        return [item.embedding for item in response.data]

    def add(self, items: list[dict]) -> int:
        """
        Thêm items vào search engine.
        Mỗi item: {"id": str, "text": str, "metadata": dict}
        """
        texts = [item["text"] for item in items]
        embeddings = self._get_embeddings(texts)

        self.collection.add(
            ids=[item["id"] for item in items],
            documents=texts,
            embeddings=embeddings,
            metadatas=[item.get("metadata", {}) for item in items],
        )
        return len(items)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: dict = None
    ) -> list[dict]:
        """
        Tìm kiếm semantic.
        Hỗ trợ minimum similarity threshold và metadata filters.
        """
        embedding = self._get_embeddings([query])[0]

        kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        formatted = []
        for i in range(len(results["documents"][0])):
            similarity = 1 - results["distances"][0][i]
            if similarity >= min_similarity:
                formatted.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": round(similarity, 4),
                })

        return formatted

    def rag_answer(self, question: str, top_k: int = 4) -> dict:
        """Trả lời câu hỏi dựa trên tài liệu đã index."""
        relevant = self.search(question, top_k=top_k, min_similarity=0.5)

        if not relevant:
            return {"answer": "Không tìm thấy thông tin liên quan.", "sources": []}

        context = "\n\n".join([r["text"] for r in relevant])

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Trả lời dựa trên tài liệu. Ngắn gọn, chính xác."},
                {"role": "user", "content": f"Tài liệu:\n{context}\n\nCâu hỏi: {question}"}
            ],
            temperature=0.1,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": [r["metadata"] for r in relevant],
            "context_items": len(relevant),
        }

    def delete(self, ids: list[str] = None, filters: dict = None):
        """Xóa documents theo ID hoặc metadata filter."""
        if ids:
            self.collection.delete(ids=ids)
        elif filters:
            self.collection.delete(where=filters)
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: Product search engine

Xây dựng semantic search engine cho danh mục sản phẩm:
- Dataset: Tạo 20 sản phẩm giả (tên, mô tả, category, price)
- Index tất cả vào ChromaDB
- Implement search với filter theo category và price range
- So sánh kết quả semantic search vs keyword search (string matching)
- Đo thời gian của cả hai

### Bài tập 2: Duplicate detector

Sử dụng vector similarity để phát hiện duplicate content:
- Nhận input là list các text documents
- Tính embedding cho tất cả
- Tìm cặp documents có similarity > 0.9 (near-duplicates)
- Cluster các documents theo chủ đề (similarity > 0.7)
- Trả về report: duplicates, clusters, unique documents

### Bài tập 3: Personal knowledge base

Xây dựng personal knowledge base:
- Thêm notes/articles bằng CLI command
- Search tìm thông tin liên quan
- Tag system (filter theo tags)
- Export results ra Markdown file
- Stats: số documents, top categories, recently added

---SOLUTION---

## Lời giải

### Giải bài tập 1: Product Search Engine

```python
# product_search.py
import time
import chromadb
from openai import OpenAI

client = OpenAI()
chroma = chromadb.Client()

# Sample products
PRODUCTS = [
    {"id": "p1", "name": "iPhone 15 Pro", "description": "Smartphone cao cấp với chip A17 Pro, camera 48MP, titanium frame", "category": "smartphone", "price": 28000000},
    {"id": "p2", "name": "Samsung Galaxy S24", "description": "Android flagship với AI features, màn hình 6.2 inch Dynamic AMOLED", "category": "smartphone", "price": 22000000},
    {"id": "p3", "name": "MacBook Air M3", "description": "Laptop siêu mỏng nhẹ, chip M3, 18 giờ pin, màn hình Liquid Retina", "category": "laptop", "price": 35000000},
    {"id": "p4", "name": "Dell XPS 15", "description": "Laptop Windows cao cấp, OLED 4K, RTX 4060, phù hợp đồ họa và gaming", "category": "laptop", "price": 42000000},
    {"id": "p5", "name": "Sony WH-1000XM5", "description": "Tai nghe chống ồn hàng đầu, 30h pin, aptX HD, phù hợp làm việc", "category": "audio", "price": 8500000},
    {"id": "p6", "name": "AirPods Pro 2", "description": "True wireless earbuds với ANC, Spatial Audio, H2 chip, IPX4", "category": "audio", "price": 6500000},
    {"id": "p7", "name": "iPad Pro M4", "description": "Tablet màn hình OLED 11 inch, chip M4, hỗ trợ Apple Pencil Pro", "category": "tablet", "price": 25000000},
    {"id": "p8", "name": "Samsung Tab S9+", "description": "Android tablet cao cấp 12.4 inch, S Pen, Snapdragon 8 Gen 2", "category": "tablet", "price": 20000000},
    {"id": "p9", "name": "Logitech MX Master 3S", "description": "Chuột không dây cao cấp cho productivity, 8K DPI, 70 ngày pin", "category": "peripheral", "price": 2800000},
    {"id": "p10", "name": "Apple Watch Series 9", "description": "Smartwatch với chip S9, Double Tap gesture, health monitoring 24/7", "category": "wearable", "price": 12000000},
    {"id": "p11", "name": "Google Pixel 8 Pro", "description": "Android phone với AI camera, Tensor G3, 7 năm update, Gemini AI", "category": "smartphone", "price": 18000000},
    {"id": "p12", "name": "LG UltraWide 34 inch", "description": "Monitor 34 inch 3440x1440, IPS, 144Hz, phù hợp làm việc đa nhiệm", "category": "monitor", "price": 15000000},
    {"id": "p13", "name": "Bose QuietComfort 45", "description": "Over-ear headphones với ANC tốt, 24h pin, âm thanh balanced", "category": "audio", "price": 7200000},
    {"id": "p14", "name": "ASUS ROG Zephyrus G14", "description": "Gaming laptop AMD Ryzen 9, RTX 4070, 14 inch, 165Hz, 1.65kg", "category": "laptop", "price": 38000000},
    {"id": "p15", "name": "Kindle Paperwhite", "description": "E-reader 6.8 inch, không phản ánh sáng, 10 tuần pin, IPX8", "category": "ereader", "price": 3500000},
]

def create_search_engine():
    """Tạo và populate search engine."""
    collection = chroma.get_or_create_collection(
        name="products",
        metadata={"hnsw:space": "cosine"}
    )

    # Tạo search text cho mỗi sản phẩm
    search_texts = [
        f"{p['name']}. {p['description']}. Danh mục: {p['category']}. Giá: {p['price']:,}đ"
        for p in PRODUCTS
    ]

    # Tạo embeddings
    response = client.embeddings.create(
        input=search_texts,
        model="text-embedding-3-small",
    )
    embeddings = [item.embedding for item in response.data]

    # Add vào Chroma
    collection.add(
        ids=[p["id"] for p in PRODUCTS],
        documents=search_texts,
        embeddings=embeddings,
        metadatas=[{
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
        } for p in PRODUCTS]
    )

    return collection

def semantic_search(collection, query: str, top_k: int = 5,
                    category: str = None, max_price: int = None) -> tuple:
    """Semantic search với timing."""
    start = time.time()

    query_embedding = client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    ).data[0].embedding

    where = {}
    if category:
        where["category"] = category
    if max_price:
        where["price"] = {"$lte": max_price}

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    elapsed = time.time() - start

    formatted = [
        {
            "name": results["metadatas"][0][i]["name"],
            "category": results["metadatas"][0][i]["category"],
            "price": results["metadatas"][0][i]["price"],
            "similarity": round(1 - results["distances"][0][i], 4),
        }
        for i in range(len(results["documents"][0]))
    ]

    return formatted, elapsed

def keyword_search(query: str, top_k: int = 5) -> tuple:
    """Simple keyword search cho so sánh."""
    start = time.time()
    query_words = set(query.lower().split())

    scored = []
    for p in PRODUCTS:
        text = f"{p['name']} {p['description']}".lower()
        match_count = sum(1 for word in query_words if word in text)
        if match_count > 0:
            scored.append((match_count, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    elapsed = time.time() - start

    return [p for _, p in scored[:top_k]], elapsed

# Demo
collection = create_search_engine()

test_queries = [
    "tai nghe chống ồn cho dân văn phòng",
    "máy tính xách tay mỏng nhẹ pin trâu",
    "điện thoại chụp ảnh đẹp giá hợp lý",
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("--- Semantic Search ---")
    semantic_results, s_time = semantic_search(collection, query, top_k=3)
    for r in semantic_results:
        print(f"  [{r['similarity']:.3f}] {r['name']} - {r['price']:,}đ")
    print(f"  Thời gian: {s_time*1000:.1f}ms")

    print("--- Keyword Search ---")
    kw_results, k_time = keyword_search(query, top_k=3)
    for p in kw_results:
        print(f"  {p['name']} - {p['price']:,}đ")
    print(f"  Thời gian: {k_time*1000:.1f}ms")
```

### Giải bài tập 2: Duplicate Detector

```python
# duplicate_detector.py
import numpy as np
from openai import OpenAI
from itertools import combinations

client = OpenAI()

def get_all_embeddings(texts: list[str]) -> list[list[float]]:
    """Tạo embeddings cho tất cả texts cùng lúc (hiệu quả hơn)."""
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]

def cosine_similarity(v1, v2) -> float:
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def find_duplicates_and_clusters(
    documents: list[str],
    duplicate_threshold: float = 0.92,
    cluster_threshold: float = 0.72
) -> dict:
    """
    Phát hiện duplicates và cluster theo chủ đề.
    """
    print(f"Đang xử lý {len(documents)} documents...")

    # Tạo tất cả embeddings
    embeddings = get_all_embeddings(documents)

    # Tính similarity matrix
    n = len(documents)
    similarity_matrix = np.zeros((n, n))

    for i, j in combinations(range(n), 2):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        similarity_matrix[i][j] = sim
        similarity_matrix[j][i] = sim

    # Tìm duplicates (similarity > threshold)
    duplicates = []
    seen_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] >= duplicate_threshold:
                pair = (min(i, j), max(i, j))
                if pair not in seen_pairs:
                    duplicates.append({
                        "doc1_idx": i,
                        "doc2_idx": j,
                        "doc1": documents[i][:80] + "...",
                        "doc2": documents[j][:80] + "...",
                        "similarity": round(similarity_matrix[i][j], 4),
                    })
                    seen_pairs.add(pair)

    # Simple clustering với union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] >= cluster_threshold:
                union(i, j)

    # Group by cluster
    clusters = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    cluster_list = [
        {
            "docs": [documents[idx][:60] + "..." for idx in indices],
            "indices": indices,
            "size": len(indices),
        }
        for indices in clusters.values()
    ]
    cluster_list.sort(key=lambda x: x["size"], reverse=True)

    unique_count = sum(1 for c in cluster_list if c["size"] == 1)

    return {
        "total_docs": n,
        "duplicates": duplicates,
        "clusters": cluster_list,
        "unique_docs": unique_count,
        "duplicate_groups": len([c for c in cluster_list if c["size"] > 1]),
    }

# Test
test_docs = [
    "Python là ngôn ngữ lập trình phổ biến được dùng trong AI và data science.",
    "Python là một ngôn ngữ lập trình được dùng nhiều trong lĩnh vực AI và khoa học dữ liệu.",
    "Machine learning là kỹ thuật dạy máy tính học từ dữ liệu.",
    "Deep learning sử dụng neural network nhiều lớp để giải quyết bài toán phức tạp.",
    "Học máy (machine learning) cho phép máy tính tự học từ dữ liệu.",
    "JavaScript là ngôn ngữ lập trình chính cho web development phía client.",
    "React là thư viện JavaScript phổ biến để xây dựng giao diện người dùng.",
    "Vue.js là framework JavaScript nhẹ và dễ học cho frontend development.",
    "AI đang thay đổi cách chúng ta làm việc và sống trong thời đại số.",
]

result = find_duplicates_and_clusters(test_docs)
print(f"\n=== KẾT QUẢ ===")
print(f"Tổng documents: {result['total_docs']}")
print(f"Cặp duplicate: {len(result['duplicates'])}")
print(f"Clusters chủ đề: {result['duplicate_groups']}")
print(f"Documents unique: {result['unique_docs']}")

if result["duplicates"]:
    print(f"\nDuplicates tìm thấy:")
    for dup in result["duplicates"]:
        print(f"  [{dup['similarity']:.3f}] Doc {dup['doc1_idx']+1} ≈ Doc {dup['doc2_idx']+1}")
```

### Giải bài tập 3: Personal Knowledge Base

```python
# knowledge_base.py - CLI tool
import json
import os
import sys
import argparse
from datetime import datetime
import chromadb
from openai import OpenAI

client = OpenAI()
chroma = chromadb.PersistentClient(path="./personal_kb")
collection = chroma.get_or_create_collection(
    name="knowledge",
    metadata={"hnsw:space": "cosine"}
)

def get_embedding(text: str) -> list[float]:
    return client.embeddings.create(
        input=[text], model="text-embedding-3-small"
    ).data[0].embedding

def add_note(title: str, content: str, tags: list[str]):
    """Thêm note mới vào knowledge base."""
    doc_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    text = f"{title}\n\n{content}"
    embedding = get_embedding(text)

    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "title": title,
            "tags": ",".join(tags),
            "created": datetime.now().isoformat(),
        }]
    )
    print(f"Đã thêm: '{title}' [ID: {doc_id}]")

def search_notes(query: str, top_k: int = 5, tag: str = None) -> list[dict]:
    """Tìm kiếm notes theo semantic search."""
    embedding = get_embedding(query)
    kwargs = {
        "query_embeddings": [embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if tag:
        kwargs["where"] = {"tags": {"$contains": tag}}

    results = collection.query(**kwargs)

    return [
        {
            "title": results["metadatas"][0][i]["title"],
            "tags": results["metadatas"][0][i]["tags"].split(","),
            "created": results["metadatas"][0][i]["created"],
            "similarity": round(1 - results["distances"][0][i], 4),
            "preview": results["documents"][0][i][:150] + "...",
        }
        for i in range(len(results["documents"][0]))
    ]

def export_markdown(query: str = None) -> str:
    """Export notes ra Markdown."""
    all_notes = collection.get(include=["documents", "metadatas"])
    output = ["# Personal Knowledge Base\n"]
    output.append(f"*Xuất lúc: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    output.append(f"*Tổng số notes: {len(all_notes['documents'])}*\n\n")

    for i, (doc, meta) in enumerate(zip(all_notes["documents"], all_notes["metadatas"])):
        output.append(f"## {meta['title']}")
        output.append(f"**Tags:** {meta['tags']} | **Ngày:** {meta['created'][:10]}\n")
        output.append(doc.split('\n\n', 1)[1] if '\n\n' in doc else doc)
        output.append("\n---\n")

    return "\n".join(output)

def get_stats() -> dict:
    all_notes = collection.get(include=["metadatas"])
    tags = {}
    for meta in all_notes["metadatas"]:
        for tag in meta["tags"].split(","):
            if tag.strip():
                tags[tag.strip()] = tags.get(tag.strip(), 0) + 1
    return {
        "total": len(all_notes["metadatas"]),
        "top_tags": sorted(tags.items(), key=lambda x: x[1], reverse=True)[:5],
    }

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Personal Knowledge Base CLI")
    subparsers = parser.add_subparsers(dest="command")

    # add command
    add_parser = subparsers.add_parser("add", help="Thêm note mới")
    add_parser.add_argument("title")
    add_parser.add_argument("content")
    add_parser.add_argument("--tags", nargs="+", default=[])

    # search command
    search_parser = subparsers.add_parser("search", help="Tìm kiếm")
    search_parser.add_argument("query")
    search_parser.add_argument("--tag", default=None)
    search_parser.add_argument("--top", type=int, default=5)

    # export command
    subparsers.add_parser("export", help="Export ra Markdown")
    subparsers.add_parser("stats", help="Thống kê")

    args = parser.parse_args()

    if args.command == "add":
        add_note(args.title, args.content, args.tags)
    elif args.command == "search":
        results = search_notes(args.query, args.top, args.tag)
        for r in results:
            print(f"[{r['similarity']:.3f}] {r['title']} (tags: {','.join(r['tags'])})")
            print(f"  {r['preview']}\n")
    elif args.command == "export":
        md = export_markdown()
        with open("knowledge_export.md", "w", encoding="utf-8") as f:
            f.write(md)
        print("Đã export ra knowledge_export.md")
    elif args.command == "stats":
        stats = get_stats()
        print(f"Tổng notes: {stats['total']}")
        print(f"Top tags: {stats['top_tags']}")
    else:
        parser.print_help()
```
