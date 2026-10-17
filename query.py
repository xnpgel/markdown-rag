"""
步骤：从 Milvus 检索 top-k 切片 → 拼接 Prompt → 调用百炼生成回答
运行: python query.py "Your question here"
"""

import sys
from datetime import datetime
import httpx

from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from openai import OpenAI
from pymilvus import connections, Collection

import config


# ────────────────── Prompt templates ──────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a document-grounded Q&A assistant. 
You MUST answer the user's question **strictly based on the provided document excerpts** below.

Rules:
1. If the documents contain sufficient information, answer concisely and cite the source (e.g., [Source 1]).
2. If the documents only partially answer the question, provide what you can and clearly state what is missing.
3. If the documents do NOT contain the answer at all, say exactly: "Based on the provided documents, I cannot answer this question."
4. Do NOT make up any information that is not present in the documents.
5. Keep your answer clear and to the point.

--- Document excerpts ---
{context}"""


def load_vector_store():
    """Reconnect to the existing Milvus collection (read-only)."""
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={
            "host": config.MILVUS_HOST,
            "port": config.MILVUS_PORT,
        },
        collection_name=config.MILVUS_COLLECTION,
    )
    return vector_store


def retrieve_context(query: str, vector_store, top_k: int = config.RETRIEVAL_TOP_K):
    """Retrieve top-k most relevant chunks from Milvus, with all metadata."""
    
    try:
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    except Exception:
        pass  # Already connected

    # 1. 用 langchain 生成查询向量
    embedding_vector = vector_store.embeddings.embed_query(query)
    
    # 2. 用 pymilvus 原生接口检索，返回所有字段
    from pymilvus import Collection
    collection = Collection(config.MILVUS_COLLECTION)
    collection.load()
    
    results = collection.search(
        data=[embedding_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": config.SEARCH_NPROBE}},
        limit=top_k,
        output_fields=["*"]  # 返回所有字段，包括 source, h1, h2, h3
    )
    
    # 3. 把 Milvus 原生结果转成 langchain Document 格式（可选，方便后面用）
    docs = []
    for hits in results:
        for hit in hits:
            doc = type('Document', (), {})()
            doc.page_content = hit.entity.get('text', '')
            doc.metadata = {
                'source': hit.entity.get('source', 'unknown'),
                'h1': hit.entity.get('h1', ''),
                'h2': hit.entity.get('h2', ''),
                'h3': hit.entity.get('h3', ''),
                'pk': hit.entity.get('pk', '')
            }
            docs.append(doc)
    
    # 4. 打印信息
    print(f"Retrieved {len(docs)} chunks:\n")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        hierarchy = " > ".join(filter(None, [doc.metadata.get('h1'), doc.metadata.get('h2'), doc.metadata.get('h3')]))
        print(f"--- Chunk {i+1} ---")
        print(f"  Source : {source}")
        print(f"  Section: {hierarchy}")
        print(f"  Content: {doc.page_content[:200]}...\n")
    
    return docs


def build_context(docs) -> str:
    """Combine the retrieved chunks into one context string with source labels."""
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i+1} | {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, docs) -> str:
    """Call the Bailian LLM to generate an answer grounded on the retrieved documents."""
    context_text = build_context(docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)

    client = OpenAI(
        api_key=config.BAILIAN_API_KEY,
        base_url=config.BAILIAN_BASE_URL,
        http_client=httpx.Client(verify=False)  # 绕过 SSL 证书验证
    )

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        extra_body={"enable_thinking": config.ENABLE_THINKING},
    )
    return response.choices[0].message.content


def rag_qa(query: str) -> str:
    """Full RAG pipeline: retrieve → generate."""
    print("=" * 60)
    print(f"Question: {query}")
    print("=" * 60)

    vector_store = load_vector_store()
    docs = retrieve_context(query, vector_store)

    if not docs:
        return "No relevant documents found. Please check your vector database."

    print("Calling the LLM to generate an answer...\n")
    answer = generate_answer(query, docs)

    print("=" * 60)
    print(f"Answer:\n{answer}")
    print("=" * 60)
    return answer


if __name__ == "__main__":
    DEFAULT_QUERY = "what is DISKANN?"
    if len(sys.argv) < 2:
        print(f"No question provided, using default query: \"{DEFAULT_QUERY}\"")
        query = DEFAULT_QUERY
    else:
        query = " ".join(sys.argv[1:])

    rag_qa(query)