"""
步骤：加载 Markdown → 切片 → 生成 Embedding → 存入 Milvus
运行：python ingest.py
"""

import os, math
import glob

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from pymilvus import CollectionSchema, FieldSchema, DataType

import config


def load_and_chunk_markdown(markdown_dir: str):
    """
    Load all .md files, split by heading levels,
    then sub-split chunks that exceed the token limit.
    """
    all_chunks = []
    md_files = glob.glob(os.path.join(markdown_dir, "**/*.md"), recursive=True)

    if not md_files:
        print(f"No .md files found in `{markdown_dir}`.")
        return []

    print(f"Found {len(md_files)} Markdown file(s). Starting to chunk...")

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by heading hierarchy
        md_splitter = MarkdownHeaderTextSplitter(config.HEADERS_TO_SPLIT)
        md_splits = md_splitter.split_text(content)

        # Sub-split long chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        final_splits = text_splitter.split_documents(md_splits)

        # Attach source filename to metadata
        for chunk_id,chunk in enumerate(final_splits):
            chunk.metadata["source"] = os.path.basename(md_file)
            print(f"--- Chunk #{chunk_id+1} from {os.path.basename(md_file)} ---")
            print(chunk.page_content[:200])  # 只打印前 200 字符，避免刷屏
            print("...")

        all_chunks.extend(final_splits)
        print(f"  {os.path.basename(md_file)} → {len(final_splits)} chunks")

    print(f"Chunking done. {len(all_chunks)} chunks in total.\n")
    return all_chunks


def init_embeddings():
    """Connect to the remote Ollama embedding service."""
    print(f"Connecting to Ollama at {config.OLLAMA_BASE_URL}")
    print(f"Embedding model: {config.EMBEDDING_MODEL}")

    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    # Quick connectivity test
    test_vec = embeddings.embed_query("connection test")
    print(f"  Ollama connected. Vector dimension: {len(test_vec)}\n")
    return embeddings


def store_to_milvus(chunks, embeddings):
    """Embed the chunks and store them into Milvus."""
    print(f"Connecting to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    print(f"Storing {len(chunks)} chunks into Milvus collection `{config.MILVUS_COLLECTION}`...")
    
    # Define Schema
    # fields = [
    #     FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100), # primary key
    #     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    #     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),  # EmbeddingGemma outputs 768 dimensions
    #     FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
    #     FieldSchema(name="h1", dtype=DataType.VARCHAR, max_length=256),
    #     FieldSchema(name="h2", dtype=DataType.VARCHAR, max_length=256),
    #     FieldSchema(name="h3", dtype=DataType.VARCHAR, max_length=256),
    # ]
    # schema = CollectionSchema(fields, description="Markdown document chunks")

    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={
            "host": config.MILVUS_HOST,
            "port": config.MILVUS_PORT,
        },
        collection_name=config.MILVUS_COLLECTION,
        drop_old=True,                               # Start fresh each time for demo purposes
        enable_dynamic_field=True,
    )
    vector_store.col.flush()
    print(f"  Flushed. Entity count: {vector_store.col.num_entities}")
    print("  Successfully stored.\n")

    # Create IVF_FLAT index on the vector field
    if vector_store.col.has_index():
        vector_store.col.release()           # Must release before dropping index
        vector_store.col.drop_index()
        print("  Dropped old index.")
        
    print(f"  Creating IVF index...")
    nlist = int(math.sqrt(vector_store.col.num_entities))
    nlist = max(1, nlist)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {
            "nlist": nlist  # Number of clusters (√n is a good starting point)
        }
    }
    vector_store.col.create_index(field_name="vector", index_params=index_params)

    # Load collection into memory for searching
    vector_store.col.load()
    print(f"  IVF index created and collection loaded into memory.")

    return vector_store


if __name__ == "__main__":
    chunks = load_and_chunk_markdown(config.MARKDOWN_DIR)
    if not chunks:
        exit(1)

    embeddings = init_embeddings()
    store_to_milvus(chunks, embeddings)
    print("\n" + "=" * 60)
    print("Ingestion complete! Next steps:")
    print(f"  1. Run `python query.py` to ask questions.")
    print(f"  2. Run `python show_data.py` to inspect the collection.")
    print("=" * 60)