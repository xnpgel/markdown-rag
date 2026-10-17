# Project-RAG: Document Retrieval System

A Retrieval-Augmented Generation (RAG) system that chunks Markdown files by heading structure, stores embeddings in Milvus with IVF indexing, and answers questions using Alibaba Bailian LLM.

---

## Prerequisites

- **Python 3.12+**
- **Docker** (for Milvus)
- **Ollama** (for local embedding models, installed as a separate desktop application)
- **LangChain** (provides the framework for text splitting, embedding generation, Milvus integration, and LLM reasoning)

---

## Installation

### Pre-installed Services

The following services are already deployed and running:

| Service | Address |
|:---|:---|
| **Ollama** | `http://172.21.0.16:11434` |
| **Milvus** | `localhost:19530` (Docker) |

### Install Python Dependencies

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Set Up API Key

This project uses Alibaba Bailian API for LLM inference. You need to obtain an API Key and configure it as an environment variable.

### How to Get a Bailian API Key

1. Go to the [Alibaba Bailian console](https://www.aliyun.com/product/bailian) and sign in with your Alibaba Cloud account.
2. If it's your first time, click **"免费体验"** (Free Trial) and agree to the terms to activate the service. You'll receive **100 million+ free tokens** (valid for 180 days) across all models.
3. In the console, click your profile icon in the top-right corner, then select **"API-KEY"** from the dropdown menu.
4. Click **"创建API KEY"** (Create API KEY). Choose the **default business space** and optionally add a description.
5. **Copy and export your API Key immediately**: it will be hidden once you close the dialog.

### Configure the API Key

Set your API Key as an environment variable to avoid hardcoding it in your code:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

## Project Structure

| File | Description |
|:---|:---|
| `requirements.txt` | Python package dependencies (langchain, pymilvus, openai, etc.) |
| `markdown_files/` | Directory containing the source Markdown documents to be chunked and indexed |
| `config.py` | Central configuration for API keys, model names, service addresses, chunk parameters, retrieval settings, etc. |
| `ingest.py` | Loads Markdown files, splits them by heading hierarchy (`#`, `##`, `###`), generates embeddings via Ollama (`EmbeddingGemma` model), stores vectors in Milvus with an IVF index |
| `show_data.py` | Connects to Milvus and displays collection existence, entity count, and schema fields |
| `delete_collection.py` | Drops the Milvus collection. No longer needed, since `ingest.py` automatically overwrites the previous collection |
| `query.py` | Performs RAG: retrieves top-K relevant chunks from Milvus, builds a prompt, calls Bailian LLM to generate a grounded answer |

## Quick Start

### Step 1: Ingest Documents

```bash
python ingest.py
```

Expected output:
- Found N Markdown files
- Chunks created and stored in Milvus
- IVF index built with `nlist = √(entity_count)`

### Step 2: Verify Data

```bash
python show_data.py
```

Expected output:
```
Connecting to Milvus at localhost:19530...
Connected successfully.

Collection 'MilvusIndexMarkdown' exists.
Number of entities: 344
Description: 
Fields in schema:
  - text (21)
  - pk (5)
  - vector (101)
  - h1-h3: dynamic field.
See this example: ...
```

### Step 3: Ask a Question

```bash
python query.py "What is DISKANN?"
```

If no question is provided, a default query will be used.

Expected output:
```
============================================================
Question: what is DISKANN?
============================================================
Retrieved 5 chunks:

--- Chunk 1 ---
  Source : diskann.md
  Section: DISKANN > Overview > Search process
  Content: Once the index (the Vamana graph on disk and PQ codes in memory) is built, DISKANN performs ANN searches as follows:  
![Diskann 2](../../../../../assets/diskann-2.png)...

--- Chunk 2 ---
  Source : diskann.md
  Section: 
  Content: ---
id: diskann.md
title: "DISKANN"
summary: "In large-scale scenarios, where datasets can include billions or even trillions of vectors, standard in-memory indexing methods (e.g., HNSW, IVF_FLAT) oft...

--- Chunk 3 ---
  Source : diskann.md
  Section: DISKANN > Overview
  Content: **DISKANN** combines two key techniques for efficient vector search:  
- **Vamana Graph** – A **disk-based**, **graph-based** index that connects data points (or vectors) for efficient navigation duri...

--- Chunk 4 ---
  Source : aisaq.md
  Section: AISAQ > How AISAQ works > Foundation: DISKANN recap
  Content: In DISKANN, the raw vectors and edge lists are stored on disk, while PQ-compressed vectors are kept in memory (DRAM).  
When DISKANN traverses to a node (e.g., *vector 0*):  
- It loads the raw vector...

--- Chunk 5 ---
  Source : diskann.md
  Section: DISKANN > Overview > Search process
  Content: 1. **Query and entry point:** A query vector is provided to locate its nearest neighbors. DISKANN starts from a selected entry point in the Vamana graph, often a node near the global centroid of the d...

Calling the LLM to generate an answer...

============================================================
Answer:
DISKANN is a disk-based approach for efficient vector search in large-scale scenarios where datasets can include billions or even trillions of vectors [Source 2]. It combines two key techniques: the Vamana Graph, which is a disk-based, graph-based index for efficient navigation during search, and Product Quantization (PQ), an in-memory compression method that reduces vector size for quick approximate distance calculations [Source 3].
============================================================
```

### Step 4: Reset (Optional)

```bash
python delete_collection.py
```

Then re-run `python ingest.py` to rebuild.