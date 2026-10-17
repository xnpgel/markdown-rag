"""
统一配置文件
修改这里的参数即可适配不同环境
"""

import os

# ==================== 路径配置 ====================
MARKDOWN_DIR = "./markdown_files"  # Markdown 文件存放目录

# ==================== Ollama 配置 ====================
OLLAMA_BASE_URL = "http://172.21.0.16:11434"  # 换你实际的 IP
EMBEDDING_MODEL = "embeddinggemma"

# ==================== Milvus 配置 ====================
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "MilvusIndexMarkdown"

# ==================== 百炼 API 配置 ====================
BAILIAN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not BAILIAN_API_KEY:
    raise ValueError(
        "DASHSCOPE_API_KEY is not set. "
        "Please set it via environment variable: export DASHSCOPE_API_KEY='your-key'"
    )
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen3-32b"

# ==================== 切片参数 ====================
HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ==================== 检索参数 ====================
RETRIEVAL_TOP_K = 5
SEARCH_NPROBE = 8

# ==================== 深度思考 ====================
ENABLE_THINKING = False