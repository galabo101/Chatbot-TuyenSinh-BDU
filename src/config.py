from pathlib import Path

# Đường dẫn
PROJECT_ROOT = Path(__file__).parent.parent
QDRANT_PATH = str(PROJECT_ROOT / "qdrant_data")
DATA_DIR = str(PROJECT_ROOT / "data")
CHUNKS_FILE = str(PROJECT_ROOT / "data" / "chunks.jsonl")

# Embedding models
EMBEDDING_MODELS = {
    "gemma": {
        "name": "google/embeddinggemma-300m",
        "dimension": 768,
        "collection_name": "bdu_chunks_gemma"
    }
}

# Retrieval
TOP_K_INITIAL = 4
TOP_K_FINAL = 2
RELEVANCE_THRESHOLD = 0.5

# LLM
LLM_MODEL = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b"
]
TEMPERATURE = 0.5
MAX_TOKENS = 1024