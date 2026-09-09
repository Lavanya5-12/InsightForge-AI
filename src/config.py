import os

# ============================================================
# OLLAMA CONFIGURATION
# ============================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

# ============================================================
# CHUNKING CONFIGURATION
# ============================================================
DEFAULT_CHUNK_SIZE = 500  # words
DEFAULT_CHUNK_OVERLAP = 100  # words

# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================
DEFAULT_TOP_K = 5
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

# Retrieval confidence / relevance threshold
# Chunks with hybrid score below this value are considered irrelevant/out-of-scope.
RETRIEVAL_THRESHOLD = 0.30

# ============================================================
# FAISS STORAGE CONFIGURATION
# ============================================================
FAISS_PATH = os.path.join("data", "faiss_db")
INDEX_FILE = os.path.join(FAISS_PATH, "index.faiss")
METADATA_FILE = os.path.join(FAISS_PATH, "metadata.json")
