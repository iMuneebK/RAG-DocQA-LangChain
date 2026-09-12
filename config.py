import os
from dotenv import load_dotenv

load_dotenv()

# Embedding Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# FAISS Vector Store path
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "faiss_index")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
