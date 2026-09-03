# ============================================================
# Configuration
# ============================================================

# Path to the saved FAISS index (relative to project root).
VECTOR_DB_PATH = "vectorstore/faiss_index"

# Number of chunks to retrieve per query.
TOP_K = 5

# OpenAI embedding model. Must match the model used to build the
# existing FAISS index (OpenAIEmbeddings() default in langchain-openai 1.2.1).
EMBEDDING_MODEL = "text-embedding-ada-002"

# Local Ollama model used for answer generation.
OLLAMA_MODEL = "rafw007/gemma4-e4b-claude-coder:latest"

# ------------------------------------------------------------
# Explore Episodes mode
# ------------------------------------------------------------

# Wide candidate pool fetched before grouping/filtering (post-hoc filter).
FETCH_K = 100

# Number of distinct episodes to return as cards.
N_EPISODES = 5

# Max chunks kept per episode to ground that episode's answer.
CHUNKS_PER_EPISODE = 3
