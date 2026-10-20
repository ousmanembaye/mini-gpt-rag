# ==================================================
# CONFIGURAZIONE DEI MODELLI AI
# ==================================================

# Modello utilizzato per generare gli embeddings
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Modello utilizzato per valutare la pertinenza dei documenti
RERANKER_MODEL_NAME = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

# Modello generativo
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ==================================================
# CONFIGURAZIONE DEL CHUNKING
# ==================================================

CHUNK_SIZE = 18
CHUNK_OVERLAP = 4


# ==================================================
# CONFIGURAZIONE DEL RETRIEVAL
# ==================================================

RETRIEVAL_TOP_K = 3

# Similarità minima richiesta per accettare la domanda
SIMILARITY_THRESHOLD = 0.5

# Soglia sperimentale per selezionare i chunk tramite reranker
RERANKER_THRESHOLD = 0.0


# ==================================================
# CONFIGURAZIONE DELLA GENERAZIONE
# ==================================================

MAX_NEW_TOKENS = 100