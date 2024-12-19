from reasonchain.rag.vector.VectorDB import VectorDB
from sentence_transformers import SentenceTransformer
from collections import deque


class Memory:
    def __init__(
        self,
        embedding_provider='sentence_transformers',
        embedding_model="all-mpnet-base-v2",
        use_gpu=False,
        dimension=768,
        rag_db_path="vector_db.index",
        rag_db_type="faiss",
        chat_history_limit=100,
        db_config=None
    ):
        """
        Initialize memory with RAG-based storage and chat history.
        :param embedding_model: SentenceTransformer model for embeddings.
        :param use_gpu: Whether to use GPU for embeddings.
        :param dimension: Embedding vector dimension.
        :param rag_db_path: Path to the vector database for RAG storage.
        :param rag_db_type: Type of vector database for RAG (e.g., 'faiss', 'milvus').
        :param chat_history_limit: Max number of chat entries to retain in memory.
        """
        # Short-term memory
        self.short_term = []

        # RAG-based long-term memory
        self.embedding_model = SentenceTransformer(embedding_model)
        self.rag_vector_db = VectorDB(use_gpu=use_gpu, dimension=dimension, db_path=rag_db_path, db_type=rag_db_type, embedding_provider=embedding_provider,embedding_model=embedding_model, db_config=db_config)

        # Chat/Query History
        self.chat_history = deque(maxlen=chat_history_limit)  # Circular buffer for chat history

    def store_short_term(self, data):
        """Store data in short-term memory."""
        self.short_term.append(data)
        print("[Memory] Stored in short-term:", data)

    def retrieve_short_term(self):
        """Retrieve all short-term memory."""
        return self.short_term

    def clear_short_term(self):
        """Clear short-term memory."""
        self.short_term = []
        print("[Memory] Cleared short-term memory.")

      # Long-term Memory (RAG)
    def store_long_term_rag(self, text, id=None):
        """Store data in RAG-based long-term memory."""
        self.rag_vector_db.add_document(text, id=id)
        print(f"[Memory] Stored in RAG-based memory: {text[:50]}...")

    def retrieve_long_term_rag(self, query, top_k=3):
        """Retrieve relevant data from RAG-based long-term memory."""
        results = self.rag_vector_db.search(query, top_k=top_k)
        return [result[0] if isinstance(result, tuple) else result for result in results]

    # Chat History
    def store_chat_history(self, user_input, assistant_response):
        """Store user and assistant interaction in chat history."""
        self.chat_history.append({"user": user_input, "assistant": assistant_response})
        print(f"[Memory] Chat history updated: {user_input[:50]} -> {assistant_response[:50]}")

    def retrieve_chat_history(self, num_entries=None):
        """
        Retrieve recent chat history.
        :param num_entries: Number of recent entries to retrieve (None for all).
        :return: List of chat entries.
        """
        if num_entries:
            return list(self.chat_history)[-num_entries:]
        return list(self.chat_history)

    def clear_chat_history(self):
        """Clear all chat history."""
        self.chat_history.clear()
        print("[Memory] Cleared chat history.")


# Share Memory Between Agents

class SharedMemory:
    def __init__(self):
        self.memory_pool = {}

    def add_entry(self, key, value):
        self.memory_pool[key] = value

    def retrieve_entry(self, key):
        return self.memory_pool.get(key, None)

    def list_keys(self):
        return list(self.memory_pool.keys())
