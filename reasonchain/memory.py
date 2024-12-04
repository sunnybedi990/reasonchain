from .faiss_vector_db import FAISSVectorDB
from sentence_transformers import SentenceTransformer

class Memory:
    def __init__(self, embedding_model="all-mpnet-base-v2", use_gpu=False, dimension=768):
        """
        Initialize short-term memory and long-term memory using FAISS.
        :param embedding_model: SentenceTransformer model for embeddings.
        """
        self.short_term = []  # Short-term memory
        self.embedding_model = SentenceTransformer(embedding_model)  # Embedding model

        # Initialize FAISS index for long-term memory
        self.long_term = FAISSVectorDB(use_gpu=use_gpu, dimension=dimension)


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

    def store_long_term(self, text):
        """
        Store data in long-term memory using FAISS.
        :param text: Text to store.
        """
        embedding = self.embedding_model.encode(text, convert_to_tensor=True).detach().cpu().numpy()
        self.long_term.add_embeddings([embedding], [text])
        print("[Memory] Stored in long-term memory:", text)

    def retrieve_long_term(self, query, top_k=3):
        """
        Retrieve relevant data from long-term memory using semantic search.
        :param query: Query text.
        :param top_k: Number of results to retrieve.
        :return: List of top-k similar texts.
        """
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).detach().cpu().numpy()
        print(query_embedding.shape)
        results = self.long_term.search(query_embedding, top_k=top_k)
        print(f"[Memory] Retrieved {len(results)} results from long-term memory.")
        return results
