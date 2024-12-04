from .memory import Memory

class RAGIntegration:
    def __init__(self, memory: Memory):
        """
        Initialize RAG integration with the library's memory.
        :param memory: Memory instance for retrieval.
        """
        self.memory = memory

    def retrieve_knowledge(self, query: str, top_k: int = 3):
        """
        Retrieve relevant knowledge from long-term memory.
        :param query: The query string.
        :param top_k: Number of top results to fetch.
        :return: List of retrieved knowledge.
        """
        return self.memory.retrieve_long_term(query, top_k)

    def augment_with_context(self, query: str, external_context: str = None):
        """
        Augment the query with context from long-term memory or external sources.
        :param query: The query string.
        :param external_context: Additional context to include.
        :return: Augmented query string.
        """
        context = self.retrieve_knowledge(query)
        context_text = "\n".join([item[0] for item in context])  # Extract text
        augmented_query = f"{query}\n\nContext:\n{context_text}"
        if external_context:
            augmented_query += f"\n\nExternal Context:\n{external_context}"
        return augmented_query
