import argparse
from reasonchain.rag.vector.VectorDB import VectorDB
from reasonchain.rag.vector.add_to_vector_db import add_pdf_to_vector_db
from reasonchain.llm_models.model_manager import ModelManager
from reasonchain.rag.llm_response.chart_parser import parse_response_and_generate_chart
from reasonchain.rag.llm_response.prompt import Prompt

def query_vector_db(db_path, db_type, db_config, query, top_k=5, embedding_provider='', embedding_model='', use_gpu=False):
    """
    Performs a query on the vector database and generates a response using the specified LLM.
    """
    print(db_type)
    # Initialize the vector database
    vector_db = VectorDB(
        db_path=db_path,
        db_type=db_type,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        use_gpu=use_gpu,
        index_name=db_path,
        db_config=db_config
    )

    # Load the database index if supported
    if hasattr(vector_db.db, "load_index"):
        vector_db.load_index(db_path)

    print(f"Querying the vector database with: '{query}'")
    if query == "*":
        # Retrieve all documents
        results = vector_db.get_all()
        print('Result shape', results)
        # # Clean results for summarization
        if isinstance(results[0], tuple) and len(results[0]) == 2:
            # Remove scores if necessary
            clean_results = [text for text, score in results]
        else:
            clean_results = [text for text in results]
        accumulate_resuls = ''.join(clean_results)
        return accumulate_resuls

    # Perform the search using VectorDB
    results = vector_db.search(query, top_k=top_k)
    if not results:
        raise RuntimeError(f"No results found in {db_type} for the query: '{query}'")

    print(f"Raw search results: {results}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Add or query the vector database.")
    parser.add_argument("mode", choices=["add", "query"], help="Mode to run: 'add' or 'query'")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file for 'add' mode")
    parser.add_argument("--db_path", type=str, default="vector_db.index", help="Path to the vector DB file")
    parser.add_argument("--db_type", type=str, default="faiss", choices=["faiss", "milvus", "pinecone", "qdrant", "weaviate"], help="Type of vector database")
    parser.add_argument("--query", type=str, help="Search query for 'query' mode")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--embedding_provider", type=str, default="sentence-transformers", help="Embedding provider")
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2", help="Embedding model")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for Faiss indexing and querying")

    args = parser.parse_args()

    if args.mode == "add":
        if not args.pdf:
            print("Error: PDF path is required in 'add' mode.")
            return
        add_pdf_to_vector_db(
            pdf_path=args.pdf,
            db_path=args.db_path,
            db_type=args.db_type,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            use_gpu=args.use_gpu
        )
    elif args.mode == "query":
        if not args.query:
            print("Error: Query is required in 'query' mode.")
            return
        results = query_vector_db(
            db_path=args.db_path,
            db_type=args.db_type,
            db_config=None,
            query=args.query,
            top_k=args.top_k,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            use_gpu=args.use_gpu
        )
        print(f"Query Results: {results}")

if __name__ == "__main__":
    main()
