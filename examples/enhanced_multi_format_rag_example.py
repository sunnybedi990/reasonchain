#!/usr/bin/env python3
"""
Enhanced Multi-Format RAG Example for ReasonChain

This example demonstrates:
1. Processing multiple file formats (PDF, DOCX, Excel, Markdown, JSON, etc.)
2. Using custom and fine-tuned embedding models
3. Creating a comprehensive RAG pipeline with mixed content types

Author: ReasonChain Team
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasonchain.rag.vector.add_to_vector_db import add_data_to_vector_db
from reasonchain.rag.embeddings.embedding_config import (
    register_huggingface_model, 
    register_local_model, 
    register_fine_tuned_model,
    list_available_models,
    get_embedding_config
)
from reasonchain.rag.rag_main import RAGPipeline

# Configuration constants
DB_PATH = "enhanced_multi_format_db.index"
DB_TYPE = "faiss"
BATCH_SIZE = 32
USE_GPU = True

def setup_custom_models():
    """Register custom embedding models for demonstration."""
    print("Setting up custom embedding models...")
    
    # Register a multilingual model from HuggingFace
    register_huggingface_model(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        384,
        "Multilingual paraphrase model for cross-language understanding"
    )
    
    # Register another HuggingFace model
    register_huggingface_model(
        "sentence-transformers/all-distilroberta-v1",
        768,
        "DistilRoBERTa model for semantic similarity"
    )
    
    # Example: Register a hypothetical fine-tuned model
    # (In practice, you would have an actual fine-tuned model)
    register_fine_tuned_model(
        "domain-specific-embeddings",
        "sentence-transformers/all-mpnet-base-v2",
        768,
        "Custom fine-tuned model for domain-specific tasks",
        model_path="your-username/your-fine-tuned-model"  # HuggingFace repo or local path
    )
    
    print("Custom models registered successfully!")

def prepare_sample_files():
    """Create sample files of different formats for testing."""
    print("Preparing sample files...")
    
    # Create directories if they don't exist
    os.makedirs("sample_data", exist_ok=True)
    
    # Sample Markdown file
    with open("sample_data/README.md", "w", encoding="utf-8") as f:
        f.write("""# Project Documentation

## Overview
This is a comprehensive project that handles multiple data formats.

## Features
- Multi-format document processing
- Advanced embedding models
- Scalable vector database integration

## Technical Stack
- Python 3.8+
- ReasonChain framework
- FAISS vector database
""")
    
    # Sample JSON configuration
    with open("sample_data/config.json", "w", encoding="utf-8") as f:
        f.write("""{
    "database": {
        "type": "faiss",
        "dimension": 768,
        "metric": "cosine"
    },
    "embedding": {
        "provider": "sentence_transformers",
        "model": "all-mpnet-base-v2"
    },
    "processing": {
        "batch_size": 32,
        "use_gpu": true
    }
}""")
    
    # Sample text file
    with open("sample_data/notes.txt", "w", encoding="utf-8") as f:
        f.write("""Important Notes:

1. Always validate input data before processing
2. Use appropriate embedding models for your domain
3. Consider batch processing for large datasets
4. Monitor GPU memory usage during processing
5. Implement proper error handling and logging

Best Practices:
- Test with small datasets first
- Use version control for model configurations
- Document your embedding model choices
- Regular backup of vector databases
""")
    
    print("Sample files created successfully!")

def process_multi_format_documents():
    """Process documents of various formats into the vector database."""
    print("\n" + "="*60)
    print("PROCESSING MULTI-FORMAT DOCUMENTS")
    print("="*60)
    
    # List of files to process (mix of different formats)
    file_paths = [
        "sample_data/README.md",
        "sample_data/config.json", 
        "sample_data/notes.txt"
    ]
    
    # Add any existing files from the project
    project_files = []
    if os.path.exists("reasonchain/rag/Database_Readme.md"):
        project_files.append("reasonchain/rag/Database_Readme.md")
    if os.path.exists("CONTRIBUTING.md"):
        project_files.append("CONTRIBUTING.md")
    
    all_files = file_paths + project_files
    
    # Filter existing files
    existing_files = [f for f in all_files if os.path.exists(f)]
    
    if not existing_files:
        print("No files found to process. Please ensure sample files exist.")
        return None
    
    print(f"Processing {len(existing_files)} files:")
    for f in existing_files:
        print(f"  - {f}")
    
    try:
        # Process with default model
        result = add_data_to_vector_db(
            file_paths=existing_files,
            db_path=DB_PATH,
            db_type=DB_TYPE,
            embedding_provider="sentence_transformers",
            embedding_model="all-mpnet-base-v2",
            use_gpu=USE_GPU,
            batch_size=BATCH_SIZE
        )
        
        print(f"\nSuccessfully processed files into vector database: {DB_PATH}")
        return result
        
    except Exception as e:
        print(f"Error processing files: {e}")
        return None

def demonstrate_custom_model_usage():
    """Demonstrate using custom embedding models."""
    print("\n" + "="*60)
    print("DEMONSTRATING CUSTOM MODEL USAGE")
    print("="*60)
    
    # List available models
    print("Available embedding models:")
    models = list_available_models()
    for provider, provider_models in models.items():
        if provider_models:
            print(f"\n{provider.upper()}:")
            for model_name, config in provider_models.items():
                print(f"  - {model_name}: {config['dimension']}D")
                print(f"    Description: {config['description']}")
    
    # Try to use a custom model if files exist
    sample_files = ["sample_data/README.md"]
    existing_files = [f for f in sample_files if os.path.exists(f)]
    
    if existing_files:
        try:
            print(f"\nProcessing with custom multilingual model...")
            custom_db_path = "custom_model_db.index"
            
            result = add_data_to_vector_db(
                file_paths=existing_files,
                db_path=custom_db_path,
                db_type=DB_TYPE,
                embedding_provider="hugging_face",
                embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                use_gpu=USE_GPU,
                batch_size=BATCH_SIZE
            )
            
            print(f"Successfully created vector database with custom model: {custom_db_path}")
            
        except Exception as e:
            print(f"Custom model processing failed: {e}")
            print("This is expected if the custom model is not actually available.")

def demonstrate_rag_query():
    """Demonstrate querying the RAG system with processed documents."""
    print("\n" + "="*60)
    print("DEMONSTRATING RAG QUERIES")
    print("="*60)
    
    if not os.path.exists(DB_PATH):
        print(f"Vector database {DB_PATH} not found. Please run document processing first.")
        return
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline(
            vector_db_path=DB_PATH,
            embedding_provider="sentence_transformers",
            embedding_model="all-mpnet-base-v2",
            use_gpu=USE_GPU
        )
        
        # Sample queries
        queries = [
            "What are the key features of this project?",
            "How should I configure the database?",
            "What are the best practices mentioned?",
            "What is the technical stack used?"
        ]
        
        print("Running sample queries:")
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            try:
                response = rag.query(query, top_k=3)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Query failed: {e}")
                
    except Exception as e:
        print(f"RAG pipeline initialization failed: {e}")

def main():
    """Main function demonstrating enhanced ReasonChain capabilities."""
    print("Enhanced Multi-Format RAG Example")
    print("=" * 60)
    
    try:
        # Step 1: Setup custom models
        setup_custom_models()
        
        # Step 2: Prepare sample files
        prepare_sample_files()
        
        # Step 3: Process multi-format documents
        process_result = process_multi_format_documents()
        
        # Step 4: Demonstrate custom model usage
        demonstrate_custom_model_usage()
        
        # Step 5: Demonstrate RAG queries
        if process_result is not None:
            demonstrate_rag_query()
        
        print("\n" + "="*60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nKey improvements demonstrated:")
        print("✓ Multi-format document processing (Markdown, JSON, TXT)")
        print("✓ Custom embedding model registration")
        print("✓ Flexible configuration system")
        print("✓ Enhanced error handling and validation")
        print("✓ Comprehensive documentation and examples")
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        raise
    
    finally:
        # Cleanup sample files if desired
        cleanup = input("\nCleanup sample files? (y/n): ").lower().strip()
        if cleanup == 'y':
            import shutil
            if os.path.exists("sample_data"):
                shutil.rmtree("sample_data")
            for db_file in [DB_PATH, "custom_model_db.index"]:
                if os.path.exists(db_file):
                    os.remove(db_file)
            print("Cleanup completed!")

if __name__ == "__main__":
    main() 