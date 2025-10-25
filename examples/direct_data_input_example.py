#!/usr/bin/env python3
"""
Direct Data Input Example for ReasonChain

This example demonstrates how to add data directly to vector databases without processing files:
1. Raw text data (with automatic embedding generation)
2. Pre-computed embeddings with associated texts
3. Structured data from APIs, databases, or other sources
4. External source integration (databases, APIs, etc.)

Author: ReasonChain Team
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasonchain.rag.vector.add_to_vector_db import (
    add_raw_data_to_vector_db,
    add_structured_data_to_vector_db,
    add_external_source_to_vector_db
)
from reasonchain.rag.rag_main import RAGPipeline

# Configuration constants
DB_PATH_RAW = "raw_data_db.index"
DB_PATH_STRUCTURED = "structured_data_db.index"
DB_PATH_EXTERNAL = "external_data_db.index"
DB_PATH_EMBEDDINGS = "precomputed_embeddings_db.index"
DB_TYPE = "faiss"
USE_GPU = True

def example_1_raw_text_data():
    """Example 1: Add raw text data directly (most common use case)."""
    print("\n" + "="*60)
    print("EXAMPLE 1: RAW TEXT DATA INPUT")
    print("="*60)
    
    # Sample text data that might come from various sources
    raw_texts = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
        "Computer vision is a field of AI that trains computers to interpret and understand the visual world using digital images and videos.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment."
    ]
    
    print(f"Adding {len(raw_texts)} raw text entries to vector database...")
    
    try:
        result = add_raw_data_to_vector_db(
            texts=raw_texts,
            db_path=DB_PATH_RAW,
            db_type=DB_TYPE,
            use_gpu=USE_GPU
        )
        
        print("✅ Successfully added raw text data!")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_2_precomputed_embeddings():
    """Example 2: Add pre-computed embeddings with associated texts."""
    print("\n" + "="*60)
    print("EXAMPLE 2: PRE-COMPUTED EMBEDDINGS INPUT")
    print("="*60)
    
    # Simulate pre-computed embeddings (in real scenario, these might come from another model)
    texts = [
        "Python is a high-level programming language",
        "JavaScript is widely used for web development",
        "SQL is used for database management"
    ]
    
    # Generate random embeddings for demonstration (768 dimensions for BERT-like models)
    embeddings = np.random.rand(len(texts), 768).astype('float32')
    
    print(f"Adding {len(texts)} texts with pre-computed embeddings...")
    print(f"Embeddings shape: {embeddings.shape}")
    
    try:
        result = add_raw_data_to_vector_db(
            texts=texts,
            embeddings=embeddings,
            db_path=DB_PATH_EMBEDDINGS,
            db_type=DB_TYPE,
            use_gpu=USE_GPU,
            create_embeddings=False  # Don't generate new embeddings
        )
        
        print("✅ Successfully added pre-computed embeddings!")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_3_structured_data():
    """Example 3: Add structured data (like API responses or database results)."""
    print("\n" + "="*60)
    print("EXAMPLE 3: STRUCTURED DATA INPUT")
    print("="*60)
    
    # Simulate structured data that might come from an API or database
    structured_data = [
        {
            "text": "Breaking: New AI model achieves state-of-the-art performance on language understanding tasks",
            "title": "AI Breakthrough in NLP",
            "category": "technology",
            "author": "Dr. Jane Smith",
            "date": "2025-01-15",
            "source": "TechNews API",
            "tags": ["AI", "NLP", "research"]
        },
        {
            "text": "Scientists discover new method for quantum computing that could revolutionize cryptography",
            "title": "Quantum Computing Advance",
            "category": "science",
            "author": "Prof. John Doe",
            "date": "2025-01-14",
            "source": "Science Journal API",
            "tags": ["quantum", "computing", "cryptography"]
        },
        {
            "text": "New framework for sustainable energy management shows 40% improvement in efficiency",
            "title": "Sustainable Energy Framework",
            "category": "environment",
            "author": "Dr. Maria Garcia",
            "date": "2025-01-13",
            "source": "Green Tech Database",
            "tags": ["energy", "sustainability", "efficiency"]
        }
    ]
    
    print(f"Adding {len(structured_data)} structured data entries...")
    
    try:
        result = add_structured_data_to_vector_db(
            data_dict=structured_data,
            text_field='text',  # Field containing the main text content
            metadata_fields=['title', 'category', 'author', 'date', 'source', 'tags'],  # Metadata to preserve
            db_path=DB_PATH_STRUCTURED,
            db_type=DB_TYPE,
            use_gpu=USE_GPU
        )
        
        print("✅ Successfully added structured data!")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_4_external_api_simulation():
    """Example 4: Simulate adding data from external API."""
    print("\n" + "="*60)
    print("EXAMPLE 4: EXTERNAL API DATA INPUT (SIMULATED)")
    print("="*60)
    
    # Simulate API response data
    api_response_data = [
        {
            "content": "Cloud computing provides on-demand access to computing resources over the internet",
            "metadata": {
                "title": "Introduction to Cloud Computing",
                "category": "technology",
                "published_date": "2025-01-10"
            }
        },
        {
            "content": "Microservices architecture breaks down applications into smaller, independent services",
            "metadata": {
                "title": "Microservices Architecture Guide",
                "category": "software-engineering", 
                "published_date": "2025-01-09"
            }
        }
    ]
    
    # Transform API data to our structured format
    transformed_data = []
    for item in api_response_data:
        transformed_item = {
            "text": item["content"],
            **item["metadata"]  # Flatten metadata
        }
        transformed_data.append(transformed_item)
    
    print(f"Processing {len(transformed_data)} items from simulated API...")
    
    try:
        result = add_structured_data_to_vector_db(
            data_dict=transformed_data,
            text_field='text',
            metadata_fields=['title', 'category', 'published_date'],
            db_path=DB_PATH_EXTERNAL,
            db_type=DB_TYPE,
            use_gpu=USE_GPU
        )
        
        print("✅ Successfully added external API data!")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_5_batch_processing():
    """Example 5: Batch processing of large datasets."""
    print("\n" + "="*60)
    print("EXAMPLE 5: BATCH PROCESSING LARGE DATASETS")
    print("="*60)
    
    # Simulate a large dataset (e.g., from web scraping, database export, etc.)
    large_dataset = []
    
    # Generate sample data
    categories = ["technology", "science", "business", "health", "education"]
    for i in range(50):  # Simulate 50 documents
        category = categories[i % len(categories)]
        large_dataset.append({
            "text": f"This is document {i+1} about {category}. It contains important information relevant to the {category} domain and provides valuable insights for research and development.",
            "doc_id": f"doc_{i+1:03d}",
            "category": category,
            "batch": f"batch_{(i//10)+1}",
            "processed_date": "2025-01-15"
        })
    
    print(f"Processing {len(large_dataset)} documents in batches...")
    
    # Process in smaller batches
    batch_size = 10
    results = []
    
    try:
        for i in range(0, len(large_dataset), batch_size):
            batch = large_dataset[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"Processing batch {batch_num} ({len(batch)} documents)...")
            
            result = add_structured_data_to_vector_db(
                data_dict=batch,
                text_field='text',
                metadata_fields=['doc_id', 'category', 'batch', 'processed_date'],
                db_path=f"batch_{batch_num}_db.index",
                db_type=DB_TYPE,
                use_gpu=USE_GPU
            )
            
            results.append(result)
        
        print("✅ Successfully processed all batches!")
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_6_mixed_data_types():
    """Example 6: Handle mixed data types and edge cases."""
    print("\n" + "="*60)
    print("EXAMPLE 6: MIXED DATA TYPES AND EDGE CASES")
    print("="*60)
    
    # Mixed data with various edge cases
    mixed_data = [
        {
            "text": "Short text",
            "type": "minimal"
        },
        {
            "text": "This is a much longer piece of text that contains multiple sentences and provides more detailed information about a complex topic. It demonstrates how the system handles variable-length content and maintains quality across different text sizes.",
            "type": "detailed",
            "word_count": 42
        },
        {
            "text": "Text with special characters: @#$%^&*()_+{}|:<>?[]\\;'\",./ and unicode: 你好 世界 🌍 🚀",
            "type": "special_chars",
            "encoding": "utf-8"
        },
        {
            "text": "   Text with extra whitespace   ",
            "type": "whitespace_issues"
        }
    ]
    
    print(f"Processing {len(mixed_data)} mixed data types...")
    
    try:
        result = add_structured_data_to_vector_db(
            data_dict=mixed_data,
            text_field='text',
            metadata_fields=['type', 'word_count', 'encoding'],
            db_path="mixed_data_db.index",
            db_type=DB_TYPE,
            use_gpu=USE_GPU
        )
        
        print("✅ Successfully processed mixed data types!")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_query_functionality():
    """Test querying the created vector databases."""
    print("\n" + "="*60)
    print("TESTING QUERY FUNCTIONALITY")
    print("="*60)
    
    # Test querying the raw text database
    if os.path.exists(DB_PATH_RAW):
        try:
            print("Testing queries on raw text database...")
            rag = RAGPipeline(
                vector_db_path=DB_PATH_RAW,
                embedding_provider="sentence_transformers",
                embedding_model="all-mpnet-base-v2",
                use_gpu=USE_GPU
            )
            
            queries = [
                "What is machine learning?",
                "Tell me about neural networks",
                "How does computer vision work?"
            ]
            
            for query in queries:
                print(f"\nQuery: {query}")
                try:
                    response = rag.query(query, top_k=2)
                    print(f"Response: {response}")
                except Exception as e:
                    print(f"Query failed: {e}")
                    
        except Exception as e:
            print(f"RAG initialization failed: {e}")
    
    # Test querying structured data
    if os.path.exists(DB_PATH_STRUCTURED):
        try:
            print("\nTesting queries on structured data...")
            rag_structured = RAGPipeline(
                vector_db_path=DB_PATH_STRUCTURED,
                embedding_provider="sentence_transformers",
                embedding_model="all-mpnet-base-v2",
                use_gpu=USE_GPU
            )
            
            structured_queries = [
                "What are the latest AI developments?",
                "Tell me about quantum computing",
                "What's new in sustainable energy?"
            ]
            
            for query in structured_queries:
                print(f"\nQuery: {query}")
                try:
                    response = rag_structured.query(query, top_k=2)
                    print(f"Response: {response}")
                except Exception as e:
                    print(f"Query failed: {e}")
                    
        except Exception as e:
            print(f"Structured RAG initialization failed: {e}")

def cleanup_example_files():
    """Clean up created database files."""
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)
    
    db_files = [
        DB_PATH_RAW,
        DB_PATH_STRUCTURED, 
        DB_PATH_EXTERNAL,
        DB_PATH_EMBEDDINGS,
        "mixed_data_db.index"
    ]
    
    # Add batch files
    for i in range(1, 6):
        db_files.append(f"batch_{i}_db.index")
    
    cleanup = input("Clean up created database files? (y/n): ").lower().strip()
    if cleanup == 'y':
        for db_file in db_files:
            if os.path.exists(db_file):
                os.remove(db_file)
                print(f"Removed: {db_file}")
        print("Cleanup completed!")
    else:
        print("Database files preserved for further testing.")

def main():
    """Main function demonstrating direct data input capabilities."""
    print("Direct Data Input Example for ReasonChain")
    print("=" * 60)
    print("This example shows how to add data directly without processing files:")
    print("• Raw text data with automatic embedding generation")
    print("• Pre-computed embeddings")
    print("• Structured data from APIs/databases")
    print("• Batch processing for large datasets")
    print("• Mixed data types and edge cases")
    
    try:
        # Run all examples
        results = []
        
        print("\n🚀 Starting direct data input examples...")
        
        # Example 1: Raw text data
        result1 = example_1_raw_text_data()
        results.append(("Raw Text Data", result1))
        
        # Example 2: Pre-computed embeddings
        result2 = example_2_precomputed_embeddings()
        results.append(("Pre-computed Embeddings", result2))
        
        # Example 3: Structured data
        result3 = example_3_structured_data()
        results.append(("Structured Data", result3))
        
        # Example 4: External API simulation
        result4 = example_4_external_api_simulation()
        results.append(("External API Data", result4))
        
        # Example 5: Batch processing
        result5 = example_5_batch_processing()
        results.append(("Batch Processing", result5))
        
        # Example 6: Mixed data types
        result6 = example_6_mixed_data_types()
        results.append(("Mixed Data Types", result6))
        
        # Test querying
        test_query_functionality()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        
        for example_name, result in results:
            status = "✅ SUCCESS" if result is not None else "❌ FAILED"
            print(f"{example_name}: {status}")
        
        print("\n🎉 All direct data input examples completed!")
        
        print("\nKey capabilities demonstrated:")
        print("✓ Raw text input with automatic embeddings")
        print("✓ Pre-computed embedding integration")
        print("✓ Structured data processing")
        print("✓ External source simulation")
        print("✓ Batch processing for scalability")
        print("✓ Mixed data type handling")
        print("✓ Query functionality testing")
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        raise
    
    finally:
        cleanup_example_files()

if __name__ == "__main__":
    main() 