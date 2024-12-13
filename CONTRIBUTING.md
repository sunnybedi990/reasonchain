# Project Structure

This document provides a detailed breakdown of the ReasonChain library's structure.

```plaintext
ReasonChain/
│
├── reasonchain/                          # Core library
│   ├── __init__.py                       # Library initializer
│   ├── agent.py                          # Agent class
│   ├── cot_pipeline.py                   # CoT pipeline
│   ├── rag/                              # RAG integration
│       ├── __init__.py                   # RAG module initializer
│       ├── adapters/                     # RAG adapters for vector databases
│       ├── charts/                       # Chart-related utilities
│       ├── document_extract/             # File extractors
│       ├── embeddings/                   # Embedding management
│           ├── __init__.py               # Embeddings module initializer
│           ├── embedding_config.py       # Embedding configuration
│           ├── embedding_initializer.py  # Embedding initialization logic
│       ├── llm_response/                 # LLM response handling
│           ├── __init__.py               # LLM response module initializer
│           ├── chart_parser.py           # Chart data parsing
│           ├── prompt.py                 # Prompt-related utilities
│       ├── vector/                       # Vector database utilities
│           ├── __init__.py               # Vector module initializer
│           ├── add_to_vector_db.py       # Add data to vector databases
│           ├── config.yaml               # Vector database configuration
│           ├── utils.py                  # Utility functions for vector databases
│           ├── VectorDB.py               # Core vector database interface
│       ├── config.py                     # RAG configuration
│       ├── Database_Readme.md            # Database usage documentation
│       ├── rag_main.py                   # Main RAG pipeline logic
│   ├── domain_templates/                 # Industry-specific domain templates
│       ├── __init__.py                   # Domain templates module initializer
│       ├── customer_support.py           # Customer support templates
│       ├── ecommerce.py                  # E-commerce templates
│       ├── education.py                  # Education templates
│       ├── energy.py                     # Energy sector templates
│       ├── finance.py                    # Finance and accounting templates
│       ├── gaming_entertainment.py       # Gaming and entertainment templates
│       ├── healthcare.py                 # Healthcare templates
│       ├── hr_recruitment.py             # HR and recruitment templates
│       ├── legal.py                      # Legal industry templates
│       ├── marketing.py                  # Marketing and advertising templates
│       ├── real_estate.py                # Real estate templates
│       ├── retail.py                     # Retail industry templates
│       ├── supply_chain.py               # Supply chain templates
│       ├── transportation.py             # Transportation templates
│       ├── travel_hospitality.py         # Travel and hospitality templates
│   ├── model_manager.py                  # LLM management
│   ├── memory.py                         # Memory handling
│   ├── environment.py                    # Environment configuration
│   ├── faiss_vector_db.py                # FAISS vector database utilities
│   ├── utils.py                          # General-purpose utilities
│
├── examples/                             # Example scripts
│   ├── domain_templates_example/         # Domain-specific examples
│   ├── faq_bot.py                        # FAQ bot example
│   ├── fine_tuned_model_example.py       # Fine-tuned model usage
│   ├── hybrid_reasoning_example.py       # Hybrid reasoning example
│   ├── long_term_memory.py               # Long-term memory management
│   ├── multi-agent_collaboration.py      # Multi-agent collaboration
│   ├── multi-agent_with_rag_example.py   # Multi-agent system with RAG
│   ├── multi-agent_with_rag_markdown.py  # Multi-agent RAG with Markdown
│   ├── parellel_reasoning_example.py     # Parallel reasoning example
│   ├── rag_pipeline_example.py           # RAG pipeline demonstration
│   ├── simple_agent.py                   # Basic agent example
│   ├── tree_of_thought_example.py        # Tree of Thought reasoning
│
├── fine_tuned_model/                     # Fine-tuned model storage
├── images/                               # General image storage
├── markdown_images/                      # Markdown image storage
├── models/                               # Model-related files
├── parsed_chunks/                        # Chunked text storage
│   ├── hybrid_chunking_chunks.txt        # Hybrid chunking output
├── pdfs/                                 # Example PDF files
│   ├── example.pdf                       # Sample PDF
├── tests/                                # Unit tests
├── .env                                  # Environment variables
├── .gitignore                            # Git ignore file
├── .pypirc                               # PyPI configuration
├── LICENSE                               # Licensing information
├── MANIFEST.in                           # Manifest file for package distribution
├── README.md                             # Documentation
├── requirements.txt                      # Dependencies
├── setup.py                              # Installation script
