# **ReasonChain**

**ReasonChain** is a modular and extensible AI reasoning library for creating intelligent agents capable of executing advanced reasoning processes. It supports **Chain of Thought (CoT)** reasoning, **Tree of Thought (ToT)** reasoning, **Parallel Chains**, and integrates with **Retrieval-Augmented Generation (RAG)** for enhanced knowledge management.

---

## **Features**

1. **Customizable Reasoning Pipelines**:
   - Seamless integration of CoT, ToT, Parallel Chains, and Hybrid Pipelines.
   - Facilitates task decomposition, execution, and validation.

2. **RAG Integration**:
   - Retrieve and augment responses using long-term memory stored in vector databases like FAISS.

3. **Short-term and Long-term Memory**:
   - Session-based short-term memory for reasoning chains.
   - Persistent long-term memory powered by FAISS.

4. **LLM Compatibility**:
   - Supports OpenAI GPT, Llama, and other models for robust reasoning and summarization.

5. **Utility Tools**:
   - Adaptive complexity evaluation for reasoning tasks.
   - Centralized model management for handling LLM interactions.

---

## **Installation**

### 1. Install via pip
You can install **ReasonChain** directly from PyPI:
```bash
pip install reasonchain
```

### 2. Clone the Repository (for development)
Alternatively, clone the repository to access examples and editable source code:
```bash
git clone https://github.com/sunnybedi990/reasonchain.git
cd reasonchain
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install as an Editable Package
```bash
pip install -e .
```

### 5. Configure API Keys
Create a `.env` file with your API keys:
```plaintext
OPENAI_API_KEY=your_openai_api_key
```

---

## **Usage**

### 1. Initialize an Agent
```python
from reasonchain import Agent

agent = Agent(name="ResearchBot", model_name="gpt-4", api_key="your_openai_api_key")
```

### 2. Build a CoT Reasoning Pipeline
```python
from reasonchain.cot_pipeline import CoTPipeline

pipeline = CoTPipeline(agent)
pipeline.add_step("Understand the user's query.")
pipeline.add_step("Retrieve data from the knowledge base.")
pipeline.add_step("Generate a detailed response.")
```

### 3. Execute the Pipeline
```python
response = pipeline.execute(agent.model_manager)
print(response)
```

### 4. Use RAG to Enhance Reasoning
Add context to reasoning using RAG integration:
```python
from reasonchain.rag.vector.add_to_vector_db import add_pdf_to_vector_db
from reasonchain.rag.rag_main import query_vector_db

# Add a document to the vector database
add_pdf_to_vector_db(
    pdf_path="path/to/document.pdf",
    db_path="vector_db.index",
    db_type="faiss",
    embedding_model="all-mpnet-base-v2"
)

# Query the database
response = query_vector_db(
    db_path="vector_db.index",
    query="Summarize key financial data."
)
print(response)
```

---

## **Examples**

Explore the `examples/` directory for ready-to-run scripts:
- **`simple_agent.py`**: Build a basic reasoning agent.
- **`rag_pipeline_example.py`**: Use RAG for context-enhanced reasoning.
- **`tree_of_thought_example.py`**: Implement Tree of Thought reasoning.
- **`hybrid_reasoning_example.py`**: Dynamically combine reasoning methods.

---

## **Project Structure**

```plaintext
ReasonChain/
│
├── reasonchain/                  # Core library
│   ├── __init__.py               # Library initializer
│   ├── agent.py                  # Agent class
│   ├── cot_pipeline.py           # CoT pipeline
│   ├── rag/                      # RAG integration
│       ├── vector/               # Vector database utilities
│       ├── embeddings/           # Embedding management
│       ├── llm_response/         # LLM response handlers
│   ├── model_manager.py          # LLM management
│   ├── memory.py                 # Memory handling
│
├── examples/                     # Example scripts
├── tests/                        # Unit tests
├── setup.py                      # Installation script
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
├── LICENSE                       # Licensing information
```

---

## **Key Features**

1. **Dynamic Pipelines**:
   - Choose from CoT, ToT, Parallel Chains, or Hybrid Pipelines.

2. **Knowledge Integration**:
   - Augment reasoning with RAG for external data retrieval.

3. **Scalable Memory**:
   - Manage short-term and long-term memory effectively.

4. **LLM Compatibility**:
   - Seamlessly work with OpenAI GPT, Llama, and similar models.

---

## **Future Enhancements**

1. **Domain-Specific Templates**:
   - Add pre-trained reasoning templates for specialized applications.

2. **Agent Collaboration**:
   - Enable seamless teamwork between agents.

3. **Extended RAG Support**:
   - Integrate with Pinecone, Milvus, and more vector databases.

4. **Fine-Tuning Support**:
   - Incorporate custom fine-tuned models for advanced reasoning.

---

## **Contributing**

Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.