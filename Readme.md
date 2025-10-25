# **ReasonChain**

**ReasonChain** is a modular and extensible AI reasoning library for creating intelligent agents capable of executing advanced reasoning processes. It supports **Chain of Thought (CoT)** reasoning, **Tree of Thought (ToT)** reasoning, **Parallel Chains**, and integrates with **Retrieval-Augmented Generation (RAG)** for enhanced knowledge management.

---

### **Features**

1. **Customizable Reasoning Pipelines**:
   - Seamless integration of **Chain of Thought (CoT)**, **Tree of Thought (ToT)**, Parallel Chains, and Hybrid Pipelines.
   - Facilitates task decomposition, execution, and validation.

2. **RAG Integration**:
   - Retrieve and augment responses using long-term memory stored in vector databases like FAISS.
   - **Direct Data Input**: Add raw text, pre-computed embeddings, or structured data directly to vector databases.
   - **External Source Integration**: Connect to databases, APIs, cloud storage, and Elasticsearch for data ingestion.
   - **Multi-Format Processing**: Enhanced support for processing diverse file formats simultaneously.

3. **Short-term and Long-term Memory**:
   - Session-based short-term memory for reasoning chains.
   - Persistent long-term memory powered by FAISS.

4. **LLM Compatibility**:
   - Supports OpenAI GPT, Llama, and other models for robust reasoning and summarization.
   - **NEW: Extensible Provider Architecture** - Easily add support for ANY LLM service (Anthropic, Cohere, AI21, etc.)

5. **Custom Embedding Models**:
   - Register and use custom embedding models from HuggingFace, local files, or fine-tuned models.
   - Support for multilingual and domain-specific embedding models.
   - **NEW: Plugin Architecture** - Add custom embedding providers without modifying core code.

6. **Utility Tools**:
   - Adaptive complexity evaluation for reasoning tasks.
   - Centralized model management for handling LLM interactions.

7. **Supported File Types**:
   - Extract text, tables, and figures from a wide range of file formats, including:
     - **Documents**: PDF, Word, RTF, Markdown, HTML, LaTeX.
     - **Spreadsheets**: Excel, CSV.
     - **Multimedia**: Images, Videos, Audio.
     - **E-books**: EPUB, MOBI.
   - Provides modular extractors tailored for each file type, ensuring efficient data retrieval.
   - **Batch Processing**: Process multiple file formats simultaneously with optimized performance.

8. **Domain Templates**:
   - Pre-built reasoning templates tailored for specific industries and applications:
     - **Customer Support**: Automates handling of customer inquiries and escalations.
     - **Healthcare**: Assists in diagnosis recommendations and treatment plans.
     - **Finance**: Analyzes trends, generates financial reports, and forecasts.
     - **Retail**: Provides insights on inventory optimization and customer behavior.
     - **Real Estate**: Summarizes market data and assists in property evaluations.
   - Fully customizable to meet domain-specific requirements.
   - Simplifies the setup of agents for new use cases.

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
```env
# Authentication Keys
LLAMA_CLOUD_API_KEY=          # API key for LLama Parser
OPENAI_API_KEY=               # API key for OpenAI
GROQ_API_KEY=                 # API key for Groq

# Vector Database Configuration
VECTOR_DB_PATH=vector_dbs/    # Path to store vector databases
USE_GPU=true                  # Set to "true" for GPU usage or "false" for CPU

# Pinecone Configuration
PINECONE_API_KEY=             # API key for Pinecone

# Weaviate Configuration
WEAVIATE_API_KEY=             # API key for Weaviate
WEAVIATE_CLUSTER_URL=         # Cluster URL for Weaviate

# Qdrant Configuration
QDRANT_API_KEY=               # API key for Qdrant (optional)
QDRANT_ADMIN_API_KEY=         # Admin API key for Qdrant (optional)
QDRANT_CLUSTER_URL=           # Cluster URL for Qdrant (optional)
```

---

## **Usage**

### 1. Initialize an Agent
```python
from reasonchain import Agent

agent = Agent(name="ResearchBot", model_name="gpt-4", api="openai")
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

### 1. Build and Integrate Agents with RAG
ReasonChain makes it easy to create multi-agent systems and integrate them with RAG for context-enhanced reasoning.

```python
from reasonchain.agent import Agent, MultiAgentSystem
from reasonchain.rag.vector.add_to_vector_db import add_data_to_vector_db
from reasonchain.rag.rag_main import query_vector_db
from reasonchain.utils import (
    store_in_shared_memory,
    retrieve_from_shared_memory,
    collaborate_on_task,
    assign_and_execute_task
)

# Initialize the Multi-Agent System
multi_agent_system = MultiAgentSystem()

# Create and register agents
agent1 = Agent(name="AgentAlpha", role="extractor", model_name="gpt-4o", api="openai")
agent2 = Agent(name="AgentBeta", role="analyst", model_name="llama3.1:latest", api="ollama")
agent3 = Agent(name="AgentGamma", role="summarizer", model_name="llama-3.1-8b-instant", api="groq")

multi_agent_system.register_agent(agent1)
multi_agent_system.register_agent(agent2)
multi_agent_system.register_agent(agent3)

# Add documents to the vector database
add_data_to_vector_db(
    file_paths=["path/to/tesla-report.pdf", "path/to/data.xlsx", "path/to/notes.md"],
    db_path="vector_db.index",
    db_type="faiss",
    embedding_provider="sentence_transformers",
    embedding_model="all-mpnet-base-v2",
    use_gpu=True
)

# Query the vector database
query = "Extract financial highlights from the Tesla Q-10 report."
response = query_vector_db(
    db_path="vector_db.index",
    db_type="faiss",
    query=query,
    embedding_provider="sentence_transformers",
    embedding_model="all-mpnet-base-v2"
)

# Store and retrieve data in shared memory
store_in_shared_memory(agent1.shared_memory, "financial_highlights", response)
highlights = retrieve_from_shared_memory(agent2.shared_memory, "financial_highlights")

# Assign tasks and collaborate
task_description = "Analyze trends and summarize Tesla's financial highlights."
collaborators = ["AgentBeta", "AgentGamma"]
successful_agents = collaborate_on_task(multi_agent_system, collaborators, task_description)

print(f"Successful Agents: {successful_agents}")
```

---

## **Extensible Provider Architecture**

### **🔌 Plugin System for LLMs and Embeddings**

ReasonChain now features a powerful plugin architecture that allows you to easily add support for ANY LLM or embedding service without modifying core code!

### **Register Custom LLM Providers**

Adding support for a new LLM service is as simple as implementing a base class:

```python
from reasonchain.llm_models.base_provider import BaseLLMProvider
from reasonchain.llm_models.provider_registry import LLMProviderRegistry

# 1. Create your custom provider
class CohereProvider(BaseLLMProvider):
    def __init__(self, model_name, api_key=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        import cohere
        self.client = cohere.Client(api_key or os.getenv("COHERE_API_KEY"))
    
    def generate_response(self, prompt, **kwargs):
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', 2000)
        )
        return response.generations[0].text
    
    def generate_chat_response(self, messages, **kwargs):
        # Implement chat format
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.generate_response(prompt, **kwargs)

# 2. Register your provider
LLMProviderRegistry.register('cohere', CohereProvider)

# 3. Use it like any other provider!
agent = Agent(name="Cohere_Agent", model_name="command", api="cohere")
```

### **Built-in LLM Providers**

ReasonChain comes with these providers out of the box:
- **OpenAI** - GPT-4, GPT-3.5, etc.
- **Groq** - Fast inference with Llama, Mixtral
- **Ollama** - Local models (Llama, Mistral, etc.)
- **Anthropic** - Claude models
- **Custom** - Your fine-tuned models

### **Built-in Embedding Providers**

ReasonChain comes with these embedding providers:
- **sentence_transformers** - All sentence-transformers models (all-mpnet-base-v2, all-MiniLM-L6-v2, etc.)
- **openai** - OpenAI embeddings (text-embedding-ada-002, text-embedding-3-small/large)
- **hugging_face** - Any HuggingFace transformer model (BERT, RoBERTa, DistilBERT, custom models)

### **Register Custom Embedding Providers**

Same simple pattern for embedding providers:

```python
from reasonchain.llm_models.base_provider import BaseEmbeddingProvider
from reasonchain.llm_models.provider_registry import EmbeddingProviderRegistry

class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name, api_key=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        import voyageai
        self.client = voyageai.Client(api_key=api_key)
        self._dimension = 1024
    
    def embed_text(self, text):
        result = self.client.embed([text], model=self.model_name)
        return result.embeddings[0]
    
    def embed_batch(self, texts, batch_size=128):
        result = self.client.embed(texts, model=self.model_name)
        return result.embeddings
    
    def get_dimension(self):
        return self._dimension

# Register and use!
EmbeddingProviderRegistry.register('voyage', VoyageEmbeddingProvider)

# Use in RAG pipeline
add_data_to_vector_db(
    file_paths=["documents.pdf"],
    embedding_provider="voyage",
    embedding_model="voyage-01"
)
```

### **Why This Matters**

✅ **No Core Changes Needed** - Add new providers without touching ReasonChain internals  
✅ **Full Backward Compatibility** - Existing code works unchanged  
✅ **Support ANY Service** - Anthropic, Cohere, AI21, Voyage, local models, custom APIs  
✅ **Easy to Share** - Package your providers as plugins  
✅ **Future-Proof** - New LLM services? Just add a provider!

---

## **Custom Embedding Models**

ReasonChain supports registering and using custom embedding models for specialized use cases:

### **Register HuggingFace Models**
```python
from reasonchain.rag.embeddings.embedding_config import register_huggingface_model

# Register a multilingual model
register_huggingface_model(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    384,
    "Multilingual paraphrase model for cross-language understanding"
)

# Use the registered model
add_data_to_vector_db(
    file_paths=["document.pdf"],
    embedding_provider="huggingface",
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

### **Register Fine-Tuned Models**
```python
from reasonchain.rag.embeddings.embedding_config import register_fine_tuned_model

# Register your fine-tuned model
register_fine_tuned_model(
    "my-domain-model",
    "/path/to/fine-tuned-model",
    768,
    "Custom model fine-tuned for domain-specific tasks"
)
```

### **List Available Models**
```python
from reasonchain.rag.embeddings.embedding_config import list_available_models

# See all registered models
models = list_available_models()
print(models)
```

---

## **Direct Data Input & External Sources**

ReasonChain now supports adding data directly to vector databases without file processing, enabling seamless integration with various data sources.

### **Raw Data Input**
Add text data directly with automatic embedding generation:

```python
from reasonchain.rag.vector.add_to_vector_db import add_raw_data_to_vector_db

# Add raw text data
texts = [
    "Machine learning enables computers to learn from experience.",
    "Natural language processing helps computers understand human language.",
    "Computer vision trains computers to interpret visual data."
]

add_raw_data_to_vector_db(
    texts=texts,
    db_path="raw_data_db.index",
    embedding_provider="sentence_transformers",
    embedding_model="all-mpnet-base-v2"
)
```

### **Pre-computed Embeddings**
Use your own embeddings with associated text:

```python
import numpy as np

# Your pre-computed embeddings (e.g., from a fine-tuned model)
embeddings = np.random.rand(3, 768)  # 3 embeddings of 768 dimensions
texts = ["Text 1", "Text 2", "Text 3"]

add_raw_data_to_vector_db(
    texts=texts,
    embeddings=embeddings,
    db_path="custom_embeddings_db.index"
)
```

### **Structured Data Integration**
Process structured data from APIs, databases, or JSON:

```python
from reasonchain.rag.vector.add_to_vector_db import add_structured_data_to_vector_db

# Structured data (e.g., from an API response)
data = [
    {"text": "Article content 1", "title": "AI Trends", "category": "technology", "date": "2025-01-01"},
    {"text": "Article content 2", "title": "ML Applications", "category": "science", "date": "2025-01-02"}
]

add_structured_data_to_vector_db(
    data,
    text_field="text",
    metadata_fields=["title", "category", "date"],
    db_path="structured_db.index"
)
```

### **External Source Integration**
Connect directly to databases, APIs, and cloud services:

```python
from reasonchain.rag.vector.add_to_vector_db import add_external_source_to_vector_db

# Database integration
db_config = {
    'connection_string': 'postgresql://user:pass@localhost/db',
    'query': 'SELECT content, title, category FROM articles WHERE published = true',
    'text_field': 'content',
    'metadata_fields': ['title', 'category']
}

add_external_source_to_vector_db(
    source_type='database',
    source_config=db_config,
    db_path="database_content_db.index"
)

# API integration
api_config = {
    'url': 'https://api.example.com/articles',
    'headers': {'Authorization': 'Bearer your-token'},
    'text_field': 'content',
    'metadata_fields': ['title', 'author', 'date']
}

add_external_source_to_vector_db(
    source_type='api',
    source_config=api_config,
    db_path="api_content_db.index"
)
```

### **Supported External Sources**
- **Databases**: PostgreSQL, SQLite with custom SQL queries
- **REST APIs**: JSON responses with configurable field mapping
- **Elasticsearch**: Direct index querying with custom search queries
- **Cloud Storage**: (Coming soon) AWS S3, Google Cloud Storage, Azure Blob

---

## **Examples**

Explore more scripts in the `examples/` directory:
- **`rag_pipeline_example.py`**: Example of using RAG for context-enhanced reasoning.
- **`direct_data_input_example.py`**: Comprehensive examples of adding raw data, structured data, and external sources directly to vector databases.
- **`enhanced_multi_format_rag_example.py`**: Demonstrates processing multiple file formats with custom embedding models.
- **`custom_provider_example.py`**: Shows how to create and register custom LLM providers (Cohere, Anthropic, local models, etc.).
- **`custom_embedding_provider_example.py`**: Demonstrates creating custom embedding providers (Voyage AI, Cohere, local services).
- **`multi-agent_collaboration.py`**: Multi-agent collaboration example.
- **`multi-agent_with_rag_example.py`**: Multi-agent systems integrated with RAG pipelines.
- **`tree_of_thought_example.py`**: Demonstrates Tree of Thought reasoning.
- **`hybrid_reasoning_example.py`**: Combines multiple reasoning methods.
- **`fine_tuned_model_example.py`**: Shows how to use custom and fine-tuned models.
- **`domain_templates_example/`**: Industry-specific examples for various domains.

---

## **Supported File Types**

ReasonChain supports extracting text, tables, and figures from the following file types. It uses specialized extractors for each type to ensure accurate and comprehensive data retrieval.

### **Supported Extensions and Extractors**
| File Type               | Extensions             | Extractor Library/Tools                                   | Extracted Data           |
|-------------------------|------------------------|----------------------------------------------------------|--------------------------|
| **PDF Documents**       | `.pdf`                | PyMuPDF, Camelot, PDFPlumber, LlamaParse                 | Text, Tables, Figures    |
| **Word Documents**      | `.doc`, `.docx`       | python-docx                                              | Text, Tables, Figures    |
| **Excel Files**         | `.xls`, `.xlsx`       | pandas                                                   | Tables                   |
| **CSV Files**           | `.csv`                | pandas                                                   | Tables                   |
| **HTML Files**          | `.html`, `.htm`       | BeautifulSoup, pandas                                    | Text, Tables, Figures    |
| **Markdown Files**      | `.md`                 | re, pandas                                               | Text, Tables, Figures    |
| **Rich Text Files**     | `.rtf`                | pyrtf                                                    | Text                    |
| **E-Books**             | `.epub`, `.mobi`      | ebooklib, BeautifulSoup                                  | Text                    |
| **Images**              | `.png`, `.jpg`, `.jpeg`, `.tiff` | pytesseract (OCR)                                  | Text                    |
| **Presentation Files**  | `.ppt`, `.pptx`       | python-pptx                                              | Text, Figures            |
| **Audio Files**         | `.mp3`, `.wav`        | SpeechRecognition, Whisper                               | Text                    |
| **Video Files**         | `.mp4`, `.avi`, `.mov`| moviepy, pytesseract (OCR)                               | Text, Figures            |
| **LaTeX Files**         | `.tex`                | regex, plain text processing                             | Text                    |

---

### **Feature Highlights for Extractors**

1. **PDF Extraction**:
   - Handles structured text, tables, and embedded images.
   - LlamaParse integration for multimodal capabilities.

2. **Word and Presentation Files**:
   - Extracts text from paragraphs, tables, and embedded images.
   - Saves embedded figures locally for further processing.

3. **E-books and Markdown**:
   - Processes text and embedded images or hyperlinks.
   - Parses tables represented as plain text in Markdown.

4. **Images and Videos**:
   - Extracts frames from videos and applies OCR for textual content.
   - Processes scanned documents and infographics using pytesseract.

5. **Audio Extraction**:
   - Converts audio to text using SpeechRecognition or Whisper.

6. **Rich Text and LaTeX**:
   - Converts RTF files into plain text.
   - Removes LaTeX commands to provide clean text content.

---

### Domain Templates

ReasonChain provides **pre-built reasoning templates** tailored for specific industries and applications. These templates simplify the process of creating intelligent agents by embedding domain-specific logic and workflows. They can be used as-is or customized to meet unique requirements.

#### Available Domain Templates

| **Domain**              | **Description**                                                                 |
|--------------------------|---------------------------------------------------------------------------------|
| **Customer Support**     | Automates resolution workflows, ticket prioritization, and customer query handling. |
| **E-commerce**           | Assists in product recommendations, inventory optimization, and user behavior analysis. |
| **Education**            | Supports personalized learning plans, content recommendations, and assessments. |
| **Energy**               | Optimizes energy consumption, monitors grid performance, and supports sustainability planning. |
| **Finance**              | Analyzes financial trends, generates reports, and forecasts revenue and expenses. |
| **Gaming & Entertainment** | Provides real-time analytics for player engagement and content recommendations. |
| **Healthcare**           | Assists in diagnosis, treatment recommendations, and patient care management. |
| **HR & Recruitment**     | Streamlines candidate screening, skill assessments, and onboarding workflows. |
| **Legal**                | Summarizes legal documents, case analysis, and compliance checks. |
| **Marketing**            | Automates campaign analysis, audience targeting, and content generation. |
| **Real Estate**          | Evaluates property data, summarizes market trends, and supports investment decisions. |
| **Retail**               | Enhances customer experience, tracks sales data, and suggests optimization strategies. |
| **Supply Chain**         | Monitors logistics, optimizes inventory, and forecasts demand. |
| **Transportation**       | Provides route optimization, fleet management, and traffic pattern analysis. |
| **Travel & Hospitality** | Personalizes travel recommendations, optimizes bookings, and improves guest experiences. |

#### Example: Using the Customer Support Template

```python
from reasonchain.domain_templates.customer_support import CustomerSupportTemplate

# Initialize a Customer Support reasoning pipeline
pipeline = CustomerSupportTemplate()
pipeline.add_step("Analyze the customer's query.")
pipeline.add_step("Retrieve similar case resolutions.")
pipeline.add_step("Provide a detailed response to resolve the issue.")

response = pipeline.execute()
print(response)
```

#### Customizing Templates
Each template is fully customizable. You can modify the reasoning steps, add additional logic, or integrate with external APIs to extend the capabilities.

For example, to add a step in the Finance template:
```python
from reasonchain.domain_templates.finance import FinanceTemplate

pipeline = FinanceTemplate()
pipeline.add_step("Analyze the quarterly revenue data.")
pipeline.add_step("Provide insights into expenditure trends.")
pipeline.add_step("Forecast revenue for the next quarter.")

response = pipeline.execute()
print(response)
```

#### Extendability
- Easily create new templates for other domains by extending the base template classes.
- Integrate templates with Retrieval-Augmented Generation (RAG) for enhanced reasoning and data retrieval.

---

## Project Structure

The ReasonChain library is organized into the following core components:

- **Core Library (`reasonchain/`)**:
  - Contains the main modules for reasoning pipelines, RAG integration, domain templates, and utilities.
- **Examples (`examples/`)**:
  - Demonstrates the use of RAG pipelines, multi-agent systems, domain-specific templates, and hybrid reasoning.
- **Unit Tests (`tests/`)**:
  - Validates the functionality of various components.
- **Pre-trained Models and Outputs**:
  - Includes directories like `fine_tuned_model/`, `markdown_images/`, and `parsed_chunks/` for storing models and processed outputs.

For a detailed breakdown of the project structure, see the [CONTRIBUTING.md](CONTRIBUTING.md) or the `docs/` folder.

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

1. **Cloud Storage Integration**:
   - Complete integration with AWS S3, Google Cloud Storage, and Azure Blob Storage.

2. **Advanced Multi-Agent Orchestration**:
   - Enhanced coordination and task distribution among agent teams.

3. **Real-time Data Streaming**:
   - Support for real-time data ingestion from Kafka, Redis, and other streaming platforms.

4. **Advanced Analytics Dashboard**:
   - Web-based interface for monitoring RAG performance and vector database metrics.

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