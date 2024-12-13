# **ReasonChain**

**ReasonChain** is a modular and extensible AI reasoning library for creating intelligent agents capable of executing advanced reasoning processes. It supports **Chain of Thought (CoT)** reasoning, **Tree of Thought (ToT)** reasoning, **Parallel Chains**, and integrates with **Retrieval-Augmented Generation (RAG)** for enhanced knowledge management.

---

### **Features**

1. **Customizable Reasoning Pipelines**:
   - Seamless integration of **Chain of Thought (CoT)**, **Tree of Thought (ToT)**, Parallel Chains, and Hybrid Pipelines.
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

6. **Supported File Types**:
   - Extract text, tables, and figures from a wide range of file formats, including:
     - **Documents**: PDF, Word, RTF, Markdown, HTML, LaTeX.
     - **Spreadsheets**: Excel, CSV.
     - **Multimedia**: Images, Videos, Audio.
     - **E-books**: EPUB, MOBI.
   - Provides modular extractors tailored for each file type, ensuring efficient data retrieval.

7. **Domain Templates**:
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
from reasonchain.rag.vector.add_to_vector_db import add_pdf_to_vector_db
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

# Add a document to the vector database
add_pdf_to_vector_db(
    pdf_path="path/to/tesla-report.pdf",
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

## **Examples**

Explore more scripts in the `examples/` directory:
- **`rag_pipeline_example.py`**: Example of using RAG for context-enhanced reasoning.
- **`multi-agent_collaboration.py`**: Multi-agent collaboration example.
- **`tree_of_thought_example.py`**: Demonstrates Tree of Thought reasoning.
- **`hybrid_reasoning_example.py`**: Combines multiple reasoning methods.

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