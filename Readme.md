# ReasonChain

**ReasonChain** is a modular and extensible AI reasoning library designed for creating intelligent agents capable of executing advanced reasoning processes. It supports **Chain of Thought (CoT)** reasoning, **Tree of Thought (ToT)** reasoning, **Parallel Chains**, and integrates with **Retrieval-Augmented Generation (RAG)** for enhanced knowledge management.

---

## Features

1. **Customizable Reasoning Pipelines**:
   - Support for Chain of Thought (CoT), Tree of Thought (ToT), Parallel Chains, and Hybrid Pipelines.
   - Task decomposition, execution, and validation.

2. **RAG Integration**:
   - Retrieve and augment responses using long-term memory stored in vector databases like FAISS.

3. **Short-term and Long-term Memory**:
   - Session-based short-term memory for reasoning chains.
   - Persistent long-term memory powered by FAISS.

4. **LLM Integration**:
   - Seamlessly integrate with OpenAI GPT, Llama, and other models for response generation and summarization.

5. **Utility Tools**:
   - Dynamic complexity evaluation for reasoning steps.
   - Centralized Model Manager for managing LLMs and tasks.

---

## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/reasonchain.git
cd reasonchain
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Configure API Keys**
Create a `.env` file and add your API keys:
```plaintext
OPENAI_API_KEY=your_openai_api_key
```

---

## Usage

### **1. Initialize an Agent**
```python
from reasonchain import Agent

agent = Agent(name="ResearchBot", model_name="gpt-4", api_key="your_openai_api_key")
```

### **2. Build a CoT Reasoning Pipeline**
```python
from reasonchain.cot_pipeline import CoTPipeline

pipeline = CoTPipeline(agent)
pipeline.add_step("Understand the user's question.")
pipeline.add_step("Fetch relevant data from the knowledge base.")
pipeline.add_step("Generate a solution based on available information.")
```

### **3. Execute the Pipeline**
```python
response = pipeline.execute(agent.model_manager)
print(response)
```

### **4. Use Hybrid Pipelines**
The Hybrid Pipeline dynamically selects CoT, ToT, or Parallel Chains based on task complexity.
```python
from reasonchain.cot_pipeline import HybridCoTPipeline

pipeline = HybridCoTPipeline(agent)
pipeline.add_step("Analyze the query.", complexity="low")
pipeline.add_step("Retrieve data from long-term memory.", complexity="medium")
pipeline.add_step("Generate and evaluate multiple solutions.", complexity="high")
response = pipeline.execute(agent.model_manager)
print(response)
```

---

## Advanced Features

### **1. Retrieval-Augmented Generation (RAG)**
Enhance responses with RAG integration using FAISS:
```python
from reasonchain.memory import Memory

memory = Memory(embedding_model="all-MiniLM-L6-v2", use_gpu=False)
query = "Explain the benefits of SQL indexing."
context = memory.retrieve_long_term(query)
print(f"Retrieved Context: {context}")
```

### **2. Model Manager**
Centralized handling of LLMs for tasks like response generation and summarization:
```python
from reasonchain.model_manager import ModelManager

model_manager = ModelManager(model_name="gpt-4", api_key="your_openai_api_key")
response = model_manager.generate_response("How can I optimize my database?")
print(response)

# Summarize text
summary = model_manager.summarize("Long text to summarize...")
print(summary)
```

---

## Examples

The `examples/` directory includes ready-to-run scripts:
- **`simple_agent.py`**: Basic usage of ReasonChain.
- **`faq_bot.py`**: Building an FAQ bot using CoT reasoning.
- **`rag_pipeline_example.py`**: Example of RAG-based reasoning.
- **`parellel_reasoning_example.py`**: Demonstrates parallel reasoning chains.
- **`tree_of_thought_example.py`**: Implements Tree of Thought reasoning.
- **`hybrid_reasoning_example.py`**: Combines CoT, ToT, and parallel reasoning.

---

## Project Structure

```plaintext
ReasonChain/
│
├── reasonchain/                  # Core library
│   ├── __init__.py               # Initialize the library
│   ├── agent.py                  # Defines the Agent class
│   ├── cot_pipeline.py           # Implements CoT pipeline
│   ├── faiss_vector_db.py        # FAISS-based vector database
│   ├── memory.py                 # Memory management
│   ├── model_manager.py          # LLM interactions and summarization
│   ├── rag_integration.py        # RAG pipeline integration
│   ├── utils.py                  # Helper functions
│   ├── enviornment.py            # External API interactions
│
├── examples/                     # Examples for using ReasonChain
├── tests/                        # Unit tests
├── setup.py                      # Installation script
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation
├── .env                          # Environment variables
```

---

## Key Features

1. **Dynamic Reasoning Pipelines**:
   - Supports task-specific reasoning with customizable steps.

2. **RAG Integration**:
   - Retrieve and incorporate external knowledge into reasoning.

3. **LLM Extensibility**:
   - Integrates with OpenAI GPT and other models seamlessly.

4. **Scalable Design**:
   - Modular architecture for easy expansion and collaboration.

---

## Future Enhancements

1. **Pre-trained Reasoning Templates**:
   - Include templates for domain-specific tasks (e.g., healthcare, finance).

2. **Agent Collaboration**:
   - Enable agents to collaborate on complex tasks.

3. **Fine-tuned Models**:
   - Add support for fine-tuning and domain adaptation.

4. **Scalable RAG Support**:
   - Expand integrations to include Pinecone, Weaviate, and other vector stores.

---

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch for your changes:
   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/new-feature
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
