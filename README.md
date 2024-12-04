# School Knowledge Base Chatbot

A sophisticated chatbot system designed to handle school-related queries using the Mistral LLM model and Milvus vector database. The system processes PDF documents (such as FAQs, handbooks, etc.), chunks them intelligently, and provides accurate responses to student queries.

## 🌟 Features

- PDF document processing and intelligent text chunking
- Vector similarity search using Milvus
- Integration with Mistral LLM for natural language understanding
- Specialized handling of FAQ-style documents
- Support for multiple document types and formats
- Metadata extraction and categorization
- Question-Answer pair extraction from documents

## 🛠️ Technologies Used

- Python 3.x
- Mistral 9B Instruct Model
- Milvus Vector Database
- Sentence Transformers (all-MiniLM-L6-v2)
- PyMilvus
- ctransformers
- pdfplumber
- Other supporting libraries (see requirements.txt)

## 📋 Prerequisites

- Python 3.x
- Milvus server running locally or remotely
- Sufficient storage for model weights and document embeddings
- GPU recommended for better performance

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/school_chatbot.git
cd school_chatbot
```

2. Install the required packages:
```bash
pip install -e .
```

3. Set up Milvus:
- Follow the [official Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)
- Start the Milvus server

4. Configure the application:
- Copy `config/config.yaml.example` to `config/config.yaml`
- Update the configuration values as needed:
  - Milvus connection settings
  - Model paths and parameters
  - Processing directories
  - API and GUI settings

## 📁 Project Structure

```
school_chatbot/
├── src/
│   ├── data_processing/     # PDF processing and chunking
│   ├── db/                  # Milvus client and database operations
│   ├── llm/                 # Mistral model integration
│   └── utils/              # Utility functions
├── tests/                  # Test files and test data
├── config/                 # Configuration files
└── setup.py               # Package setup file
```

## 🔧 Usage

1. Process PDF documents:
```python
from src.data_processing.pdf_processor import PDFProcessor

processor = PDFProcessor(input_dir="data/raw", output_dir="data/processed")
processor.process_directory()
```

2. Load processed documents into Milvus:
```python
from src.data_loading.data_loader import DataLoader

loader = DataLoader()
loader.load_directory("data/processed")
```

3. Query the system:
```python
from src.llm.mistral_client import MistralClient
from src.db.milvus_client import MilvusClient

# Initialize clients
mistral = MistralClient()
milvus = MilvusClient()

# Search for relevant context
results = milvus.search(query_embedding)

# Generate response using Mistral
response = mistral.generate_response(query, context=results)
```

The test suite includes:
- PDF processing tests
- Text chunking tests
- Milvus integration tests
- Data loading tests

## ⚙️ Configuration

Key configuration options in `config.yaml`:

```yaml
milvus:
  host: "localhost"
  port: 19530
  collection_name: "school_docs"

model:
  name: "Mistral-9B-Instruct"
  path: "/path/to/model/weights"
  context_length: 4096

embedding:
  model_name: "all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
```




## 🙏 Acknowledgments

- Mistral AI for the language model
- Milvus team for the vector database
- Sentence Transformers team for the embedding models
