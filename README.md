# Offline RAG Brain 🧠

An intelligent, fully offline Document Management and Retrieval-Augmented Generation (RAG) system. This tool not only allows you to chat with your local documents but also automatically renames them semantically based on their content using a local LLM.

## 🚀 Features

*   **Semantic Renaming**: Uses Llama 3 to analyze document content and generate descriptive filenames (Format: `YYYY-MM-DD_Topic_Type`).
*   **Fully Offline**: Your data never leaves your machine. Uses Ollama for LLM tasks and HuggingFace for local embeddings.
*   **Multi-format Support**: Handles `.pdf`, `.docx`, `.txt`, and `.md` files.
*   **Persistent Vector DB**: Stores document embeddings in a local ChromaDB instance for fast retrieval.
*   **Expert Logging**: Keeps track of all processing and errors in `system.log`.

## 📋 Requirements

*   **Python**: 3.10 or higher.
*   **Ollama**: Installed and running on your system.
*   **LLM Model**: `llama3` (or your preferred model configured in the script).
*   **Disk Space**: For the local vector database and model storage.

## 🛠️ How to Setup

### 1. Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com). Once installed, pull the required model:
```bash
ollama pull llama3
```

### 2. Prepare the Environment
Create and activate the virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
Run the following command to install all necessary libraries:
```bash
pip install langchain langchain-community langchain-ollama langchain-classic langchain-text-splitters chromadb pypdf python-docx sentence-transformers unidecode
```

## 📂 Configuration

Open `offline_brain.py` and update the `TARGET_FOLDER` path to the directory containing your documents:

```python
TARGET_FOLDER = r"C:\Users\YourName\Documents\MyFolder"
```

## 📖 How to Use

Run the main script:
```powershell
.\.venv\Scripts\python offline_brain.py
```

### Workflow:
1.  **Batch Processing**: When prompted, type `yes` to scan your folder. The system will:
    *   Read each document.
    *   Rename the file based on its content (if it hasn't been renamed already).
    *   Split the text into chunks and index them into the Vector DB.
2.  **Chatting**: Type `no` if you've already indexed your files and want to go straight to chatting.
3.  **Ask Questions**: Type your queries in the terminal. The AI will answer based on your local documents and provide the source filenames as references.
4.  **Exit**: Type `exit` to close the session.

## 📁 Project Structure

*   `offline_brain.py`: The main application logic.
*   `local_chroma_db/`: Directory where the vector database is persisted.
*   `system.log`: Log file for debugging and tracking actions.
*   `.venv/`: Python virtual environment.

## ⚙️ Technical Architecture

This project implements a sophisticated **local-first RAG pipeline** with an automated pre-processing layer:

1.  **Semantic File Pre-processing**: 
    *   Unlike traditional RAG, this system uses an **LLM-in-the-loop** approach for file management. 
    *   Before indexing, the first 2000 characters of each file are sent to `llama3` with a specific prompt to determine the document's date, topic, and type. 
    *   The `SemanticRenamer` class then safely renames the physical file on disk using `unidecode` for filename sanitization and logic to handle naming collisions.

2.  **Document Ingestion Engine**: 
    *   The system uses specialized loaders (`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`) to extract raw text.
    *   Text is processed via `RecursiveCharacterTextSplitter` into 1000-character chunks with a 200-character overlap to ensure context is preserved across boundaries.

3.  **Vectorization & Storage**:
    *   **Embeddings**: Uses the `all-MiniLM-L6-v2` Sentence-Transformer model via HuggingFace. This is a lightweight, high-performance model that runs locally on CPU/GPU.
    *   **Vector Store**: `ChromaDB` is used as the persistent vector database to store the high-dimensional embeddings and document metadata.

4.  **Retrieval & Generation (RAG)**:
    *   When a user asks a question, the system performs a **Similarity Search** against the vector store to find the top 5 most relevant chunks.
    *   These chunks are injected into a prompt context and sent to the local `llama3` model via `ChatOllama`.
    *   The `RetrievalQA` chain (from `langchain-classic`) manages the "Stuff" chain logic, ensuring the LLM only answers based on the provided local data.

## ⚠️ Important Notes
*   Ensure **Ollama** is running before starting the script.
*   The first time you run the script, it will download the embedding model (~20MB) from HuggingFace.
