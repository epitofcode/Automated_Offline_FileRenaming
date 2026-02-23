import os
import shutil
import logging
import hashlib
from typing import List, Optional
from datetime import datetime

# Document Processing
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# AI & Vector Logic
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from unidecode import unidecode

# --- CONFIGURATION ---
TARGET_FOLDER = r"C:\Users\Sritej\Desktop\ayman\Computer_architecture"
DB_FOLDER = "./local_chroma_db"             # Where the vector DB lives
MODEL_NAME = "llama3"                       # Ensure you ran `ollama pull llama3`

# --- LOGGING SETUP (Expert Practice) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DocumentIngestor:
    """Handles the messy reality of reading file formats."""
    
    @staticmethod
    def read_file(file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext == ".txt" or ext == ".md":
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                return []
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []

class SemanticRenamer:
    """Uses a Local LLM to analyze content and generate a filename."""
    
    def __init__(self, model_name=MODEL_NAME):
        self.llm = ChatOllama(model=model_name, temperature=0)

    def generate_filename(self, text_content: str) -> str:
        """Reads the first 2000 chars and hallucinates a filename."""
        prompt = (
            "You are a file management assistant. Analyze the text below and generate a concise, "
            "descriptive filename. "
            "Format: YYYY-MM-DD_Topic_Type (e.g., 2023-05-12_Physics_Notes). "
            "If no date is found, use 'UnknownDate'. "
            "ONLY output the filename. Do not output extension. Do not output explanation.\n\n"
            f"TEXT: {text_content[:2000]}"
        )
        try:
            response = self.llm.invoke(prompt)
            # Clean up the output (remove spaces, illegal chars)
            clean_name = unidecode(response.content.strip()).replace(" ", "_").replace("/", "-")
            return clean_name
        except Exception as e:
            logger.error(f"LLM Renaming failed: {e}")
            return "Unknown_Document"

    def safe_rename(self, original_path: str, new_name: str) -> str:
        """Renames file safely, handling duplicates."""
        directory = os.path.dirname(original_path)
        ext = os.path.splitext(original_path)[1]
        new_filename = f"{new_name}{ext}"
        new_path = os.path.join(directory, new_filename)

        # Collision handling: if file exists, append _1, _2, etc.
        counter = 1
        while os.path.exists(new_path):
            if new_path == original_path: return original_path # It's already named correctly
            new_filename = f"{new_name}_{counter}{ext}"
            new_path = os.path.join(directory, new_filename)
            counter += 1
            
        try:
            os.rename(original_path, new_path)
            logger.info(f"Renamed: {os.path.basename(original_path)} -> {new_filename}")
            return new_path
        except Exception as e:
            logger.error(f"OS Rename failed: {e}")
            return original_path

class OfflineRAG:
    """The Brain. Indexes files and answers questions."""
    
    def __init__(self, persist_dir=DB_FOLDER):
        self.persist_dir = persist_dir
        # Free, offline, high-quality embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = None
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)

    def ingest_and_index(self, folder_path):
        """Phase 1 & 2: Rename files, then Index them."""
        renamer = SemanticRenamer()
        all_docs = []

        logger.info("Starting Batch Processing...")
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                original_path = os.path.join(root, file)
                
                # 1. Read Content
                docs = DocumentIngestor.read_file(original_path)
                if not docs: continue
                
                # Combine text for analysis
                full_text = " ".join([d.page_content for d in docs])
                
                # 2. Smart Rename (Only if file hasn't been processed/renamed yet)
                # (In a real pro system, we'd check a database to see if we already did this)
                suggested_name = renamer.generate_filename(full_text)
                final_path = renamer.safe_rename(original_path, suggested_name)
                
                # 3. Prepare for Indexing (Update source metadata to new path)
                for d in docs:
                    d.metadata['source'] = final_path
                    d.metadata['filename'] = os.path.basename(final_path)
                all_docs.extend(docs)

        if not all_docs:
            logger.warning("No documents found to index.")
            return

        # 4. Split and Index
        logger.info(f"Indexing {len(all_docs)} document pages...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        logger.info("Indexing Complete. Database Saved.")

    def load_db(self):
        """Loads existing DB without re-indexing."""
        if os.path.exists(self.persist_dir):
            self.vector_db = Chroma(persist_directory=self.persist_dir, embedding=self.embeddings)
            return True
        return False

    def chat(self, query):
        if not self.vector_db: return "Database not loaded."
        
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        result = qa.invoke({"query": query})
        return result['result'], result['source_documents']

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    system = OfflineRAG()
    
    print("--- 1. PROCESS & RENAME FILES? ---")
    choice = input("Type 'yes' to scan folder, rename files, and rebuild index. Type 'no' to just chat: ")
    
    if choice.lower() == 'yes':
        system.ingest_and_index(TARGET_FOLDER)
    else:
        if not system.load_db():
            print("No database found. You must run 'yes' at least once.")
            exit()

    print("\n--- 2. CHAT SYSTEM (OFFLINE) ---")
    print("Ask questions about your files. Type 'exit' to quit.\n")
    
    while True:
        q = input("User: ")
        if q.lower() == 'exit': break
        
        answer, docs = system.chat(q)
        print(f"\nAI: {answer}\n")
        print("References:")
        for d in docs:
            print(f"- {d.metadata.get('filename', 'Unknown')}")