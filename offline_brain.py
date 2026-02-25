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
TARGET_FOLDER = r"C:\Users\Sritej\Desktop\qwerty"
DB_FOLDER = "./local_chroma_db"             # Where the vector DB lives
MODEL_NAME = "llama3"                       # Ensure you ran `ollama pull llama3`

# --- LOGGING SETUP (Expert Practice) ---
class PersistentFlushHandler(logging.FileHandler):
    """Custom handler that flushes to disk immediately so WebSockets see the data."""
    def emit(self, record):
        super().emit(record)
        self.flush()

log_handler = PersistentFlushHandler("system.log", mode='w')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[log_handler, logging.StreamHandler()]
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
                # Ensure we handle markdown and text with proper encoding
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
        """Strict few-shot prompting to force Topic_Type output."""
        prompt = (
            "TASK: Generate a 2-3 word Topic_Type filename from the text.\n"
            "RULES:\n"
            "1. Output ONLY words and underscores (e.g. Topic_Type).\n"
            "2. NO sentences. NO 'Based on'. NO 'Here is'.\n"
            "3. NO conversational filler.\n\n"
            "EXAMPLES:\n"
            "Text: 'Calculus 101 Lecture notes on derivatives...' -> Calculus_Notes\n"
            "Text: 'Project Alpha Final Report for Q3 2023...' -> ProjectAlpha_Report\n"
            "Text: 'Official meeting minutes from Jan 5th...' -> Meeting_Minutes\n\n"
            f"TEXT TO ANALYZE:\n{text_content[:2500]}\n\n"
            "FINAL FILENAME (Topic_Type):"
        )
        try:
            response = self.llm.invoke(prompt)
            # Remove ALL conversational fluff (even if it's the first line)
            lines = response.content.strip().split("\n")
            # Look for the line that actually looks like a filename (contains underscore, no spaces)
            result = ""
            for line in lines:
                candidate = line.replace("Topic_Type:", "").replace("Result:", "").strip()
                if "_" in candidate and " " not in candidate:
                    result = candidate
                    break
            
            if not result:
                result = lines[-1].strip() # Fallback to last line

            # Clean up: Remove introductory phrases
            bad_phrases = ["based_on", "here_is", "the_document", "i_will_generate", "i_would_generate", "lets_apply"]
            clean_name = unidecode(result).replace(" ", "_").replace("/", "-")
            
            import re
            # If the LLM still gave a sentence, take only the last few words or words with underscores
            if len(clean_name) > 40:
                parts = [p for p in clean_name.split("_") if p.lower() not in bad_phrases]
                clean_name = "_".join(parts[-3:]) # Take last 3 relevant words

            clean_name = re.sub(r'[^\w\-]', '', clean_name)
            clean_name = re.sub(r'_\d{4}-\d{2}-\d{2}', '', clean_name)
            
            if len(clean_name) > 50:
                clean_name = clean_name[:47] + "..."
                
            return clean_name if clean_name else "Processed_Doc"
        except Exception as e:
            logger.error(f"Heuristic Renaming failed: {e}")
            return "General_Document"

    def safe_rename(self, original_path: str, new_base_name: str, file_date: str) -> str:
        """Renames file safely, handling duplicates, with date at the end."""
        directory = os.path.dirname(original_path)
        ext = os.path.splitext(original_path)[1]
        
        # New format: Topic_Type_YYYY-MM-DD
        final_base = f"{new_base_name}_{file_date}"
        new_filename = f"{final_base}{ext}"
        new_path = os.path.join(directory, new_filename)

        # Collision handling
        counter = 1
        while os.path.exists(new_path):
            if os.path.abspath(new_path) == os.path.abspath(original_path): 
                return original_path
            new_filename = f"{final_base}_{counter}{ext}"
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
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = None
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)

    def get_creation_date(self, path):
        """Gets formatted creation date from file metadata."""
        try:
            timestamp = os.path.getctime(path)
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        except:
            return "UnknownDate"

    def ingest_and_index(self, folder_path):
        """Phase 1 & 2: Rename files (Topic_Type_Date), then Index them."""
        renamer = SemanticRenamer()
        all_docs = []

        # CLEAN START: If we are indexing, we want a fresh DB to avoid conflicts
        if os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
                logger.info("Cleared existing database for fresh indexing.")
            except Exception as e:
                logger.warning(f"Could not clear database folder: {e}")

        logger.info(f"Starting Batch Processing in: {folder_path}")
        
        target_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                target_files.append(os.path.join(root, file))

        for original_path in target_files:
            if not os.path.exists(original_path): continue
            
            docs = DocumentIngestor.read_file(original_path)
            if not docs: continue
            
            full_text = " ".join([d.page_content for d in docs])
            
            # Use OS creation date
            c_date = self.get_creation_date(original_path)
            
            # Smart Rename (Skip if already follows our pattern exactly)
            base_name = os.path.basename(original_path)
            # Pattern check: Does it have underscores and end with a 10-char date?
            parts = base_name.split("_")
            is_already_renamed = len(parts) >= 3 and len(parts[-1].split(".")[0]) == 10
            
            # Additional check: If the name is ridiculously long or contains conversational keywords, force a re-rename
            is_bad_name = len(base_name) > 60 or "Please_provide" in base_name or "text_you'd_like" in base_name
            
            if not is_already_renamed or is_bad_name:
                logger.info(f"Analyzing content for renaming: {base_name}")
                suggested_topic = renamer.generate_filename(full_text)
                final_path = renamer.safe_rename(original_path, suggested_topic, c_date)
            else:
                logger.info(f"Skipping rename (already formatted): {base_name}")
                final_path = original_path
            
            for d in docs:
                d.metadata['source'] = final_path
                d.metadata['filename'] = os.path.basename(final_path)
            all_docs.extend(docs)

        if not all_docs:
            logger.warning("No documents found to index.")
            return

        logger.info(f"Indexing {len(all_docs)} document pages...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        
        try:
            from chromadb.config import Settings
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                client_settings=Settings(anonymized_telemetry=False)
            )
            logger.info("Indexing Complete. Database Saved.")
        except Exception as e:
            logger.error(f"Vector DB Failure: {e}")

    def load_db(self):
        """Loads existing DB without re-indexing."""
        if os.path.exists(self.persist_dir):
            self.vector_db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
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