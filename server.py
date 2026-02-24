import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from offline_brain import OfflineRAG, logger

app = FastAPI(title="Offline RAG API")

# Allow requests from your Vercel frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://automated-offline-filerenaming.vercel.app", # Replace with your actual Vercel URL
        "*" 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG System globally
rag_system = OfflineRAG()
# Attempt to load existing DB on startup
rag_system.load_db()

class ProcessRequest(BaseModel):
    folder_path: str

class ChatRequest(BaseModel):
    query: str

@app.post("/api/process")
async def process_folder(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Endpoint to trigger folder ingestion."""
    if not os.path.exists(request.folder_path) or not os.path.isdir(request.folder_path):
        raise HTTPException(status_code=400, detail="Folder path does not exist on the local machine.")
    
    # Run in background so the HTTP request doesn't hang the UI
    background_tasks.add_task(rag_system.ingest_and_index, request.folder_path)
    return {"message": "Processing started in the background. Check logs.", "status": "processing"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Endpoint for chatting with the documents."""
    if not rag_system.vector_db:
        if not rag_system.load_db():
            raise HTTPException(status_code=400, detail="Database not loaded. Please process a folder first.")
    
    answer, docs = rag_system.chat(request.query)
    
    # Extract unique references
    references = list(set([d.metadata.get('filename', 'Unknown') for d in docs]))
    return {"answer": answer, "references": references}

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """Streams system.log to the frontend in real-time."""
    await websocket.accept()
    log_file = "system.log"
    
    try:
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f: pass
            
        with open(log_file, 'r', encoding='utf-8') as f:
            # Start at the beginning of the file for the current session
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue
                await websocket.send_text(line.strip())
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try: await websocket.send_text(f"Log Error: {e}")
        except: pass

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
