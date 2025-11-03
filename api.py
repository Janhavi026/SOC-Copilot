# main.py

import uuid
import os
import sys
import configparser
import logging
import asyncio
import time
import json
import pprint 
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
from sse_starlette.sse import EventSourceResponse

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Project Path and Module Imports ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.append(parent_dir)
    from chatbot import DynamicChatbot
    from embedder import Embedder
    from retriever import Retriever
    from elasticsearch_realtime_query import ElasticsearchQueryHandler
except ImportError as e:
    logger.critical(f"A required module could not be imported: {e}. The application cannot start.", exc_info=True)
    sys.exit(1)

# --- FastAPI App and CORS Configuration ---
app = FastAPI(title="Security Chatbot API", version="1.0.0", description="An API for a security operations chatbot with real-time query and RAG capabilities.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models for API Data Validation ---
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str

class InteractiveOption(BaseModel):
    title: str
    payload: str

class ActionStatusData(BaseModel):
    sr_no: Optional[int] = None
    status: Optional[str] = None

class DetailsTableItem(BaseModel):
    Field: str
    Value: Any 

class AlertResultItem(BaseModel):
    id: str
    title: str
    timestamp: str
    summary: str
    relevance_score: float
    full_document: Dict[str, Any]
    details_table: List[DetailsTableItem]

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    options: Optional[List[InteractiveOption]] = None
    data: Optional[List[Dict[str, Any]]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    is_ready: bool

# --- Global State ---
chatbot_instance: Optional[DynamicChatbot] = None
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# --- Middleware for Request Logging ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Took: {process_time:.2f}ms")
    return response

# --- Chatbot Initialization ---
def initialize_chatbot():
    global chatbot_instance
    try:
        config = configparser.ConfigParser()
        config_path = os.path.join(script_dir, 'config.ini')
        if not os.path.exists(config_path):
            raise FileNotFoundError("config.ini not found.")
        config.read(config_path)
        
        # +++ ADDED FOR OLLAMA +++
        logger.info("Initializing with Ollama settings...")
        
        # FIX: Removed invalid characters (**) from the end of these lines
        OLLAMA_BASE_URL = config.get('OLLAMA_SETTINGS', 'ollama_api_base')
        OLLAMA_MODEL = config.get('OLLAMA_SETTINGS', 'ollama_model')
        
        if not OLLAMA_MODEL or not OLLAMA_BASE_URL:
            raise ValueError("Ollama settings ('ollama_model', 'ollama_api_base') not found in config.ini.")
        # ++++++++++++++++++++++
        
        EMBED_MODEL = config.get('EMBEDDER_SETTINGS', 'embedding_model_name')
        ES_TIMEOUT = config.getint('ELASTICSEARCH', 'timeout', fallback=60)
        
        embedder = Embedder(model_name=EMBED_MODEL)
        retriever = Retriever(chroma_data_path="chroma_data", collection_name="rag_documents")
        es_query_handler = ElasticsearchQueryHandler(config_path=config_path, request_timeout=ES_TIMEOUT)

        # NOTE: This assumes the DynamicChatbot class will be updated to accept
        # ollama_base_url and ollama_model parameters for its initialization.
        chatbot_instance = DynamicChatbot(
            ollama_base_url=OLLAMA_BASE_URL, 
            ollama_model=OLLAMA_MODEL,
            embedder=embedder, 
            retriever=retriever, 
            es_query_handler=es_query_handler
        )

        logger.info("✅ Chatbot initialization completed successfully.")
        return True
    except Exception as e:
        logger.critical(f"❌ Failed to initialize chatbot: {e}", exc_info=True)
        return False

# --- Application Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting chatbot API server...")
    if not initialize_chatbot():
        logger.critical("🚨 Startup failed: Chatbot could not be initialized.")

@app.on_event("shutdown")
def shutdown_event():
    executor.shutdown(wait=True)

# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    is_ready = chatbot_instance is not None
    return HealthResponse(status="ready" if is_ready else "initialization_failed", timestamp=datetime.now().isoformat(), is_ready=is_ready)

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot is not available.")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info("--- NEW CHAT REQUEST ---")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"User Message: '{request.message}'")
    
    try:
        loop = asyncio.get_running_loop()
        response_data: Dict[str, Any] = await loop.run_in_executor(
            executor, 
            chatbot_instance.unified_response_orchestrator, 
            request.message, 
            session_id
        )
        
        logger.info(f"--- FINAL RESPONSE DATA FOR SESSION {session_id} ---")
        pprint.pprint(response_data)
        logger.info("-------------------------------------------------")

        return ChatResponse(
            response=response_data.get("message", "Request processed."),
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            options=response_data.get("options"),
            data=response_data.get("data")
        )
    except Exception as e:
        logger.error(f"Error processing chat request for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")

@app.get("/stream-session-updates/{session_id}", tags=["Chat"])
async def stream_session_updates(session_id: str):
    """
    Provides a continuous stream of updates for a given session,
    primarily for completed SOAR actions.
    """
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot is not available.")

    async def event_generator():
        try:
            logger.info(f"SSE stream opened for session_id: {session_id}")
            while True:
                session = chatbot_instance.session_contexts.get(session_id)
                
                if session:
                    logger.info(f"SSE Check: Session {session_id} is tracking SR_NOs: {session.get('pending_sr_numbers')}")
                
                if not session or not session.get('pending_sr_numbers'):
                    await asyncio.sleep(5)
                    continue

                sr_numbers_to_check = session['pending_sr_numbers'].copy()
                completed_sr_numbers = []

                for sr_no in sr_numbers_to_check:
                    status_result = chatbot_instance.response_orchestrator.get_action_status_by_sr_no(sr_no)
                    current_status = status_result.get("db_status", "pending").lower()

                    if current_status != 'pending':
                        logger.info(f"Detected status change for SR_NO {sr_no} to '{current_status.upper()}' in session {session_id}")
                        
                        final_status_message = "has been successfully completed."
                        if current_status in ['failed', 'error']:
                            final_status_message = f"has failed with status: {current_status.upper()}."
                        
                        icon = "✅" if current_status not in ['failed', 'error'] else "❌"
                        message = f"{icon} Update: Your action for **SR_NO {sr_no}** {final_status_message}"
                        
                        yield {
                            "event": "update", 
                            "data": json.dumps({"status": current_status, "message": message})
                        }
                        
                        completed_sr_numbers.append(sr_no)
                
                if completed_sr_numbers:
                    session['pending_sr_numbers'] = [
                        sr for sr in session['pending_sr_numbers'] if sr not in completed_sr_numbers
                    ]
                
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info(f"SSE stream closed for session_id: {session_id}")
        except Exception as e:
            logger.error(f"Error in SSE stream for session {session_id}: {e}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"status": "error", "message": "An error occurred in the update stream."})}

    return EventSourceResponse(event_generator())

@app.post("/clear_session", tags=["Chat"])
async def clear_session(request: SessionRequest):
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot is not available.")
    chatbot_instance.clear_session(request.session_id)
    return {"status": "success", "message": f"Session {request.session_id} cleared."}

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8501)