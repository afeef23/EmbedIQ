from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel
from typing import List, Optional, Tuple
import asyncio
import logging
import socket

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client for reuse
http_client = None

class QueryInput(BaseModel):
    question: str
    kb_type: str
    chat_history: Optional[List[Tuple[str, str]]] = None

async def get_client():
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=60.0)
    return http_client

class QueryService:
    async def check_rag_service(self) -> bool:
        try:
            client = await get_client()
            response = await client.get("http://rag-service:8002/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"RAG service check failed: {str(e)}")
            return False

    async def wait_for_rag_service(self, max_retries: int = 10, delay: int = 5) -> bool:
        for attempt in range(max_retries):
            if await self.check_rag_service():
                logger.info("Successfully connected to RAG service")
                return True
            logger.info(f"Waiting for RAG service, attempt {attempt + 1}/{max_retries}")
            await asyncio.sleep(delay)
        return False

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up query service...")
    service = QueryService()
    if not await service.wait_for_rag_service():
        logger.error("Failed to connect to RAG service during startup")
    else:
        logger.info("Query service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    if http_client:
        await http_client.aclose()

@app.get("/health")
async def health_check():
    try:
        # Simple self-check first
        hostname = socket.gethostname()
        logger.info(f"Health check initiated from {hostname}")
        
        # Basic service check
        return {"status": "healthy", "hostname": hostname}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/query")
async def process_query(query: QueryInput):
    service = QueryService()
    if not await service.check_rag_service():
        raise HTTPException(status_code=503, detail="RAG service is unavailable")
    
    client = await get_client()
    try:
        # Check if vector store exists
        store_check = await client.get(f"http://rag-service:8002/store/{query.kb_type}/exists")
        store_check.raise_for_status()
        
        if not store_check.json()["exists"]:
            raise HTTPException(
                status_code=400, 
                detail=f"No knowledge base found for {query.kb_type}. Please upload documents first."
            )
        
        # Get embeddings
        logger.info("Getting embeddings...")
        embed_response = await client.post(
            "http://rag-service:8002/embed",
            json={"texts": [query.question]}
        )
        embed_response.raise_for_status()
        question_embedding = embed_response.json()["embeddings"][0]
        
        # Search vectors
        logger.info("Searching vectors...")
        search_response = await client.post(
            f"http://rag-service:8002/search/{query.kb_type}",
            json=question_embedding
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        
        if "error" in search_data:
            raise HTTPException(status_code=400, detail=search_data["error"])
            
        logger.info(f"Search response: {search_data}")
        context = search_data.get("documents", [])
        
        # Generate response
        logger.info("Generating response...")
        llm_response = await client.post(
            "http://rag-service:8002/generate",
            json={
                "question": query.question,
                "context": context,
                "chat_history": query.chat_history
            }
        )
        llm_response.raise_for_status()
        return llm_response.json()
        
    except httpx.HTTPError as e:
        logger.error(f"RAG service error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"RAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Internal error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
