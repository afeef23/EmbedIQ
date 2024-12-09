from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import os
import logging
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global RAG service instance
rag_service = None

class TextInput(BaseModel):
    texts: List[str]

class VectorInput(BaseModel):
    texts: List[str]
    embeddings: List[List[float]]

class QueryInput(BaseModel):
    question: str
    context: List[str]
    chat_history: Optional[List[tuple]] = None

class RAGService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            logger.info("Initializing RAG Service...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vector_stores = {}
            self.llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name="mixtral-8x7b-32768"
            )
            self.initialized = True
            logger.info("RAG Service initialized")

    async def create_embeddings(self, texts: List[str]):
        return self.embeddings.embed_documents(texts)

    def create_store(self, kb_type: str, texts: List[str], embeddings: List[List[float]]):
        logger.info(f"Creating vector store for {kb_type}")
        self.vector_stores[kb_type] = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=self.embeddings
        )
        logger.info(f"Vector store created for {kb_type}")
        return {"status": "success"}

    def truncate_text(self, text: str, max_chars: int = 4000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def search(self, kb_type: str, query_embedding: List[float], k: int = 5):
        """Search with increased number of results and better relevance"""
        if kb_type not in self.vector_stores:
            logger.error(f"Vector store not found for {kb_type}")
            return {"error": "Vector store not found"}
        
        # Retrieve more documents for better context
        docs = self.vector_stores[kb_type].similarity_search_by_vector(
            query_embedding, 
            k=5,  # Increased from 3 to 5
            fetch_k=10  # Consider more candidates before selecting top k
        )
        
        # Log retrieved documents for debugging
        logger.info(f"Retrieved {len(docs)} documents for query")
        for i, doc in enumerate(docs):
            logger.info(f"Document {i+1} preview: {doc.page_content[:100]}...")
        
        return {"documents": [doc.page_content for doc in docs]}

    async def generate_response(self, query: QueryInput):
        try:
            # Get more context but still limit size
            truncated_context = [
                self.truncate_text(doc, 1500) for doc in query.context[:5]
            ]
            
            # Enhanced prompt for better information extraction
            formatted_prompt = f"""
            You are an AWS documentation expert. Your task is to provide accurate, detailed information from AWS documentation.

            Important guidelines:
            1. ALWAYS cite specific numbers, limits, and technical details directly from the documentation
            2. For questions about AWS services (like EC2 instances, free tier, etc.), include ALL relevant details found in the context
            3. If multiple pieces of information are related, combine them to provide a complete answer
            4. For technical specifications or limits, quote the exact values from the documentation
            5. If information appears to be missing or unclear, say so explicitly

            Context (AWS Documentation):
            {' '.join(truncated_context)}

            Question: {query.question}

            Please provide a comprehensive answer using ONLY the information from the provided AWS documentation. Include specific details, numbers, and limits where available.
            """
            
            # Generate response using the LLM
            response = self.llm.invoke(formatted_prompt)
            return {"response": response.content}
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating response: {str(e)}"
            )

@app.on_event("startup")
async def startup_event():
    global rag_service
    rag_service = RAGService()
    logger.info("RAG service started")

# Embedding endpoints
@app.post("/embed")
async def create_embeddings(input_data: TextInput):
    global rag_service
    embeddings = await rag_service.create_embeddings(input_data.texts)
    return {"embeddings": embeddings}

# Vector store endpoints
@app.post("/store/{kb_type}")
async def create_vector_store(kb_type: str, input_data: VectorInput):
    global rag_service
    return rag_service.create_store(kb_type, input_data.texts, input_data.embeddings)

@app.post("/search/{kb_type}")
async def search_vectors(kb_type: str, query_embedding: List[float], k: int = 3):
    global rag_service
    return rag_service.search(kb_type, query_embedding, k)

# LLM endpoints
@app.post("/generate")
async def generate_response(query: QueryInput):
    global rag_service
    return await rag_service.generate_response(query)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/store/{kb_type}/exists")
async def check_vector_store(kb_type: str):
    service = RAGService()
    exists = kb_type in service.vector_stores
    return {"exists": exists}
