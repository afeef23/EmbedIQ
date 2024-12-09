from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from typing import List
import httpx
import logging
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

class DocumentService:
    def __init__(self):
        self.upload_folder = "uploaded_docs"
        self.ensure_upload_folder()
    
    def ensure_upload_folder(self):
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
    
    async def save_document(self, file: UploadFile, kb_type: str):
        try:
            kb_folder = os.path.join(self.upload_folder, kb_type)
            os.makedirs(kb_folder, exist_ok=True)
            file_path = os.path.join(kb_folder, file.filename)
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            return {"status": "success", "path": file_path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_documents(self, kb_type: str) -> List[str]:
        try:
            kb_folder = os.path.join(self.upload_folder, kb_type)
            if os.path.exists(kb_folder):
                return [f for f in os.listdir(kb_folder) if f.endswith('.pdf')]
            return []
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

async def process_document(file_path: str, knowledge_base: str) -> httpx.Response:
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # Read the PDF content
        content = ""
        logger.info(f"Processing document: {file_path}")
        
        try:
            pdf_reader = PdfReader(file_path)
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                logger.info(f"Extracted {len(page_text)} characters from page {i+1}")
                content += page_text + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")
        
        if not content.strip():
            logger.error("No text content extracted from PDF")
            raise HTTPException(status_code=400, detail="No text content extracted from PDF")
            
        logger.info(f"Total extracted content length: {len(content)}")
        
        # Send to RAG service for processing
        async with httpx.AsyncClient() as client:
            # First create embeddings
            logger.info("Sending content for embedding...")
            embed_response = await client.post(
                "http://rag-service:8002/embed",
                json={"texts": [content]}
            )
            embed_response.raise_for_status()
            embeddings = embed_response.json()["embeddings"]
            logger.info("Successfully created embeddings")
            
            # Then store in vector store
            store_response = await client.post(
                f"http://rag-service:8002/store/{knowledge_base}",
                json={
                    "texts": [content],
                    "embeddings": embeddings
                }
            )
            store_response.raise_for_status()
            return store_response
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/upload/{knowledge_base}")
async def upload_document(knowledge_base: str, file: UploadFile):
    service = DocumentService()
    try:
        # Save the file using DocumentService
        result = await service.save_document(file, knowledge_base)
        
        # Process the document
        response = await process_document(result["path"], knowledge_base)
        if not response.is_success:
            logger.error(f"Failed to process document: {response.text}")
            return {"error": f"Failed to process document: {response.text}"}
            
        return {"message": "Document processed successfully"}
        
    except Exception as e:
        logger.error(f"Failed to handle upload: {str(e)}")
        return {"error": f"Failed to handle upload: {str(e)}"}

@app.get("/documents/{kb_type}")
async def get_documents(kb_type: str):
    service = DocumentService()
    documents = service.get_documents(kb_type)
    return {"documents": documents}

@app.get("/health")
async def health_check():
    try:
        # Check if upload folder exists and is writable
        test_folder = os.path.join("uploaded_docs", "test")
        os.makedirs(test_folder, exist_ok=True)
        test_file = os.path.join(test_folder, "test.txt")
        
        with open(test_file, "w") as f:
            f.write("test")
        
        os.remove(test_file)
        os.rmdir(test_folder)
        
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
