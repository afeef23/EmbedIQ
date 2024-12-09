from fastapi import FastAPI
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI()

class QueryInput(BaseModel):
    question: str
    context: List[str]
    chat_history: Optional[List[tuple]] = None

class LLMService:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"
        )
    
    async def generate_response(self, query: QueryInput):
        formatted_prompt = f"""
        Context: {' '.join(query.context)}
        
        Chat History: {query.chat_history if query.chat_history else 'No previous conversation'}
        
        Question: {query.question}
        
        Please provide a response based on the context above.
        """
        response = self.llm.invoke(formatted_prompt)
        return {"response": response.content}

@app.post("/generate")
async def generate_response(query: QueryInput):
    service = LLMService()
    return await service.generate_response(query)
