from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from app.services import rag_service

router = APIRouter()
templates = Jinja2Templates(directory="app/templates") # Note the new path

class ChatRequest(BaseModel):
    message: str

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@router.post("/chat")
async def chat(req: ChatRequest):
    lang = rag_service.detect_language(req.message)
    standalone = rag_service.rewrite_query(req.message)
    # Call your service functions
    records = rag_service.retrieve(standalone, top_k=5)
    answer = rag_service.generate_answer(req.message, records)
    rag_service.store_memory(req.message, answer)
    
    primary_source = records[0]["node"] if records else {}
    return {
        "answer": answer,
        "language_detected": lang,
        "source_title": primary_source.get("title", "Unknown"),
        "page": primary_source.get("page", "N/A")
    }

