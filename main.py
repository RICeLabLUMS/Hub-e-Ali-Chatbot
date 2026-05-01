import os
import re
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
app = FastAPI()
load_dotenv()
# Configuration
GROQ_API_KEY =os.getenv("GROQ_API_KEY")
NEO4J_URL =os.getenv("NEO4J_URL")
NEO4J_USER =os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD =os.getenv("NEO4J_PASSWORD")

# Initialize Tools
client = Groq(api_key=GROQ_API_KEY)
hf_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
# In both load_documents.py and main.py
# hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Semantic Memory Storage
conversation_history = []
memory_embeddings = []

#  RAG Logic Functions 

def normalize_text(text):
    text = text.strip()
    if re.search(r'[\u0600-\u06FF]', text):
        text = re.sub("[ًٌٍَُِّْـ]", "", text)
        text = re.sub("[إأآا]", "ا", text)
        text = text.replace("ك", "ک").replace("ی", "ی")
    return text.lower()

def embed(texts):
    return hf_embeddings.embed_documents(texts)

def retrieve_memory(query, top_k=1):
    if not memory_embeddings: return ""
    q_emb = embed([query])[0]
    scores = [np.dot(q_emb, m_emb) for m_emb in memory_embeddings]
    idx = np.argmax(scores)
    return conversation_history[idx]

def rewrite_query(question):
    memory_context = retrieve_memory(question)
    if not memory_context: return question
    prompt = f"Rewrite the question into a complete standalone query.\nConversation:\n{memory_context}\nFollow-up:\n{question}\nRewritten:"
    response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], temperature=0)
    return response.choices[0].message.content.strip()

def hybrid_search(question, embedding, k=3):
    query = """
    CALL () {
        /* Added () after CALL to fix the deprecation warning */
        CALL db.index.vector.queryNodes('vector', $k, $embedding) YIELD node, score
        RETURN node, (score * 0.7) AS score
        UNION
        CALL db.index.fulltext.queryNodes('ftChunk', $question, {limit:$k}) YIELD node, score
        RETURN node, (score * 0.3) AS score
    }
    WITH node, max(score) AS score ORDER BY score DESC LIMIT $k
    RETURN node.text AS text, score
    """
    records, _, _ = driver.execute_query(query, embedding=embedding, question=question, k=k)
    return records

def generate_answer(question, records):
    context = "\n".join([r["text"] for r in records])
    memory_context = retrieve_memory(question)
    prompt = f"Assistant Rules: Use ONLY context. If missing say 'I don't know'.\n\nMemory: {memory_context}\nContext: {context}\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], temperature=0.3)
    return response.choices[0].message.content

#  FastAPI Routes 

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str
    source: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/chat")
async def chat(req: ChatRequest):
    # Process the message
    standalone = normalize_text(rewrite_query(req.message))
    emb = embed([standalone])[0]
    records = hybrid_search(standalone, emb)
    answer = generate_answer(req.message, records)
    
    # Store memory
    conversation_history.append(f"Q: {req.message} A: {answer}")
    memory_embeddings.append(embed([f"Q: {req.message} A: {answer}"])[0])
    
    # Return keys that script.js expects: 'answer' and 'source_label'
    return {"answer": answer, "source_label": req.source.capitalize()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)