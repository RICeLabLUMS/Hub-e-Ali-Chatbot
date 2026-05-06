import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# ==========================================================
# CONFIG & INITIALIZATION
# ==========================================================
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
VECTOR_INDEX_NAME = "vector"  # Match your FastAPI index name
FULLTEXT_INDEX_NAME = "ftChunk"

# Initialize Models and Drivers
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), connection_timeout=30)
hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ==========================================================
# LANGUAGE & NORMALIZATION FUNCTIONS
# ==========================================================

def detect_language_simple(text: str) -> str:
    """Heuristic to detect language if Lingua is not installed."""
    if not text: return "unknown"
    arabic_script_count = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    urdu_specific = len(re.findall(r"[ٹڈڑںےہکگچپژ]", text))

    if arabic_script_count > latin_count:
        return "ur" if urdu_specific > 2 else "ar"
    return "en" if latin_count > 0 else "unknown"

def normalize_for_search(text: str, lang: str) -> str:
    """Aggressive normalization for full-text search indexes."""
    if not text: return ""
    text = text.lower().strip()
    if lang == "ar":
        text = re.sub("[إأآٱا]", "ا", text)
        text = text.replace("ة", "ه").replace("ى", "ي")
    elif lang == "ur":
        text = text.replace("ك", "ک").replace("ي", "ی").replace("ة", "ہ")
    return text

# ==========================================================
# EXTRACTION & CHUNKING FUNCTIONS
# ==========================================================

def read_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extracts text page by page using PyMuPDF."""
    pages = []
    doc = fitz.open(file_path)
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            lang = detect_language_simple(text)
            pages.append({
                "page": i + 1,
                "text": text,
                "language": lang
            })
    return pages
     
def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Simple character-based chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ==========================================================
# NEO4J STORAGE FUNCTIONS
# ==========================================================

def create_indexes():
    """Sets up Vector and Fulltext indexes in Neo4j with explicit session management."""
    try:
        with driver.session() as session:
            # Vector Index[cite: 1]
            session.run(f"""
                CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIM},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
            # Fulltext Index[cite: 1]
            session.run(f"""
                CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS
                FOR (c:Chunk) ON EACH [c.text_normalized, c.title]
            """)
            print("Indexes created or verified successfully.")
    except Exception as e:
        print(f"Failed to create indexes. Connection Error: {e}")

def store_in_neo4j(chunks_data: List[Dict[str, Any]], file_name: str):
    """Batches and stores chunks and embeddings into Neo4j."""
    query = """
    UNWIND $rows AS row
    MERGE (c:Chunk {chunk_id: row.chunk_id})
    SET c.text = row.text,
        c.text_normalized = row.text_normalized,
        c.embedding = row.embedding,
        c.page = row.page,
        c.title = row.title,
        c.language = row.language
    """
    rows = []
    for i, chunk in enumerate(chunks_data):
        # Generate embedding
        emb = hf_embeddings.embed_query(chunk["text"])
        
        rows.append({
            "chunk_id": f"{hashlib.md5(file_name.encode()).hexdigest()}_{i}",
            "text": chunk["text"],
            "text_normalized": normalize_for_search(chunk["text"], chunk["language"]),
            "embedding": emb,
            "page": chunk["page"],
            "title": file_name,
            "language": chunk["language"]
        })

    with driver.session() as session:
        session.run(query, rows=rows)

# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main(folder_path: str):
    create_indexes() # Remove from here
    path = Path(folder_path)
    
    for pdf_file in path.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        pages = read_pdf(str(pdf_file))
        
        all_chunks = []
        for p in pages:
            text_chunks = chunk_text(p["text"])
            for tc in text_chunks:
                all_chunks.append({
                    "text": tc,
                    "page": p["page"],
                    "language": p["language"]
                })
        
        store_in_neo4j(all_chunks, pdf_file.name)
        print(f"Finished {pdf_file.name}")

if __name__ == "__main__":
    # Ensure you have a folder named 'data' with your PDFs
    if not os.path.exists("data"):
        os.makedirs("data")
    main("data")



# from neo4j import GraphDatabase
# from dotenv import load_dotenv
# load_dotenv()
# import os

# uri = os.getenv('NEO4J_URL')
# user = os.getenv('NEO4J_USERNAME')
# password = os.getenv('NEO4J_PASSWORD')
# # print(uri)
# # print(user)
# # print(password)

# driver = GraphDatabase.driver(uri, auth=(user, password))

# with driver.session() as session:
#     result = session.run("RETURN 1")
#     print(result.single())