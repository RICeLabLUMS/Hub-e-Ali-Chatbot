# multilingual normalization approach 



import os
import re
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.documents import Document
from dotenv import load_dotenv
from neo4j import GraphDatabase


# pdf_path = 'AlKafiV1.pdf'
pdf_path = 'books/AlKafiV1Part2.pdf'
def normalize_text(text):
    """Normalizes Arabic and Urdu text characters."""
    text = text.strip()
    if re.search(r'[\u0600-\u06FF]', text):
        # Normalize Arabic
        text = re.sub("[ًٌٍَُِّْـ]", "", text)
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        # Normalize Urdu
        text = text.replace("ك", "ک").replace("ی", "ی")
    return text.lower()

def run_setup():
    load_dotenv()
    print("Pdf Loaded Successfully!")
    hf_token = os.getenv("HF_TOKEN")
    
    # Initialize Multilingual Embeddings
    print("Downloading/Loading Embedding Model (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu', 'token': hf_token},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 1. Read and Normalize PDF
    print("Reading PDF...")
    reader = PdfReader(pdf_path)
    docs = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            normalized_content = normalize_text(content)
            docs.append(Document(page_content=normalized_content))
    print(f"Total pages parsed: {len(docs)}")
    print("Storing in Neo4j (Vectorizing)...")        
    # 2. Store in Neo4j and Create Indices[cite: 1, 2]
    db = Neo4jVector.from_documents(
        docs, embeddings,
        url=os.getenv('NEO4J_URL'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD'),
        database="28efd2d0",
        index_name="vector",
        node_label="Chunk", # Changed to 'Chunk' to match memory notebook
        text_node_property="text",
        embedding_node_property="embedding"
    )

    # 3. Create Full-text Index for Hybrid Search[cite: 1]
    driver = GraphDatabase.driver(os.getenv('NEO4J_URL'), 
                                 auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
    with driver.session() as session:
        session.run("CREATE FULLTEXT INDEX ftChunk IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]")
    driver.close()
    print("Data Stored in Vector DB Successfully!")

if __name__ == "__main__":
    run_setup()