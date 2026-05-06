import os
from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

class Config:
    NEO4J_URI = os.getenv("NEO4J_URL")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    GENERATION_MODEL = "llama-3.3-70b-versatile"
    VECTOR_INDEX_NAME = "vector"
    FULLTEXT_INDEX_NAME = "ftChunk"

# Initialize Shared Clients
client = Groq(api_key=Config.GROQ_API_KEY)
driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD))
hf_embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)