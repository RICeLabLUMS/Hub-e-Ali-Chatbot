import re
import json
from typing import List, Dict, Any
from app.core.config import client, driver, hf_embeddings, Config

# Global Memory (Ideally move to a DB later)
conversation_history: List[str] = []

def detect_language(text: str) -> str:
    """Heuristic language detection."""
    if not text: return "en"
    arabic_script_count = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    urdu_specific = len(re.findall(r"[ٹڈڑںےہکگچپژ]", text))
    
    if arabic_script_count > latin_count:
        return "ur" if urdu_specific > 2 else "ar"
    return "en"

def normalize_text_by_lang(text: str, lang: str) -> str:
    """Language-specific normalization for search."""
    text = text.lower().strip()
    if lang == "ar":
        text = re.sub("[إأآٱا]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    elif lang == "ur":
        text = text.replace("ك", "ک").replace("ي", "ی").replace("ة", "ہ")
    return text

def expand_query_llm(query: str) -> List[str]:
    """Generates multilingual variants of the query using LLM."""
    prompt = f"Generate 3 search variants (English, Urdu, Arabic) for: {query}. Return ONLY a JSON list: {{\"queries\": []}}"
    try:
        response = client.chat.completions.create(model=Config.GENERATION_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
        data = json.loads(response.choices[0].message.content)
        return data.get("queries", [query])
    except:
        return [query]

def hybrid_search(question: str, embedding: List[float], k: int = 5):
    """Combines Vector and Fulltext search in Neo4j."""
    query = f"""
    CALL () {{
        CALL db.index.vector.queryNodes('{Config.VECTOR_INDEX_NAME}', $k, $embedding) YIELD node, score
        RETURN node, (score * 0.7) AS score
        UNION
        CALL db.index.fulltext.queryNodes('{Config.FULLTEXT_INDEX_NAME}', $question, {{limit: $k}}) YIELD node, score
        RETURN node, (score * 0.3) AS score
    }}
    WITH node, max(score) AS score ORDER BY score DESC LIMIT $k
    RETURN node, score
    """
    records, _, _ = driver.execute_query(query, embedding=embedding, question=question, k=k)
    return records

def retrieve(question: str, top_k: int = 5):
    """Orchestrates expansion and hybrid search."""
    variants = expand_query_llm(question)
    all_results = []
    for q in variants:
        q_lang = detect_language(q)
        q_norm = normalize_text_by_lang(q, q_lang)
        q_emb = hf_embeddings.embed_query(q)
        all_results.extend(hybrid_search(q_norm, q_emb, k=top_k))
    
    # Deduplicate by chunk_id
    unique = {r["node"]["chunk_id"]: r for r in all_results if "chunk_id" in r["node"]}
    return sorted(unique.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def rewrite_query(question: str) -> str:
    """Uses history to make follow-up questions standalone."""
    if not conversation_history: return question
    context = conversation_history[-1]
    prompt = f"Rewrite this question to be a standalone search query based on context: {context}\nQuestion: {question}"
    response = client.chat.completions.create(model=Config.GENERATION_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    return response.choices[0].message.content.strip()


def generate_answer(question: str, records: List[Dict[str, Any]]) -> str:
    """Generates the final response using retrieved context."""
    context_text = "\n".join([f"Source: {r['node']['title']}, Page: {r['node']['page']}\nText: {r['node']['text']}" for r in records])
    lang = detect_language(question)
    prompt = f"Answer in {lang}. Use ONLY the context below. If not found, say you don't know.\nContext: {context_text}\nQuestion: {question}"
    response = client.chat.completions.create(model=Config.GENERATION_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2)
    return response.choices[0].message.content

def store_memory(question: str, answer: str):
    """Stores the exchange for conversation context."""
    text = f"Q: {question} A: {answer}"
    conversation_history.append(text)
    if len(conversation_history) > 5: conversation_history.pop(0)
