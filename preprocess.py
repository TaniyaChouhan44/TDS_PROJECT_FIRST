import os
import json
import logging
from dotenv import load_dotenv
import httpx

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# === Load environment ===
load_dotenv()

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === AI Proxy Configuration ===
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set in the environment.")

AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
HEADERS = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
EMBEDDING_URL = f"{AIPROXY_BASE_URL}/v1/embeddings"
CHAT_URL = f"{AIPROXY_BASE_URL}/v1/chat/completions"

# === Helper: POST request with error handling ===
def safe_post(url: str, headers: dict, payload: dict) -> dict:
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"HTTP error: {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Request error: {str(e)}")
        raise RuntimeError("Embedding or chat request failed.")

# === Function: Embed Text ===
def embed_text(texts: list[str]) -> list[list[float]]:
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }
    data = safe_post(EMBEDDING_URL, HEADERS, payload)
    return [item["embedding"] for item in data["data"]]

# === Function: Load Vectorstore ===
def load_vectorstore(path: str = "vectorstore"):
    dummy_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=AIPROXY_TOKEN,
        base_url=f"{AIPROXY_BASE_URL}/v1"
    )
    return FAISS.load_local(path, dummy_embeddings, allow_dangerous_deserialization=True)

# === Function: Build Prompt ===
def build_prompt(context: str) -> str:
    return (
        "You are a helpful Virtual TA for the Tools for Data Science (TDS) course. "
        "Answer clearly based on the following context:\n\n"
        f"{context}\n\n"
        "If the context contains source links, return a JSON like:\n"
        "{\n  \"answer\": \"...\",\n  \"links\": [\n    {\"url\": \"...\", \"text\": \"...\"}, ...\n  ]\n}\n"
        "Otherwise, just return a plain text answer."
    )

# === Function: Answer Question ===
def answer_question(question: str, vectorstore, k: int = 5) -> tuple[str, list[dict]]:
    question_vector = embed_text([question])[0]
    docs = vectorstore.similarity_search_by_vector(question_vector, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": build_prompt(context)},
            {"role": "user", "content": question}
        ]
    }

    result = safe_post(CHAT_URL, HEADERS, payload)
    raw_answer = result["choices"][0]["message"]["content"].strip()

    try:
        parsed = json.loads(raw_answer)
        answer = parsed["answer"]
        links = parsed.get("links", [])
    except (json.JSONDecodeError, KeyError, TypeError):
        answer = raw_answer
        links = []
        for doc in docs:
            url = doc.metadata.get("source", "Unknown")
            snippet = doc.page_content.strip().split("\n")[0][:300]
            links.append({"url": url, "text": snippet})

    return answer, links
