import os
from dotenv import load_dotenv


import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from preprocess import load_vectorstore, embed_text, answer_question

load_dotenv()
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace ["*"] with specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the FAISS vectorstore once on startup
vectorstore = load_vectorstore()

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded string

class LinkItem(BaseModel):
    url: str
    text: str

class QuestionResponse(BaseModel):
    answer: str
    links: List[LinkItem]



@app.post("/api/", response_model=QuestionResponse)
async def ask_question(data: QuestionRequest):
    try:
        answer, sources = answer_question(data.question, vectorstore)

        # sources is a list of dicts with keys 'url' and 'text'
        links = list(sources)

        return JSONResponse(
            content={
                "answer": answer,
                "links": links
            },
            media_type="application/json"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/")
def home():
    return {"message": "TDS Virtual TA is running!"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Railway provides dynamic PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)

  