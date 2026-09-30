"""FastAPI interface.  Run: uvicorn app:app --reload"""
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config
from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")
graph = build_rag_graph(index_name=config.PINECONE_INDEX_NAME)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0,
        "similarity_scores": [],
        "pages": [],
    }
    result = graph.invoke(initial_state)

    return QueryResponse(
        query=request.query,
        final_answer=result["answer"],
        retrieved_context_chunks=result["context"],
        confidence_score=result["score"],
    )
