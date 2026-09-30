"""LangGraph RAG workflow: START -> retrieve -> generate -> END."""
from typing import List, TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from src import config


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float
    # Extra bookkeeping used to compute the score / for debugging
    similarity_scores: List[float]
    pages: List[int]


SYSTEM_PROMPT = f"""You are a strict assistant. Answer the user's question relying ONLY on the context provided.
Do not use any outside or general knowledge, even if you know the answer.
If the context does not contain enough information to answer, reply exactly:
'{config.REFUSAL_MESSAGE}'"""


def _is_refusal(answer: str) -> bool:
    return config.REFUSAL_MESSAGE.lower().rstrip(".") in answer.lower()


def build_rag_graph(index_name: str = config.PINECONE_INDEX_NAME):
    config.require_env()
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)

    def retrieve_node(state: AgentState):
        """Embed the question and fetch the top-k most similar chunks from Pinecone."""
        results = vectorstore.similarity_search_with_score(state["question"], k=config.TOP_K)
        return {
            "context": [doc.page_content for doc, _ in results],
            "similarity_scores": [float(score) for _, score in results],
            "pages": [int(doc.metadata.get("page", 0)) for doc, _ in results],
        }

    def generate_node(state: AgentState):
        """Answer strictly from the retrieved context and compute the score."""
        if not state["context"]:
            return {"answer": config.REFUSAL_MESSAGE, "score": 0.0}

        context_str = "\n\n---\n\n".join(state["context"])
        response = llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", f"Context:\n{context_str}\n\nQuestion: {state['question']}"),
            ]
        )
        answer = response.content.strip()

        # Score = mean cosine similarity of retrieved chunks (0 if the model refused).
        # This measures retrieval relevance, NOT answer correctness.
        if _is_refusal(answer):
            score = 0.0
        else:
            sims = state["similarity_scores"]
            score = round(max(0.0, min(1.0, sum(sims) / len(sims))), 2) if sims else 0.0

        return {"answer": answer, "score": score}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
