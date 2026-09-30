"""Environment setup and project constants."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Environment variables (never hard-code credentials) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# --- Source document ---
PDF_FILE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"
PDF_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={PDF_FILE_ID}"
PDF_PATH = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"

# --- Chunking (assignment: 500-1000 chars, ~100 overlap) ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- Models ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
LLM_MODEL = "openai/gpt-oss-20b"

# --- Pinecone ---
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"          # serverless placement, required by Pinecone's API
PINECONE_REGION = "us-east-1"   # free-tier region

# --- Retrieval / generation ---
TOP_K = 3
REFUSAL_MESSAGE = "I cannot answer based on the provided document."


def require_env() -> None:
    """Fail early with a clear message if credentials are missing."""
    missing = [
        name
        for name, value in [
            ("GROQ_API_KEY", GROQ_API_KEY),
            ("PINECONE_API_KEY", PINECONE_API_KEY),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
