"""PDF download -> load -> chunk -> embed -> Pinecone index.

Run once:  python -m src.ingestion
"""
import time

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config


def download_pdf() -> None:
    """Download the eBook to data/Ebook-Agentic-AI.pdf (skipped if present)."""
    if config.PDF_PATH.exists():
        print(f"[1/5] PDF already present: {config.PDF_PATH}")
        return
    print("[1/5] Downloading PDF from Google Drive...")
    config.PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(config.PDF_DOWNLOAD_URL, timeout=60)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise RuntimeError(
            "Download did not return a PDF (Google Drive may have blocked it). "
            f"Download it manually and save it as {config.PDF_PATH}"
        )
    config.PDF_PATH.write_bytes(resp.content)


def ensure_index(pc: Pinecone, index_name: str) -> None:
    """Create the Pinecone index (1536 dims, cosine) if it does not exist."""
    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        print(f"[4/5] Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=config.EMBEDDING_DIMENSION,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
    else:
        print(f"[4/5] Pinecone index '{index_name}' already exists.")


def run_ingestion(pdf_path: str = str(config.PDF_PATH), index_name: str = config.PINECONE_INDEX_NAME):
    config.require_env()
    download_pdf()

    # 2. Load pages (one Document per page; metadata has 'page' 0-indexed)
    print("[2/5] Loading PDF pages...")
    docs = PyPDFLoader(pdf_path).load()
    for d in docs:
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1  # human-friendly page number
        d.metadata["source"] = "Ebook-Agentic-AI.pdf"
    print(f"      Loaded {len(docs)} pages.")

    # 3. Chunk
    print("[3/5] Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"      Created {len(chunks)} chunks.")

    # 4. Index
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc, index_name)

    # 5. Embed + upsert. Chunk text is stored as the 'text' field; page in metadata.
    # Deterministic IDs make re-running ingestion overwrite instead of duplicate.
    print("[5/5] Embedding and upserting to Pinecone...")
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        ids=[f"chunk-{i}" for i in range(len(chunks))],
    )
    print("Done. Ingestion complete.")
    return vector_store


if __name__ == "__main__":
    run_ingestion()
