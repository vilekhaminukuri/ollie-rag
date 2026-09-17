import os
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from docx import Document
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

from sentence_transformers import SentenceTransformer

# Setup simple console logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OllieIngest")

load_dotenv()

# Configuration Fallbacks
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "raw"))
CHROMA_DB_DIR = Path(os.getenv("CHROMA_PATH", os.getenv("CHROMA_DB_DIR", "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "mtx_docs")

# Local, free, open-source embedding model (no API calls, no per-token cost)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


class IngestionPipeline:
    def __init__(self):
        logger.info(f"Loading local embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads weights, then cached)...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text(self, file_path: Path) -> str:
        """Parses PDF, DOCX, and TXT files cleanly returning extracted text strings."""
        ext = file_path.suffix.lower()
        text = ""

        try:
            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            elif ext == ".pdf":
                reader = PdfReader(file_path)
                text = "".join([page.extract_text() or "" for page in reader.pages])
            elif ext == ".docx":
                doc = Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
            else:
                logger.warning(f"Unsupported file format dropped: {file_path.name}")
        except Exception as e:
            logger.error(f"Error extracting from file {file_path.name}: {str(e)}")

        return text.strip()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings locally via a free, open-source sentence-transformers model."""
        vectors = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vectors.tolist()

    def run(self):
        """Discovers, transforms, generates vectors, and stores data in ChromaDB."""
        supported_extensions = {".txt", ".pdf", ".docx"}
        files_to_process = [p for p in RAW_DATA_DIR.iterdir() if p.suffix.lower() in supported_extensions]

        if not files_to_process:
            logger.info(f"No documents discovered inside '{RAW_DATA_DIR.resolve()}'. Please drop source documentation here.")
            return

        logger.info(f"Discovered {len(files_to_process)} document(s) to ingest using local embedding pipeline.")

        total_chunks = 0

        for file_path in files_to_process:
            logger.info(f"Processing: {file_path.name}")
            raw_text = self.extract_text(file_path)

            if not raw_text:
                logger.warning(f"Skipping empty text payload extracted from: {file_path.name}")
                continue

            chunks = self.text_splitter.split_text(raw_text)
            num_chunks = len(chunks)
            logger.info(f"Split structural text into {num_chunks} chunks.")

            if num_chunks == 0:
                continue

            batch_size = 32
            embeddings: List[List[float]] = []

            for i in range(0, num_chunks, batch_size):
                batch_texts = chunks[i:i + batch_size]
                batch_vectors = self.get_embeddings(batch_texts)
                embeddings.extend(batch_vectors)

            ids = [f"{file_path.stem}_chunk_{idx}" for idx in range(num_chunks)]
            metadatas = [
                {
                    "source": file_path.name,
                    "title": file_path.stem.replace("_", " ").title(),
                    "chunk_index": idx
                } for idx in range(num_chunks)
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=chunks
            )

            total_chunks += num_chunks
            logger.info(f"Successfully indexed {file_path.name}")

        print("\n" + "="*50)
        print("     LOCAL EMBEDDING INGESTION PIPELINE COMPLETE")
        print("="*50)
        print(f" Embedding model:           {EMBEDDING_MODEL_NAME}")
        print(f" Total files indexed:       {len(files_to_process)}")
        print(f" Total chunks produced:     {total_chunks}")
        print(f" Collection Name:           {COLLECTION_NAME}")
        print(f" Database Destination:      {CHROMA_DB_DIR.resolve()}")
        print("="*50 + "\n")


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
