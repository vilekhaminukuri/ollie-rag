import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

# Setup explicit logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OllieRetriever")

load_dotenv()

# System Config Fallbacks — Aligned across ingest, api, and GCP Cloud Run
CHROMA_DB_DIR = Path(os.getenv("CHROMA_PATH", os.getenv("CHROMA_DB_DIR", "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "mtx_docs")

# Must match the model used in ingest.py — different models produce incompatible vector spaces
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")


class MTXRetriever:
    def __init__(self):
        # Local, free, open-source embedding model — no API calls, no per-token cost
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Ensure directory exists safely
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

        # Connect to the persistent local/mounted ChromaDB instance
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        # Safe collection creation/fetch using Cosine distance space
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Connected to ChromaDB collection: '{COLLECTION_NAME}' at '{CHROMA_DB_DIR.resolve()}'")

    def _encode_query(self, query_text: str) -> List[float]:
        """Blocking call: runs the local embedding model. Offloaded to a thread by callers."""
        vector = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vector[0].tolist()

    def _query_collection(self, query_vector: List[float], top_k: int) -> Dict[str, Any]:
        """Blocking call: queries the local ChromaDB collection. Offloaded to a thread by callers."""
        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

    async def query_to_vector(self, query_text: str) -> List[float]:
        """Converts user text into an embedding vector using the local sentence-transformers model."""
        return await asyncio.to_thread(self._encode_query, query_text)

    async def retrieve(self, query_text: str, top_k: int = 4, score_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for similar text chunks. Filters results out if they
        don't meet the similarity score baseline.
        """
        # 1. Convert query to vector space (CPU-bound, run off the event loop)
        query_vector = await self.query_to_vector(query_text)

        # 2. Query ChromaDB (also offloaded — local disk I/O + HNSW search)
        results = await asyncio.to_thread(self._query_collection, query_vector, top_k)

        parsed_results = []

        # Check for empty response payloads
        if not results or not results.get("documents") or len(results["documents"][0]) == 0:
            logger.warning(f"No vector records found for query: '{query_text}'")
            return parsed_results

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # 3. Calculate Cosine similarity (1 - distance) and filter by threshold
        for idx in range(len(documents)):
            similarity_score = 1.0 - distances[idx]

            if similarity_score >= score_threshold:
                parsed_results.append({
                    "content": documents[idx],
                    "metadata": metadatas[idx],
                    "score": round(similarity_score, 4)
                })
            else:
                logger.info(f"Dropped chunk from '{metadatas[idx].get('source')}' due to low score: {similarity_score:.4f} < {score_threshold}")

        return parsed_results

if __name__ == "__main__":
    import asyncio as _asyncio

    async def _dry_run():
        retriever = MTXRetriever()
        test_query = "What holidays are observed in 2026?"
        print(f"\n⚡ Executing dry-run query test for: '{test_query}'")
        matches = await retriever.retrieve(test_query, top_k=3)

        if not matches:
            print("❌ No matching chunks retrieved above the threshold!")
        else:
            for i, match in enumerate(matches):
                print(f"\n[Match #{i+1}] Score: {match['score']} | Source: {match['metadata'].get('source', 'Unknown')}")
                print(f"Content Snippet: {match['content'][:150]}...")

    _asyncio.run(_dry_run())
