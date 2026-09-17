# Ollie — MTX Connect RAG Chatbot

Ollie is a retrieval-augmented generation (RAG) chatbot that answers questions about MTX
internal policies (PTO, holidays, attendance, key control, referrals, separation, video
meetings, signatory authority) by grounding every answer in the company's own policy
documents.

The stack runs entirely on free-tier infrastructure:

- **Embeddings**: local, open-source [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
  via `sentence-transformers` — no API calls, no per-token cost.
- **Chat generation**: [Groq](https://groq.com)'s free-tier API running the open-weight
  `openai/gpt-oss-20b` model.
- **Vector store**: [ChromaDB](https://www.trychroma.com/), persisted to local disk.

## Architecture

```
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────┐
│  Streamlit UI    │ ───► │  FastAPI backend      │ ───► │  Groq API        │
│  (src/app.py)     │      │  (src/api.py)         │      │  (chat generation)│
└─────────────────┘      └──────────┬───────────┘      └─────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  MTXRetriever          │
                          │  (src/retriever.py)    │
                          │  bge-small embeddings  │
                          │  + ChromaDB             │
                          └──────────────────────┘
```

Documents are ingested once (offline) via `src/ingest.py`, which chunks raw policy files,
embeds them locally, and stores the vectors in ChromaDB. At query time, `src/retriever.py`
embeds the user's question with the same model and retrieves the closest matching chunks,
which are injected into the prompt sent to Groq.

## Prerequisites

- Docker & Docker Compose
- A free [Groq API key](https://console.groq.com/keys)

## Setup

1. **Clone and configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set:

   | Variable | Description |
   |---|---|
   | `GROQ_API_KEY` | Your Groq API key (chat generation) |
   | `OLLIE_API_KEY` | Shared secret required in the `X-API-Key` header on `/api/chat`. Set any random string; leave unset to disable the check locally. |
   | `CHROMA_DB_DIR` | Where the vector database is persisted (default `data/processed/chroma_db`) |
   | `RAW_DATA_DIR` | Where source policy documents live (default `data/raw`) |

2. **Add source documents**

   Drop `.pdf`, `.docx`, or `.txt` policy files into `data/raw/` (a starter set of MTX
   policy PDFs is already included).

3. **Build and start the stack**

   ```bash
   docker compose up -d --build
   ```

   This starts:
   - `ollie_backend` — FastAPI RAG API on `http://localhost:8000`
   - `ollie_frontend` — Streamlit chat UI on `http://localhost:8501`

4. **Ingest the documents**

   Run this once (and again any time documents in `data/raw/` change, or after switching
   embedding models — different models produce incompatible vector dimensions):

   ```bash
   docker compose exec backend python -m src.ingest
   ```

5. **Verify**

   ```bash
   curl http://localhost:8000/health
   ```

   Should return `"database_connected": true`. Then open `http://localhost:8501` and start
   chatting.

## API

### `GET /health`
Returns service uptime and database connectivity status.

### `POST /api/chat`
Main RAG endpoint. Requires header `X-API-Key: <OLLIE_API_KEY>` if that variable is set.

Request body (either `messages` or `prompt`):

```json
{
  "messages": [{"role": "user", "content": "How many PTO days do I get?"}],
  "stream": false
}
```

or, for single-shot callers (e.g. Salesforce Flow):

```json
{
  "prompt": "How many PTO days do I get?",
  "stream": false
}
```

Set `"stream": true` to receive a `text/event-stream` of `token` events followed by a
final `done` event carrying `sources` and `context`.

Response (non-streaming):

```json
{
  "answer": "US employees accrue up to 15 days of PTO per year...",
  "sources": ["MTX US Holiday & Paid Time Off (PTO).pdf"],
  "context": [
    {"source": "...", "title": "...", "chunk_index": 2, "score": 0.83}
  ]
}
```

## Local development (without Docker)

```bash
pip install -r requirements.txt
python -m src.ingest          # build the vector index
uvicorn src.api:app --reload  # backend on :8000
streamlit run src/app.py      # frontend on :8501
```

## Notes

- `torch` is pinned to the CPU-only wheel (`torch==2.14.0+cpu` via
  `--extra-index-url https://download.pytorch.org/whl/cpu`) to avoid pulling in several GB
  of unused CUDA libraries — this backend only ever runs embeddings on CPU.
- The vector database (`data/processed/chroma_db/`) and `.env` are gitignored; re-run
  ingestion after a fresh clone.
