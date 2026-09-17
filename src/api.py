import os
import json
import time
import secrets
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from src.retriever import MTXRetriever

# Setup explicit structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("OllieAPI")

# Shared-secret gate for /api/chat — keeps the endpoint itself locked even when
# the Cloud Run service is deployed with --allow-unauthenticated (required so
# non-GCP callers like Salesforce Flow can still reach it).
EXPECTED_API_KEY = os.getenv("OLLIE_API_KEY")
if not EXPECTED_API_KEY:
    logger.warning("[SECURITY] OLLIE_API_KEY is not set — /api/chat is running WITHOUT API key protection.")

async def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if EXPECTED_API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, EXPECTED_API_KEY)):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

# Global track for health check uptime calculation
START_TIME = time.time()

app = FastAPI(
    title="MTX Connect Chatbot API",
    description="Production-ready FastAPI engine powered by OpenAI and ChromaDB RAG architecture.",
    version="1.2.0"
)

# Enable CORS for Salesforce and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LATENCY TRACKING MIDDLEWARE ---
@app.middleware("http")
async def log_requests_latency(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Exclude routine health check polling from verbose logs
    if request.url.path != "/health":
        logger.info(f"Path: {request.url.path} | Status: {response.status_code} | Latency: {process_time_ms}ms")

    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
    return response


# Global Client/Retriever Instantiations
# Repointed at Groq's OpenAI-compatible endpoint to use a free open-weight model
openai_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
retriever_instance = MTXRetriever()

# Canned reply for pure greetings — skips retrieval AND the LLM call entirely
GREETING_RESPONSE = (
    "Hello! I'm Ollie, your MTX Connect assistant. Ask me about company policies, "
    "holidays, PTO, referrals, or other internal procedures, and I'll do my best to help!"
)


# --- PYDANTIC SCHEMAS (UPDATED FOR SALESFORCE EXTERNAL SERVICES) ---
class Message(BaseModel):
    role: str = Field(..., description="Role of the sender: 'user' or 'assistant'.")
    content: str = Field(..., description="The raw textual message payload body.")

class ChatRequest(BaseModel):
    messages: Optional[List[Message]] = Field(
        default=None,
        description="Full historical log trace of conversation (used by web UI)."
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Single prompt or query string (ideal for direct Salesforce Flow callouts)."
    )
    stream: bool = Field(default=False, description="Whether to stream response tokens natively.")

class SourceContext(BaseModel):
    source: str = Field(..., description="Source document file name.")
    title: str = Field(..., description="Formatted title of the source document.")
    chunk_index: int = Field(..., description="Chunk or fragment index.")
    score: float = Field(..., description="Similarity relevance score.")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The definitive text response crafted by Ollie.")
    sources: List[str] = Field(default=[], description="Unique list of source filenames for frontend UI badges.")
    context: List[SourceContext] = Field(default=[], description="Full metadata and scores of retrieved fragments.")


# --- HELPER FUNCTIONS ---
def is_pure_greeting(query: str) -> bool:
    """Accurately checks if user query is strictly a greeting without an actual question attached."""
    greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "yo", "sup", "help"}
    words = query.lower().strip().replace("?", "").replace("!", "").split()

    # Only treat as greeting if query is 1-2 words and matches known greetings
    return len(words) <= 2 and any(w in greetings for w in words)


def build_system_prompt(contexts: List[Dict[str, Any]]) -> str:
    current_date_str = datetime.now().strftime("%A, %B %d, %Y")
    time_anchor = f"CRITICAL TEMPORAL CONTEXT: The current actual date is exactly {current_date_str}.\n\n"

    # NO CONTEXT / NEGATIVE CASE FALLBACK
    if not contexts:
        return (
            f"{time_anchor}"
            "You are Ollie, an expert corporate support specialist for MTX Connect.\n"
            "- If the user provides a simple greeting (e.g., 'hello', 'hi'), greet them back warmly and ask how you can help with MTX Connect policies.\n"
            "- If the user asks about an MTX Connect policy, holiday, benefit, or procedure, state strictly: "
            "'I cannot find that information within the verified MTX Connect documentation repository right now. Please verify the relevant policy document has been uploaded.'\n"
            "- If the user asks about unrelated topics or other companies, politely state that you can only assist with MTX Connect internal topics."
        )

    # POSITIVE CASE WITH RETRIEVED CONTEXT
    context_str = ""
    for idx, ctx in enumerate(contexts):
        source_name = ctx.get("metadata", {}).get("source", "Unknown Document")
        context_str += f"\n--- Document Source Fragment #{idx+1} ({source_name}) ---\n"
        context_str += f"{ctx['content']}\n"

    return (
        f"{time_anchor}"
        "You are Ollie, an expert corporate support specialist for MTX Connect.\n"
        "Answer the user's question using ONLY the factual context provided below.\n"
        "- Do not extrapolate, assume, or hallucinate outside the given text.\n"
        "- If the query asks about details or organizations not present in the context below, clearly state that the documentation does not contain that detail.\n"
        "- Do not cite document source names inside the text body of your response; sources will be displayed automatically in the UI badges.\n\n"
        f"--- START MTX CONTEXT DATA ---{context_str}--- END MTX CONTEXT DATA ---"
    )


def compute_source_data(
    matched_contexts: List[Dict[str, Any]], generated_answer: str
) -> Tuple[List[SourceContext], set]:
    """Builds the full SourceContext list plus the filtered set of sources allowed to badge in the UI."""
    source_responses: List[SourceContext] = []
    unique_sources = set()

    answer_lower = generated_answer.lower()
    fallback_keywords = [
        "does not contain", "cannot find", "do not have",
        "no details", "only provide information regarding mtx",
        "only assist with mtx", "documentation repository"
    ]
    is_fallback = any(kw in answer_lower for kw in fallback_keywords)

    # Minimum score required for a document badge to render in the UI
    SOURCE_BADGE_SCORE_CUTOFF = 0.45

    for item in matched_contexts:
        meta = item.get("metadata", {})
        src_file = meta.get("source", "Unknown")
        item_score = item.get("score", 0.0)

        source_responses.append(
            SourceContext(
                source=src_file,
                title=meta.get("title", src_file.replace("_", " ").title()),
                chunk_index=meta.get("chunk_index", 0),
                score=item_score
            )
        )
        # Only display badges if NOT a fallback response AND the retrieval score is high enough
        if src_file != "Unknown" and not is_fallback and item_score >= SOURCE_BADGE_SCORE_CUTOFF:
            unique_sources.add(src_file)

    return source_responses, unique_sources


def sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- API ROUTE ENDPOINTS ---
@app.get(
    "/health",
    tags=["System Maintenance"],
    operation_id="getHealthStatus",
    summary="Uptime and database status check"
)
def health_check():
    """Uptime monitoring and database status check."""
    uptime_seconds = round(time.time() - START_TIME, 2)
    db_connected = retriever_instance.collection is not None

    return {
        "status": "healthy" if db_connected else "degraded",
        "uptime_seconds": uptime_seconds,
        "database_connected": db_connected,
        "version": app.version
    }


@app.post(
    "/api/chat",
    response_model=None,
    tags=["Core AI Operations"],
    operation_id="chatInteraction",
    summary="Main RAG Chat Route",
    dependencies=[Depends(verify_api_key)]
)
async def chat_interaction(payload: ChatRequest):
    """
    Main RAG Chat Route with complete edge case handling, logging, and source extraction.
    Supports both message traces (Streamlit) and simple prompt strings (Salesforce).
    When payload.stream is True, responds with a text/event-stream of `token` events
    followed by a final `done` event carrying sources/context (or an `error` event on failure).
    """
    # 1. Normalize input to support both payload.messages and payload.prompt
    messages_history: List[Message] = []
    query_text = ""

    if payload.messages and len(payload.messages) > 0:
        messages_history = payload.messages
        last_user_message = messages_history[-1]
        if last_user_message.role != "user":
            raise HTTPException(status_code=400, detail="The terminating payload sequence role must be assigned to 'user'.")
        query_text = last_user_message.content.strip()
    elif payload.prompt and payload.prompt.strip():
        query_text = payload.prompt.strip()
        messages_history = [Message(role="user", content=query_text)]
    else:
        raise HTTPException(
            status_code=400,
            detail="Payload must contain either a non-empty 'messages' array or a 'prompt' string."
        )

    logger.info(f"[USER QUERY]: '{query_text}'")

    # 2. Pure greeting short-circuit — bypasses vector retrieval AND the LLM entirely
    if is_pure_greeting(query_text):
        logger.info("[ROUTING]: Pure greeting detected. Returning canned response, bypassing retrieval and LLM.")
        if payload.stream:
            async def greeting_stream():
                yield sse_event("token", {"content": GREETING_RESPONSE})
                yield sse_event("done", {"sources": [], "context": []})
            return StreamingResponse(greeting_stream(), media_type="text/event-stream")
        return ChatResponse(answer=GREETING_RESPONSE, sources=[], context=[])

    # 3. RAG Query: Score threshold for ChromaDB candidate retrieval
    matched_contexts = []
    try:
        matched_contexts = await retriever_instance.retrieve(query_text, top_k=3, score_threshold=0.25)

        if matched_contexts:
            top_score = matched_contexts[0].get("score", 0.0)
            extracted_sources = list(set([c.get("metadata", {}).get("source", "Unknown") for c in matched_contexts]))
            logger.info(f"[RETRIEVAL SUCCESS] Matches: {len(matched_contexts)} | Top Score: {top_score} | Sources: {extracted_sources}")
        else:
            logger.warning(f"[KNOWLEDGE GAP DETECTED] No contexts found above score threshold for query: '{query_text}'")

    except Exception as e:
        logger.error(f"[RETRIEVAL FAULT] ChromaDB retrieval system failure: {str(e)}")
        matched_contexts = []

    # 4. Construct System Prompt Injector + complete API payload format for OpenAI
    system_instruction = build_system_prompt(matched_contexts)
    openai_messages = [{"role": "system", "content": system_instruction}]
    for msg in messages_history:
        openai_messages.append({"role": msg.role, "content": msg.content})

    # 5a. Streaming Model Generation
    if payload.stream:
        async def event_generator():
            full_answer = ""
            try:
                completion_stream = await openai_client.chat.completions.create(
                    model="openai/gpt-oss-20b",  # free open-weight model hosted on Groq
                    messages=openai_messages,
                    temperature=0.2,
                    max_tokens=800,
                    stream=True,
                )
                async for chunk in completion_stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_answer += delta
                        yield sse_event("token", {"content": delta})

            except Exception as e:
                logger.error(f"[LLM FAULT] OpenAI Chat Generation internal API failure: {str(e)}")
                yield sse_event("error", {"detail": "AI Generation pipeline engine encountered a critical error."})
                return

            source_responses, unique_sources = compute_source_data(matched_contexts, full_answer)
            yield sse_event("done", {
                "sources": list(unique_sources),
                "context": [s.model_dump() for s in source_responses]
            })

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 5b. Non-streaming Model Generation (Salesforce / direct API callers)
    try:
        response = await openai_client.chat.completions.create(
            model="openai/gpt-oss-20b",  # free open-weight model hosted on Groq
            messages=openai_messages,
            temperature=0.2,
            max_tokens=800
        )
        generated_answer = response.choices[0].message.content

        source_responses, unique_sources = compute_source_data(matched_contexts, generated_answer)

        return ChatResponse(
            answer=generated_answer,
            sources=list(unique_sources),
            context=source_responses
        )

    except Exception as e:
        logger.error(f"[LLM FAULT] OpenAI Chat Generation internal API failure: {str(e)}")
        raise HTTPException(status_code=500, detail="AI Generation pipeline engine encountered a critical error.")
