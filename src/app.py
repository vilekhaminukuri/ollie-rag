import os
import json
import streamlit as st
import requests

# 1. Clean Layout Setup
st.set_page_config(
    page_title="Ollie AI",
    page_icon="⚡",
    layout="centered"
)

# 2. Immersive High-Contrast CSS Patch
st.markdown("""
    <style>
        html, body, .stApp, [data-testid="stHeader"], [data-testid="stFooter"] {
            background-color: #0F172A !important;
            color: #F8FAFC !important;
        }
        
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        .app-title {
            font-size: 2.8rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 5px;
            color: #FFFFFF;
        }
        
        .app-caption {
            font-size: 1.15rem;
            text-align: center;
            color: #94A3B8;
            margin-bottom: 2.5rem;
        }

        .welcome-box {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 2rem;
            color: #E2E8F0;
            line-height: 1.6;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        
        .source-container {
            margin-top: 14px;
            padding: 12px;
            background-color: #0F172A;
            border-radius: 8px;
            border: 1px solid #334155;
        }
        
        .source-tag {
            background-color: #1E293B;
            color: #38BDF8;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin: 4px;
            border: 1px solid #334155;
        }
        
        div[data-testid="stChatMessage"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
            padding: 18px !important;
        }
        
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] ul,
        div[data-testid="stChatMessage"] ol,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] h1,
        div[data-testid="stChatMessage"] h2,
        div[data-testid="stChatMessage"] h3,
        div[data-testid="stChatMessage"] h4,
        div[data-testid="stChatMessage"] h5,
        div[data-testid="stChatMessage"] h6 {
            color: #F8FAFC !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
        }

        div[data-testid="stChatInput"] {
            padding: 0px !important;
        }

        div[data-testid="stChatInput"] textarea {
            background-color: #1E293B !important; 
            color: #FFFFFF !important;            
            border: 1px solid #475569 !important; 
            border-radius: 10px !important;
            padding: 14px !important;
            font-size: 1.05rem !important;
            outline: none !important;
            box-shadow: none !important;
        }
        
        div[data-testid="stChatInput"] textarea:focus,
        div[data-testid="stChatInput"] textarea:active,
        div[data-testid="stChatInput"] > div:focus-within {
            border-color: #38BDF8 !important; 
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
        }
        
        div[data-testid="stChatInput"] > div {
            background-color: #1E293B !important;
            border: 1px solid #475569 !important;
            border-radius: 10px !important;
        }
        
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #64748B !important;
        }
    </style>
""", unsafe_allow_html=True)

# Resolves backend URL dynamically
raw_backend_url = os.getenv("BACKEND_URL", os.getenv("FASTAPI_URL", "http://127.0.0.1:8000"))
if not raw_backend_url.endswith("/api/chat"):
    FASTAPI_URL = f"{raw_backend_url.rstrip('/')}/api/chat"
else:
    FASTAPI_URL = raw_backend_url

# Shared secret expected by the backend's /api/chat gate (see src/api.py verify_api_key)
OLLIE_API_KEY = os.getenv("OLLIE_API_KEY")


def stream_chat_response(url: str, payload: dict, result: dict):
    """
    Yields answer text chunks from the backend's SSE stream as they arrive.
    Populates `result` with 'sources' and/or 'error' once the stream completes.
    """
    headers = {"X-API-Key": OLLIE_API_KEY} if OLLIE_API_KEY else {}
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=60) as response:
        if response.status_code != 200:
            result["error"] = f"Backend Server Error (Status {response.status_code}): {response.text}"
            return

        event_type = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                event_type = None
                continue
            if raw_line.startswith("event:"):
                event_type = raw_line.split(":", 1)[1].strip()
            elif raw_line.startswith("data:"):
                data = json.loads(raw_line.split(":", 1)[1].strip())
                if event_type == "token":
                    yield data.get("content", "")
                elif event_type == "done":
                    result["sources"] = data.get("sources", [])
                elif event_type == "error":
                    result["error"] = data.get("detail", "The AI service returned an error.")

# --- BRANDING LAYOUT ---
st.markdown('<div class="app-title">Ollie</div>', unsafe_allow_html=True)
st.markdown('<div class="app-caption">Got questions? Let’s chat it out.</div>', unsafe_allow_html=True)

# --- SYSTEM SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
        <div class="welcome-box">
            ✨ <b>Systems Active & Live</b><br><br>
            I'm Ollie, your data companion. Ask me anything about internal compliance, 
            time-off rules, or standard operating procedures, and I'll find it for you!
        </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            source_html = "<div class='source-container'><span style='font-size:0.75rem; color:#94A3B8; font-weight:700;'>VERIFIED SOURCES USED</span><br>"
            for src in msg["sources"]:
                source_html += f'<span class="source-tag">📄 {src}</span>'
            source_html += "</div>"
            st.markdown(source_html, unsafe_allow_html=True)

# --- LIVE USER QUERY INPUT ---
if user_input := st.chat_input("Ask Ollie anything..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        clean_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        payload = {"messages": clean_history, "stream": True}
        result: dict = {}

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer_placeholder.markdown("_Ollie is thinking..._")

            assistant_answer = ""
            for chunk in stream_chat_response(FASTAPI_URL, payload, result):
                assistant_answer += chunk
                answer_placeholder.markdown(assistant_answer + "▌")
            answer_placeholder.markdown(assistant_answer)

            if result.get("error"):
                st.error(result["error"])

            sources = result.get("sources", [])
            if sources:
                source_html = "<div class='source-container'><span style='font-size:0.75rem; color:#94A3B8; font-weight:700;'>VERIFIED SOURCES USED</span><br>"
                for src in sources:
                    source_html += f'<span class="source-tag">📄 {src}</span>'
                source_html += "</div>"
                st.markdown(source_html, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_answer,
            "sources": sources
        })

    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach backend server at `{FASTAPI_URL}`. Make sure your backend service is running.")