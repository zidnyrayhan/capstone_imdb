# main.py — IMDB Movie Chatbot (RAG + Qdrant + LangChain + Streamlit)

import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

# --- Konfigurasi untuk halaman
st.set_page_config(page_title="🎬 IMDB Movie Chatbot", page_icon="🎥")

# --- Preflight: untuk cek secrets wajib ada
REQUIRED = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
missing = [k for k in REQUIRED if not st.secrets.get(k)]
if missing:
    st.error(f"Missing secrets in Streamlit Cloud: {', '.join(missing)}")
    st.stop()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_URL     = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# --- Set ENV untuk SDK OpenAI/LangChain auto-read tanpa argumen
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# --- Inisialisasi LLM & Embeddings (OPER API KEY eksplisit)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",          
    openai_api_key=OPENAI_API_KEY,  
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,  
)


# --- Hubungkan ke Qdrant (hasil ingest)
collection_name = "imdb_movies"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# --- Tool: mencari info film di vector DB
@tool
def search_movie_info(query: str):
    """Use this tool to search for relevant movie information from IMDB dataset."""
    try:
        docs = qdrant.similarity_search(query, k=5)
    except Exception as e:
        return f"[Tool Error] Failed to query Qdrant: {e}"
    if not docs:
        return "No related movies found."
    rows = []
    for d in docs:
        meta = d.metadata or {}
        rows.append(
            f"🎬 {meta.get('title', 'Unknown')} ({meta.get('year', '—')}) — "
            f"{meta.get('genre', '—')}, ⭐ {meta.get('rating', '—')}, "
            f"Directed by {meta.get('director', '—')}"
        )
    return "\n\n".join(rows)

tools = [search_movie_info]

# --- Agent runner
def imdb_agent(question: str, history: str):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "Kamu adalah asisten film ceria dan menguasai IMDB Top 1000.\n"
            "Jawab berdasarkan informasi yang ada dari tool `search_movie_info`.\n"
            "Tampilkan judul, tahun, genre, rating, dan juga sutradara bila diperlukan.\n"
            "Jika tidak yakin dengan jawaban, katakan tidak tahu (jangan mengarang).\n"
            "Gunakan bahasa yang ceria dan ramah.\n"
        ),
    )

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    # Token usage (jika tersedia)
    total_input_tokens = 0
    total_output_tokens = 0
    for m in result["messages"]:
        md = getattr(m, "response_metadata", {}) or {}
        usage = md.get("usage_metadata") or md.get("token_usage") or {}
        total_input_tokens += usage.get("input_tokens") or usage.get("prompt_tokens", 0) or 0
        total_output_tokens += usage.get("output_tokens") or usage.get("completion_tokens", 0) or 0

    price = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000

    # Kumpulkan tool calls (jika ada)
    tool_messages = []
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            tool_messages.append(m.content)

    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages,
    }

# --- UI
st.title("🎬 IMDB Movie Chatbot")
st.caption("Tanya apa saja soal IMDB Top 1000 movies (RAG + Qdrant + OpenAI)")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input user
if prompt := st.chat_input("Ask me about movies..."):
    messages_history = st.session_state.get("messages", [])[-20:]
    history = "\n".join([f'{m["role"]}: {m["content"]}' for m in messages_history]) or " "

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = imdb_agent(prompt, history)
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    with st.expander("🔍 Tool Results"):
        st.code("\n\n".join(response["tool_messages"]) or "(no tool calls)")
    with st.expander("📜 Token Usage"):
        st.write(f"Input tokens: {response['total_input_tokens']}")
        st.write(f"Output tokens: {response['total_output_tokens']}")
        st.write(f"Estimated cost: Rp {response['price']:.4f}")
