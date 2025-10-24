import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

# ---- Ambil secrets dan preflight
REQUIRED = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
missing = [k for k in REQUIRED if not st.secrets.get(k)]
if missing:
    st.error(f"Missing secrets in Streamlit Cloud: {', '.join(missing)}")
    st.stop()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_URL     = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# ---- Inisialisasi model dan embeddings
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,  # <—  openai_api_key
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,  # <—  openai_api_key
)


# Ambil API key dari Streamlit secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- Inisialisasi model LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

# --- Inisialisasi embeddings dan koneksi ke Qdrant
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

collection_name = "imdb_movies"

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# --- Definisikan Tool untuk pencarian film di Qdrant
@tool
def search_movie_info(query: str):
    """Use this tool to search for relevant movie information from IMDB dataset."""
    docs = qdrant.similarity_search(query, k=5)
    if not docs:
        return "No related movies found."
    results = []
    for d in docs:
        meta = d.metadata
        results.append(f"🎬 {meta.get('title')} ({meta.get('year')}) — {meta.get('genre')}, ⭐ {meta.get('rating')}, Directed by {meta.get('director')}")
    return "\n\n".join(results)

tools = [search_movie_info]

# --- Fungsi utama agent
def imdb_agent(question, history):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are a helpful and cheerful movie assistant that knows about IMDB Top 1000 movies.\n"
            "Always answer based on movie information from the provided tool.\n"
            "If unsure about the answer, say you don't know rather than making things up.\n"
            "Use clear and friendly language.\n"
        ),
    )

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    # Hitung token usage
    total_input_tokens = 0
    total_output_tokens = 0
    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]

    price = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000

    # Kumpulkan hasil tool calls
    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_messages.append(message.content)

    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages,
    }

# --- Streamlit UI ---
st.set_page_config(page_title="🎥 IMDB Movie Chatbot", page_icon="🎬")
st.title("🎬 IMDB Movie Chatbot")
st.caption("Ask me anything about IMDB Top 1000 movies!")

# Inisialisasi history
if "messages" not in st.session_state:
     st.session_state.messages = []

# Tampilkan history lama
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
       st.markdown(msg["content"])

# Input dari user
if prompt := st.chat_input("Ask me about movies..."):
    messages_history = st.session_state.get("messages", [])[-20:]
    history = "\n".join([f'{m["role"]}: {m["content"]}' for m in messages_history]) or " "

    # Tampilkan pertanyaan user
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Jalankan agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = imdb_agent(prompt, history)
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    # Expander untuk detail teknis
    with st.expander("🔍 Tool Results"):
        st.code("\n\n".join(response["tool_messages"]))
    with st.expander("📜 Token Usage"):
        st.write(f"Input tokens: {response['total_input_tokens']}")
        st.write(f"Output tokens: {response['total_output_tokens']}")
        st.write(f"Estimated cost: Rp {response['price']:.4f}")
