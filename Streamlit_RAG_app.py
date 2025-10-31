# Paste your full Streamlit app code here
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pinecone
import os

st.set_page_config(page_title="Atlanta RAG Chatbot", layout="centered")
st.title("Police Report Database")

# safe secrets access (fail early with helpful message)
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    INDEX_NAME = st.secrets["INDEX_NAME"]
except Exception as e:
    st.error("Secrets not configured. Add GOOGLE_API_KEY, PINECONE_API_KEY, INDEX_NAME to Streamlit secrets.")
    st.stop()

# cache heavy resources so they load once per worker
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def init_pinecone(api_key, env=None):
    # adjust environment param if you set PINECONE_ENV in secrets
    pinecone.init(api_key=api_key, environment=env)  # environment optional
    return pinecone.Index(INDEX_NAME)

model = get_model()

# genai config (safe to run once)
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_chat_model():
    return genai.GenerativeModel('gemini-2.0-flash-exp')

chat_model = get_chat_model()

# RAG retrieval
def create_rag_output(question, top_k=30):
    q_vec = model.encode(question, convert_to_numpy=True).tolist()
    # index is a pinecone.Index object
    idx = pinecone.Index(INDEX_NAME)
    res = idx.query(vector=q_vec, top_k=top_k, include_metadata=True, include_values=False)
    # res typically contains 'matches' list; return it directly
    return res.get("matches", [])

def do_alex_single_question(question: str):
    system_prompt = (
        "You will receive a question. Please use all the JSON objects you are handed. "
        "Answer using the provided information. If insufficient, give the links with the highest score. "
        "Return answer plus relevant date and URL. Put the URL on its own line. "
        "If RAG score < 0.1, say you're not sure but still give top links."
    )
    chat = chat_model.start_chat(system_instruction=system_prompt)
    response = chat.send_message(question)
    return response.text.strip()

# UI
query = st.text_input("Ask a question about disciplinary cases:", placeholder="e.g., What happened in January 2025?")
if st.button("Query") and query:
    with st.spinner("Thinking..."):
        answer = do_alex_single_question(query)
        st.markdown("### Response")
        st.write(answer)

        st.markdown("### Retrieved Context")
        matches = create_rag_output(query)
        if not matches:
            st.write("No results returned from vector DB.")
        else:
            for m in matches:
                # matches usually contain 'id', 'score', 'metadata'
                meta = m.get("metadata", {}) or {}
                case = meta.get("case", meta.get("title", m.get("id", "unknown case")))
                date = meta.get("date", "unknown date")
                tags = meta.get("tags", [])
                excerpt = meta.get("excerpt", meta.get("text", ""))
                link_url = meta.get("link_url") or meta.get("url") or ""
                link_text = meta.get("link_text") or link_url
                st.markdown(f"**{case} — {date}**")
                st.markdown(f"Tags: `{', '.join(tags)}`")
                st.markdown(f"Excerpt: {excerpt}...")
                if link_url:
                    st.markdown(f"[{link_text}]({link_url})")
                st.markdown("---")
