import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import time

# --- CONFIG & SECRETS ---
st.set_page_config(page_title="Atlanta RAG Chatbot", layout="centered")

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["INDEX_NAME"]

# --- INITIALIZATION (CACHED) ---

@st.cache_resource
def init_models():
    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using the stable 2026 Gemini 2.5 Flash model
    gemini = genai.GenerativeModel("gemini-2.5-flash")
    
    # Init Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    # Init Embedding Model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    return gemini, index, embed_model

chat_model, index, embed_model = init_models()

# --- LOGIC ---

def create_rag_output(question):
    # Encode question
    q_vec = embed_model.encode(question, convert_to_numpy=True).tolist()
    # Query Pinecone
    res = index.query(vector=q_vec, top_k=5, include_metadata=True)
    return res["matches"]

def do_alex_single_question(question):
    # Define system instruction (Ideally passed into the GenerativeModel constructor)
    system_prompt = (
        "You are an assistant for the Atlanta Police Report Database. "
        "Use the provided JSON RAG context to describe keywords/officers. "
        "Answer using provided info; if insufficient, give top links. "
        "Format: Answer + Date + URL (URL on its own line). "
        "If RAG score < 0.1, state uncertainty but provide links. Be verbose."
    )

    rag_context = create_rag_output(question)
    
    # Construct the prompt
    prompt = f"{system_prompt}\n\nCONTEXT FROM DATABASE:\n{rag_context}\n\nUSER QUESTION: {question}"

    # Error handling for Rate Limits (429)
    try:
        # We use generate_content for single RAG lookups rather than start_chat 
        # to keep the overhead low on the free tier.
        response = chat_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        if "429" in str(e):
            return "⚠️ The system is busy (Rate Limit). Please wait 30 seconds and try again."
        return f"An error occurred: {e}"

# --- UI ---
st.title("Atlanta Police Report Database")
st.info("Searching internal disciplinary records and ACRB board letters.")

query = st.text_input("Ask a question about disciplinary cases:", placeholder="e.g., cases involving excessive force in 2024")

if st.button("Query") and query:
    with st.spinner("Analyzing reports..."):
        response = do_alex_single_question(query)
        st.markdown("---")
        st.markdown("### Analysis")
        st.markdown(response)
