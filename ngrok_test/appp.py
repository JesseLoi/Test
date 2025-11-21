import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# UI setup
st.set_page_config(page_title="Atlanta RAG Chatbot", layout="centered")
st.title("📂 Police Report Database")

# Secrets
OLLAMA_URL = st.secrets["OLLAMA_URL"]  # e.g., "https://your-ngrok-url/api/generate"
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["INDEX_NAME"]

# Init
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

# RAG query (top 2 only)
def create_rag_output(question):
    q_vec2 = model.encode(question, convert_to_numpy=True).tolist()
    res3 = index.query(vector=q_vec2, top_k=1, include_metadata=True, include_values=False)
    return res3["matches"]

def clean_metadata(md):
    text = str(md)
    return text[:200]  # keep at most 500 chars
#formatted_rag = "\n".join([clean_metadata(entry["metadata"]) for entry in rag_context])


# Ollama call
def call_ollama(prompt):
    payload = {
        "model": "mistral:latest",
        "prompt": prompt,
        "stream": False  # IMPORTANT: no streaming over ngrok TCP
    }

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)

    try:
        response = session.post(
            OLLAMA_URL, 
            json=payload, 
            timeout=200  # longer for LLM
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error calling Ollama: {e}"

# Main query function
def do_alex_single_question(question):
    system_prompt = (
        "Use the JSON objects to tell the user about the keywords. Tell them about the context and how that relates to the query. Please give the date, relevant tags, and url of the incidents "
        "Keep it concise.\n\n"
        "Here is an example:\n"
        "* **Case #19-060:** Excessive Force, 4-day suspension. Date: 25-Feb-2020\n"
        "  URL: https://acrbgov.org/wp-content/uploads/2020/03/Board-Letter-to-Chief.19-060..pdf\n"
    )

    rag_context = create_rag_output(question)
    formatted_rag = "\n".join([clean_metadata(entry["metadata"]) for entry in rag_context])
    #formatted_rag = "\n".join([str(entry["metadata"]) for entry in rag_context])
    combined_prompt = f"SYSTEM INSTRUCTION:\n{system_prompt}\n\nQUESTION:\n{question}\n\nRAG OUTPUT:\n{formatted_rag}"
    return call_ollama(combined_prompt)

# UI
query = st.text_input("Ask a question about disciplinary cases:", placeholder="e.g., What happened in January 2025?")
if st.button("Query") and query:
    with st.spinner("Thinking..."):
        response = do_alex_single_question(query)
        st.markdown("### Response")
        st.write(response)












