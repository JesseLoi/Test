import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

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
    res3 = index.query(vector=q_vec2, top_k=5, include_metadata=True, include_values=False)
    return res3["matches"]


# Ollama call
def call_ollama(prompt):
    print(f"Prompt length: {len(prompt)} characters")
    payload = {
        "model": "mistral:latest",
        "prompt": prompt,
        "stream": True  # force streaming mode
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=1000)
        full_text = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                import json
                chunk = json.loads(data)
                full_text += chunk.get("response", "")
                if chunk.get("done", False):
                    break
        return full_text.strip()
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
    formatted_rag = "\n".join([str(entry["metadata"]) for entry in rag_context])
    combined_prompt = f"SYSTEM INSTRUCTION:\n{system_prompt}\n\nQUESTION:\n{question}\n\nRAG OUTPUT:\n{formatted_rag}"
    return call_ollama(combined_prompt)

# UI
query = st.text_input("Ask a question about disciplinary cases:", placeholder="e.g., What happened in January 2025?")
if st.button("Query") and query:
    with st.spinner("Thinking..."):
        response = do_alex_single_question(query)
        st.markdown("### Response")
        st.write(response)






