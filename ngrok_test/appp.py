import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# UI setup
st.set_page_config(page_title="Atlanta RAG Chatbot", layout="centered")
st.title("Police Complaints Database")

# Secrets
OLLAMA_URL = st.secrets["OLLAMA_URL"] #This is going to change very often since the URL is actually pointing to the cloudflare tunnel that goes to our model
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"] #This is my personal pinecone API key. You may use your own (I believe there are no query limits)
INDEX_NAME = st.secrets["INDEX_NAME"] #This is the name of my pinecone database

# Init
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2") #The reason we need to use this particular transformer is because all our data is encoded this way, so our query should also be encoded that way

# RAG query (top 2 only)
def create_rag_output(question):
    q_vec2 = model.encode(question, convert_to_numpy=True).tolist()
    res3 = index.query(vector=q_vec2, top_k=2, include_metadata=True, include_values=False)
    return res3["matches"]
    #The reason I only return the top 2 is because my model is pretty slow

def clean_metadata(md):
    text = str(md)
    return text[:200]  # keep at most 200 chars
#formatted_rag = "\n".join([clean_metadata(entry["metadata"]) for entry in rag_context])
#in the ideal case, we would be taking all the characters

# Ollama call
def call_ollama(prompt):
    payload = {
        "model": "llama3.1:70b",
        #you can use whatever model you want. We will probably change this to ollama3.1:70b for fun
        "prompt": prompt,
        "stream": True #there is a pretty important reason we're using streaming as opposed to not. Cloudflare closes its connection if you spend more than 100 seconds without input
        #So we need to stream just to send it something
    }

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)#https is ok for cloudflare (I tried it on ngrok and https could not authenticate such a large portion

    try:
        response = session.post(
            OLLAMA_URL, 
            json=payload, 
            stream=True,  # stream mode
            timeout=1000 #our model is slow so we set a very large timeout
        )
        response.raise_for_status()

        full_text = ""
        output_container = st.empty()  # placeholder for live output
        for line in response.iter_lines(): #this is to make our model look like it's typing
            if line:
                import json
                chunk = json.loads(line.decode("utf-8"))
                text_chunk = chunk.get("response", "")
                full_text += text_chunk
                # update the container with the accumulated text
                output_container.markdown(full_text)
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
        "Officer Johnson was accused of exessive force when he pushed a civilian. The complaint was sustained and he received a 4-day suspension"
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


















