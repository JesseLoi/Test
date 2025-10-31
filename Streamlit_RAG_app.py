import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os



#secrets

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["INDEX_NAME"]


# init
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

# reag query-er
def create_rag_output(question):
  q_vec2 = model.encode(question, convert_to_numpy=True).tolist()
  res3 = index.query(vector=q_vec2, top_k=30, include_metadata=True, include_values=False)
  return res3["matches"]
#Question function

@st.cache_resource
def get_chat_model():
    return genai.GenerativeModel("gemini-2.0-flash-exp")

chat_model = get_chat_model()

def do_alex_single_question(question):
    system_prompt = (
        "You will receive a question. Please use the JSON objects you are handed. "
        "Answer using the provided information. If insufficient, give the links with the highest score. "
        "Return answer plus relevant date and URL. Put the URL on its own line. "
        "If RAG score < 0.1, say you're not sure but still give top links."
    )

    chat = chat_model.start_chat()
    rag_context=create_rag_output(question)
    combined = f"SYSTEM INSTRUCTION:\n{system_prompt}\n\nQUESTION:\n{question}\n\nRAG OUTPUT \n\n{rag_context}"
    response = chat.send_message(combined)
    return response.text.strip()
# ui
st.set_page_config(page_title="Atlanta RAG Chatbot", layout="centered")
st.title(" Police Report Database")

query = st.text_input("Ask a question about disciplinary cases:", placeholder="e.g., What happened in January 2025?")
if st.button("Query") and query:
    with st.spinner("Thinking..."):
        response = do_alex_single_question(query)
        st.markdown("###Response")
        st.write(response)

        st.markdown("### Retrieved Context")
        rag_data = create_rag_output(query)
        for item in rag_data["results"]:
            st.markdown(f"**{item['case']} — {item['date']}**")
            st.markdown(f"Tags: `{', '.join(item['tags'])}`")
            st.markdown(f"Excerpt: {item['excerpt']}...")
            st.markdown(f"[{item['link_text']}]({item['link_url']})")
            st.markdown("---")
