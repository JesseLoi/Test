import streamlit as st
st.set_page_config(page_title="Atlanta RAG Chatbot", layout="centered")
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
  res3 = index.query(vector=q_vec2, top_k=5, include_metadata=True, include_values=False)
  return res3["matches"]
#Question function

@st.cache_resource
def get_chat_model():
    return genai.GenerativeModel("gemini-2.0-flash-exp")
system_prompt = """
You will receive a question and a set of JSON objects representing police disciplinary cases.
Use ONLY the provided information to answer.
If information is insufficient, return the top links with the highest RAG scores.
Return the answer plus relevant dates and URLs, with each URL on its own line.
If RAG score < 0.1, say you are not sure but still give the top links.
Be verbose and structured.
"""


#chat_model = get_chat_model()
chat_model = genai.GenerativeModel(
    "gemini-2.0-flash-exp",
    system_instruction=system_prompt
)
def do_alex_single_question(question):
    rag_matches = create_rag_output(question)

    # Format RAG context into readable text
    formatted_context = "\n".join(
        f"- Case: {m['metadata'].get('case')}\n"
        f"  Date: {m['metadata'].get('date')}\n"
        f"  Excerpt: {m['metadata'].get('excerpt')}\n"
        f"  URL: {m['metadata'].get('link_url')}\n"
        f"  Score: {m.get('score')}\n"
        for m in rag_matches
    )

    prompt = f"""
QUESTION:
{question}

RAG CONTEXT:
{formatted_context}
"""

    response = chat_model.generate_content(prompt)
    return response.text.strip()

# ui
st.title(" Police Report Database")

query = st.text_input("Ask a question about disciplinary cases:", placeholder="e.g., What happened in January 2025?")
if st.button("Query") and query:
    with st.spinner("Thinking..."):
        response = do_alex_single_question(query)
        st.markdown("###Response")
        st.write(response)


