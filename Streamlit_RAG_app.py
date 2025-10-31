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
  res3 = index.query(vector=q_vec2, top_k=40, include_metadata=True, include_values=False)
  return res3["matches"]
#Question function

@st.cache_resource
def get_chat_model():
    return genai.GenerativeModel("gemini-2.0-flash-exp")

chat_model = get_chat_model()

def do_alex_single_question(question):
    system_prompt = ("""
        "You will receive a question. Please use the JSON objects you are handed. "
        "Answer using the provided information. If insufficient, give the links with the highest score. "
        "Return answer plus relevant date and URL. Put the URL on its own line. "
        "If RAG score < 0.1, say you're not sure but still give top links. Be verbose."
        "Here is an example:
Here are some cases involving allegations of excessive force:

*   **Case #19-060:** The allegation of Excessive Force related to a physical assault claim was assigned a finding of Sustained, with a recommended penalty of a Four (4) Day Suspension and Training on the proper Use of Force. Date: 25-Feb-2020
    URL: https://acrbgov.org/wp-content/uploads/2020/03/Board-Letter-to-Chief.19-060..pdf
*   **Case #19-045:** The allegation of Excessive Force related to claims #2-4 was assigned a finding of Sustained, with a recommendation that Sgt. Hines be demoted and receive psychological intervention. Date: 14-Sept-2020
    URL: https://acrbgov.org/wp-content/uploads/2020/09/Board-Letter-to-the-Chief_Case-19-045_Redacted.pdf
*   **Case #19-023:**  An allegation of Excessive Force was made against an officer after an individual was tased and pepper-sprayed. Date: 9-Jan-2020
    URL: https://acrbgov.org/wp-content/uploads/2020/02/Board-Letter-to-Chief-Complaint.19-023-1.pdf
*   **Case #19-074:** Involved allegations of excessive force, abusive language, and violation of APD SOP, also mentioning an officer abusing his authority by threatening to arrest someone for filing a complaint. Date: 9-Jan-2020
    URL: https://acrbgov.org/wp-content/uploads/2020/02/Board-Letter-to-Chief-Complaint.19-074.pdf"
    """)

    chat = chat_model.start_chat()
    rag_context=create_rag_output(question)
    combined = f"SYSTEM INSTRUCTION:\n{system_prompt}\n\nQUESTION:\n{question}\n\nRAG OUTPUT \n\n{rag_context}"
    response = chat.send_message(combined)
    return response.text.strip()
# ui
st.title(" Police Report Database")

query = st.text_input("Ask a question about disciplinary cases:", placeholder="e.g., What happened in January 2025?")
if st.button("Query") and query:
    with st.spinner("Thinking..."):
        response = do_alex_single_question(query)
        st.markdown("###Response")
        st.write(response)


