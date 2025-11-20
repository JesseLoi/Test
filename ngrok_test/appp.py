import streamlit as st
import requests


OLLAMA_URL = "http://charis-oesophageal-edmundo.ngrok-free.dev/api/generate"

st.title("Ask Ollama (via Ngrok)")
prompt = st.text_area("Enter your prompt:", height=150)

if st.button("Send"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False
                }
                response = requests.post(OLLAMA_URL, json=payload, verify=False, timeout=1000)
                response.raise_for_status()
                st.success("Response:")
                st.write(response.json()["response"])
            except Exception as e:

                st.error(f"Error: {e}")
