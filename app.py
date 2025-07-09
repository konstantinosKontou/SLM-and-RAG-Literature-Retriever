import streamlit as st

# SIDEBAR
st.sidebar.title("Settings")
pdf_folder = st.sidebar.text_input("PDF Folder Path", value="papers/")
top_k = st.sidebar.slider("Top K Chunks to Retrieve", 1, 10, 3)

# CHAT SECTION
st.title("Local Academic RAG Assistant")
st.markdown("Ask a question about your local academic PDFs.")
user_question = st.text_input("Ask your question:")