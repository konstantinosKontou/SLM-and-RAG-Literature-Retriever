import os
import streamlit as st
from indexing.indexer import index_pdfs
from indexing.embedder import get_embedder
from retrieval.retriever import retrieve_chunks
from model.local_api import ask_model

from config import EMBED_MODEL_NAME, INDEX_PATH, DOC_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# SIDEBAR
st.sidebar.title("Settings")
pdf_folder = st.sidebar.text_input("PDF Folder Path", value="papers/")
top_k = st.sidebar.slider("Top K Chunks to Retrieve", 1, 10, 3)

# CHAT SECTION
st.title("Local Academic RAG Assistant")
st.markdown("Ask a question about your local academic PDFs.")
user_question = st.text_input("Ask your question:")
ask_button = st.button("Submit Question")

if ask_button and user_question:

    # CHECK IF THE PATH EXISTS
    # IF NOT EXISTS
    if not os.path.exists(pdf_folder):
        st.error(f"Folder '{pdf_folder}' not found.")

    # IF EXISTS
    else:
        st.success(f"Folder '{pdf_folder}' found.")

        # PROCEED WITH THE INDEXING OF THE DOCUMENTS (INDEX, CHUNK, EMBEDDING)
        st.info("Indexing documents (if not already indexed)...")
        embedder = get_embedder(EMBED_MODEL_NAME)
        index_pdfs(pdf_folder, embedder, INDEX_PATH, DOC_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

        # st.info(f"Searching Top {top_k} documents...")
        chunks = retrieve_chunks(user_question, embedder, INDEX_PATH, DOC_STORE_PATH, top_k)

        context = "\n\n".join(c['text'] for c in chunks)
        metadata_info = "\n".join(
            f"- Source: {c['source']}, Title: {c['title'] or 'N/A'}, Author: {c['author'] or 'N/A'}, Year: {c['year']}"
            for c in chunks
        )

        prompt = f"""You are an academic research assistant. Your task is to answer the user's question strictly based on the provided document excerpts and their metadata.

        ## Instructions:
        - Use **only** the information provided in the "Excerpts".
        - **Do not** use external knowledge.
        - Analyze the **user question** and identify which excerpts are relevant.
        - In your answer, **cite each relevant source once**, but quote **every matching excerpt** from that source.
        - Group multiple relevant excerpts under the same source if they belong to the same document.

        ---

        ## Excerpts:
        {context}

        ---

        ## Metadata:
        {metadata_info}

        ---

        ## User Question:
        {user_question}

        ---

        ## Expected Answer Format:

        Answer: <a summarise of the decided excepts>

        Sources Cited:
        1. **<source file>** — Title: <title or N/A>, Year: <year>, Author: <author>
           - Relevant excerpt(s):
             - "<first matching quote>"
             - "<second matching quote>" (if needed)
        2. **<another source>** — ...

        Only include documents that contain relevant information. If no relevant excerpts are found, say so explicitly.
        """

        with st.spinner("Thinking..."):
            answer = ask_model(prompt)

        st.subheader("Answer")

        if "Answer:" in answer and "Sources Cited:" in answer:
            # Split the answer into 3 parts: intro, answer, sources
            model_thought_part, rest = answer.split("Answer:", 1)
            answer_part, sources_part = rest.split("Sources Cited:", 1)

            st.markdown("**Answer:**")
            st.markdown(answer_part.strip(), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Sources Cited:**")
            st.info(sources_part.strip())

            with st.expander("Model Thought"):
                st.markdown(model_thought_part.strip(), unsafe_allow_html=True)

        elif "Answer:" in answer:
            model_thought_part, answer_part = answer.split("Answer:", 1)

            st.markdown("**Answer:**")
            st.markdown(answer_part.strip(), unsafe_allow_html=True)

            with st.expander("Model Thought"):
                st.markdown(model_thought_part.strip(), unsafe_allow_html=True)

        elif "Sources Cited:" in answer:
            answer_part, sources_part = answer.split("Sources Cited:", 1)

            st.markdown("**Answer:**")
            st.markdown(answer_part.strip(), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Sources Cited:**")
            st.info(sources_part.strip())

        else:
            st.warning("Could not find 'Answer:' or 'Sources Cited:' in the response.")
            st.success(answer)