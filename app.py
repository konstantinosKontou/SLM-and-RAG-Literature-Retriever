import os
import time
import threading
from datetime import datetime

import streamlit as st
import psutil

from indexing.indexer import index_pdfs
from indexing.embedder import get_embedder
from retrieval.retriever import retrieve_chunks
from model.local_api import ask_model
from config import EMBED_MODEL_NAME, INDEX_PATH, DOC_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def find_ollama_pids():
    """Return list of PIDs for the 'ollama' server/workers."""
    pids = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama serve" in cmd:
                pids.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids


def start_sampler(pids, bucket, stop_event, interval=1.0):
    """Sample CPU and RAM for ollama PIDs every `interval` seconds into `bucket`."""
    procs = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
            p.cpu_percent(interval=None)  # prime
            procs.append(p)
        except psutil.Error:
            pass

    def loop():
        while not stop_event.is_set():
            cpu_sum = 0.0
            rss_sum_mb = 0.0
            alive = []
            for p in procs:
                try:
                    with p.oneshot():
                        cpu_sum += p.cpu_percent(interval=None)
                        rss_sum_mb += p.memory_info().rss / (1024 * 1024)
                        alive.append(p)
                except psutil.Error:
                    pass
            procs[:] = alive
            bucket.append({"t": time.time(), "cpu_sum_pct": cpu_sum, "rss_mb": rss_sum_mb})
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# -------------------- UI --------------------
st.sidebar.title("Settings")
pdf_folder = st.sidebar.text_input("PDF Folder Path", value="papers/")
top_k = st.sidebar.slider("Top K Chunks to Retrieve", 1, 10, 3)

st.title("Local Academic RAG Assistant")
st.markdown("Ask a question about your local academic PDFs.")
user_question = st.text_input("Ask your question:")
ask_button = st.button("Submit Question")

if ask_button and user_question:
    overall_start = time.time()

    if not os.path.exists(pdf_folder):
        st.error(f"Folder '{pdf_folder}' not found.")
        st.stop()
    else:
        st.success(f"Folder '{pdf_folder}' found.")

    # --- Indexing ---
    t0 = time.time()
    embedder = get_embedder(EMBED_MODEL_NAME)
    index_pdfs(pdf_folder, embedder, INDEX_PATH, DOC_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    t1 = time.time()

    # --- Retrieval ---
    chunks = retrieve_chunks(user_question, embedder, INDEX_PATH, DOC_STORE_PATH, top_k)
    t2 = time.time()

    context = "\n\n".join(c['text'] for c in chunks)
    metadata_info = "\n".join(
        f"- Source: {c.get('source')}, Title: {c.get('title') or 'N/A'}, "
        f"Author: {c.get('author') or 'N/A'}, Year: {c.get('year')}"
        for c in chunks
    )

    prompt = f"""You are an academic research assistant. Your task is to answer the user's question strictly based 
    on the provided document excerpts and their metadata.

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

    # --- Generation + sampling ---
    gen_start = time.time()
    ollama_pids = find_ollama_pids()
    timeline = []
    stop_ev = threading.Event()
    sampler_thread = start_sampler(ollama_pids, timeline, stop_ev, interval=1.0)

    try:
        with st.spinner("Thinking..."):
            answer = ask_model(prompt)
    finally:
        stop_ev.set()
        sampler_thread.join(timeout=2)
    gen_end = time.time()

    overall_end = time.time()

    # --- Show answer only ---
    st.subheader("Answer")
    if "Answer:" in answer and "Sources Cited:" in answer:
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

    # --- Save metrics ---
    os.makedirs("efficiency_metrics", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"efficiency_metrics/metrics_{ts}.txt"

    indexing_s = t1 - t0
    retrieval_s = t2 - t1
    generation_s = gen_end - gen_start
    overall_s = overall_end - overall_start

    cpu_avg = sum(s["cpu_sum_pct"] for s in timeline) / len(timeline) if timeline else None
    ram_peak = max(s["rss_mb"] for s in timeline) if timeline else None

    with open(out_path, "w") as f:
        f.write(f"Timestamp: {ts}\n")
        f.write(f"User Question: {user_question}\n\n")
        f.write("--- Stage Timings ---\n")
        f.write(f"Indexing: {indexing_s:.2f} s\n")
        f.write(f"Retrieval: {retrieval_s:.2f} s\n")
        f.write(f"Generation: {generation_s:.2f} s\n")
        f.write(f"Overall: {overall_s:.2f} s\n\n")
        f.write("--- Generation Metrics (Ollama processes) ---\n")
        if cpu_avg is not None:
            f.write(f"Avg CPU% (sum of cores): {cpu_avg:.2f}\n")
        if ram_peak is not None:
            f.write(f"Peak RAM (MB): {ram_peak:.2f}\n")
