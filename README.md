# SLM-and-RAG-Literature-Retriever

This tool is a local academic literature retriever that uses Small Language Models (SLMs) and Retrieval-Augmented Generation (RAG).

## Required Dependencies
- faiss_cpu==1.11.0
- PyMuPDF==1.26.1
- langchain==0.3.27
- Requests==2.32.4
- scikit_learn==1.7.1
- sentence_transformers==4.1.0
- streamlit==.37.1

## Cloning the Repository
```bash
git clone https://github.com/yourusername/SLM-and-RAG-Literature-Retriever.git
```
```bash
cd SLM-and-RAG-Literature-Retriever
```

## Setup Instructions

Follow the steps below to install the necessary dependencies and run the app:

1. Create a virtual environment
```bash
python3 -m venv .venv
```

If you get an error, install the venv module (for Python 3.12)
```bash
sudo apt install python3.12-venv
```

2. Activate the virtual environment
```bash
source .venv/bin/activate
```

3. Install the required Python packages
```bash
pip install -r requirements.txt
```

4. Install Ollama (needed to run the SLM locally)
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

5. Pull the Qwen3 model (1.7b)
```bash
ollama pull qwen3:1.7b
```

6. Start the Streamlit application
```bash
streamlit run app.py
```
