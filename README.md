# SLM-and-RAG-Literature-Retriever

This tool is a local academic literature retriever that uses Small Language Models (SLMs) and Retrieval-Augmented Generation (RAG).

## Cloning the Repository
```bash
git clone https://github.com/yourusername/SLM-and-RAG-Literature-Retriever.git
```

## Setup Instructions

Follow the steps below to install the necessary dependencies and run the app:

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# If you get an error, install the venv module (for Python 3.12)
sudo apt install python3.12-venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install the required Python packages
pip install -r requirements.txt

# 4. Install Ollama (needed to run the Qwen model locally)
curl -fsSL https://ollama.com/install.sh | sh

# 5. Pull the Qwen3 model (1.7b)
ollama pull qwen3:1.7b

# 6. Start the Streamlit application
streamlit run app.py
