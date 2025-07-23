import requests

def ask_model(prompt):
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen3:1.7b",
        "prompt": prompt,
        "stream": False
    })
    return res.json()["response"].replace("<think>", "").replace("</think>", "").strip()
