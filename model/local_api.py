import requests

def ask_model(prompt: str) -> str:
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen3:1.7b", "prompt": prompt, "stream": False},
            timeout=600,
        )
        res.raise_for_status()
        data = res.json()
        return (data.get("response") or "").replace("<think>", "").replace("</think>", "").strip()
    except requests.RequestException as e:
        return f"Answer: (request to local model failed: {e})\n\nSources Cited:\n"
