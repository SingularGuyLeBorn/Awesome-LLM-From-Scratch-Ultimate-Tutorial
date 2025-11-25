import requests

url = "http://127.0.0.1:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "llm-from-scratch",
    "messages": [
        {"role": "user", "content": "请写一首关于 AI 的诗。"}
    ]
}

try:
    response = requests.post(url, headers=headers, json=data)
    print("Status:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)