
from flask import Flask, request, jsonify, Response
import requests
import json
from duckduckgo_search import DDGS

app = Flask(__name__, static_folder='.', static_url_path='')

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:7b"

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multi AI Coder</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #05060a;
    --panel: #0d0f18;
    --panel2: #12141f;
    --accent: #7c6fee;
    --accent2: #39d9d9;
    --text: #eceeff;
    --muted: #7a7f99;
    --border: #202336;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    height: 100%;
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
    overflow: hidden;
  }
  body {
    background: var(--bg);
    position: relative;
  }
  #stars, #stars2, #stars3 {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
  }
  #stars { background-image: radial-gradient(1px 1px at 20px 30px, white, transparent), radial-gradient(1px 1px at 90px 120px, white, transparent), radial-gradient(1px 1px at 160px 60px, white, transparent), radial-gradient(1px 1px at 220px 180px, white, transparent), radial-gradient(1px 1px at 300px 40px, white, transparent), radial-gradient(1px 1px at 380px 150px, white, transparent); background-size: 420px 220px; opacity: 0.5; animation: twinkle 6s ease-in-out infinite alternate; }
  #stars2 { background-image: radial-gradient(1.5px 1.5px at 50px 90px, white, transparent), radial-gradient(1.5px 1.5px at 150px 30px, white, transparent), radial-gradient(1.5px 1.5px at 260px 140px, white, transparent), radial-gradient(1.5px 1.5px at 340px 70px, white, transparent); background-size: 400px 200px; opacity: 0.3; animation: twinkle 8s ease-in-out infinite alternate-reverse; }
  #nova {
    position: fixed;
    top: -20%;
    right: -10%;
    width: 800px;
    height: 800px;
    z-index: 0;
    pointer-events: none;
    background: radial-gradient(circle, rgba(124,111,238,0.35) 0%, rgba(124,111,238,0.15) 35%, transparent 65%);
    filter: blur(20px);
    animation: pulse 10s ease-in-out infinite;
  }
  #nova2 {
    position: fixed;
    bottom: -25%;
    left: -15%;
    width: 700px;
    height: 700px;
    z-index: 0;
    pointer-events: none;
    background: radial-gradient(circle, rgba(57,217,217,0.22) 0%, rgba(57,217,217,0.08) 40%, transparent 65%);
    filter: blur(20px);
    animation: pulse 13s ease-in-out infinite reverse;
  }
  @keyframes twinkle { 0% { opacity: 0.3; } 100% { opacity: 0.65; } }
  @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.12); } }

  body {
    display: flex;
    flex-direction: column;
    height: 100vh;
    position: relative;
  }
  header {
    padding: 22px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    z-index: 2;
    position: relative;
  }
  header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 19px;
    font-weight: 800;
    margin: 0;
    letter-spacing: 0.5px;
    color: var(--text);
    text-shadow: 0 0 24px rgba(124,111,238,0.5);
  }
  header h1 span { color: var(--accent2); }

  #chatWrap {
    flex: 1;
    overflow-y: auto;
    display: flex;
    justify-content: center;
    z-index: 1;
    position: relative;
  }
  #chat {
    width: 100%;
    max-width: 760px;
    padding: 20px 24px 40px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  .msg {
    line-height: 1.65;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 15px;
  }
  .user {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 19px;
    color: var(--text);
    padding-top: 8px;
  }
  .ai {
    background: rgba(13,15,24,0.85);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 10px;
  }
  .ai pre {
    background: #05060a;
    color: #d9d4ff;
    padding: 14px;
    border-radius: 10px;
    overflow-x: auto;
    border: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
  }

  #inputArea {
    display: flex;
    justify-content: center;
    padding: 18px 24px 26px;
    z-index: 2;
    position: relative;
  }
  #inputBar {
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 760px;
    background: rgba(13,15,24,0.9);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 14px 16px 10px;
    gap: 10px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
  }
  #promptInput {
    width: 100%;
    background: transparent;
    border: none;
    color: var(--text);
    resize: none;
    font-size: 15px;
    font-family: 'Inter', sans-serif;
  }
  #promptInput:focus { outline: none; }
  #toolRow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }
  #toolRowLeft {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .pill {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12.5px;
    color: var(--muted);
    cursor: pointer;
    user-select: none;
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 12px;
  }
  .pill input { accent-color: var(--accent); width: 14px; height: 14px; cursor: pointer; margin: 0; }
  .pill.active { color: var(--accent2); font-weight: 600; border-color: var(--accent2); }
  select#modelSelect {
    background: var(--panel2);
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 12px;
    font-size: 12.5px;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
  }
  button#sendBtn {
    background: linear-gradient(135deg, var(--accent), #5a4fd6);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 9px 20px;
    cursor: pointer;
    font-size: 13.5px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
  }
  button#sendBtn:hover { opacity: 0.88; }
  #chat::-webkit-scrollbar, #chatWrap::-webkit-scrollbar { width: 8px; }
  #chatWrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
</style>
</head>
<body>

<div id="stars"></div>
<div id="stars2"></div>
<div id="nova"></div>
<div id="nova2"></div>

<header>
  <h1>MULTI<span>AI</span></h1>
</header>

<div id="chatWrap"><div id="chat"></div></div>

<div id="inputArea">
  <div id="inputBar">
    <textarea id="promptInput" rows="1" placeholder="Ask anything..."></textarea>
    <div id="toolRow">
      <div id="toolRowLeft">
        <label class="pill" id="searchToggleLabel">
          <input type="checkbox" id="searchToggle">
          🌐 Search
        </label>
        <select id="modelSelect">
          <option value="qwen2.5-coder:7b">qwen2.5-coder</option>
          <option value="llama3.1">llama3.1</option>
          <option value="qwen2.5">qwen2.5</option>
          <option value="mistral">mistral</option>
        </select>
      </div>
      <button id="sendBtn" onclick="send()">Send</button>
    </div>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const chatWrap = document.getElementById('chatWrap');
const input = document.getElementById('promptInput');
const modelSelect = document.getElementById('modelSelect');
const searchToggle = document.getElementById('searchToggle');
const searchLabel = document.getElementById('searchToggleLabel');

searchToggle.addEventListener('change', () => {
  searchLabel.classList.toggle('active', searchToggle.checked);
});

function addMessage(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = formatText(text);
  chat.appendChild(div);
  chatWrap.scrollTop = chatWrap.scrollHeight;
  return div;
}

function formatText(text) {
  return text.replace(/```([\\s\\S]*?)```/g, '<pre>$1</pre>');
}

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

async function send() {
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, 'user');
  input.value = '';

  const aiDiv = addMessage(searchToggle.checked ? 'searching web...' : '...', 'ai');
  const startTime = performance.now();

  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ prompt: text, model: modelSelect.value, web_search: searchToggle.checked })
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = '';
  aiDiv.textContent = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    full += decoder.decode(value, { stream: true });
    aiDiv.innerHTML = formatText(full);
    chatWrap.scrollTop = chatWrap.scrollHeight;
  }

  const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = 'thought for ' + elapsed + 's';
  aiDiv.appendChild(meta);
}
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return HTML_PAGE

def web_search(query, max_results=4):
    try:
        results = DDGS().text(query, max_results=max_results)
        formatted = ""
        for r in results:
            formatted += f"Title: {r.get('title','')}\nSnippet: {r.get('body','')}\nURL: {r.get('href','')}\n\n"
        return formatted
    except Exception as e:
        return f"[web search failed: {e}]"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "")
    model = data.get("model", DEFAULT_MODEL)
    use_search = data.get("web_search", False)

    final_prompt = prompt
    if use_search:
        results = web_search(prompt)
        final_prompt = f"""Use these live web search results to answer the question accurately. Cite facts from them where relevant.

SEARCH RESULTS:
{results}

QUESTION: {prompt}"""

    def generate():
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": final_prompt}],
            "stream": True
        }
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    msg = chunk.get("message", {}).get("content", "")
                    if msg:
                        yield msg

    return Response(generate(), mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
