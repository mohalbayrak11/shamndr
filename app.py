from flask import Flask, request, Response, jsonify
import requests
import json
import base64
import concurrent.futures
from duckduckgo_search import DDGS

app = Flask(__name__, static_folder=".", static_url_path="")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

DEFAULT_MODEL = "qwen2.5-coder:7b"

# Models that are generally useful for quick responses.
# The app will only use models that are actually installed.
FAST_MODEL_HINTS = [
    "phi",
    "gemma",
    "qwen2.5",
    "llama3.1",
    "mistral",
]

# Models that are commonly used for coding.
CODING_MODEL_HINTS = [
    "qwen2.5-coder",
    "deepseek-coder",
    "codellama",
    "starcoder",
    "codegemma",
    "deepseek-r1",
]

# Common vision-capable Ollama model naming patterns.
VISION_MODEL_HINTS = [
    "llava",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "gemma3",
    "minicpm-v",
    "moondream",
    "granite3.2-vision",
]


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MULTIAI — AI Team Workspace</title>
<meta name="viewport" content="width=device-width, initial-scale=1">

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #071018;
    --bg2: #0b141d;

    --glass: rgba(18, 28, 38, .68);
    --glass-strong: rgba(19, 30, 41, .88);
    --glass-soft: rgba(255,255,255,.045);

    --text: #f5f8fb;
    --muted: #91a0ad;
    --muted2: #65737f;

    --border: rgba(255,255,255,.09);
    --border2: rgba(255,255,255,.15);

    --blue: #54a8ff;
    --cyan: #4be4d9;
    --purple: #9a7cff;
    --green: #56db84;
    --orange: #ffbb62;
    --red: #ff758f;

    --shadow: 0 20px 70px rgba(0,0,0,.40);
}

* {
    box-sizing: border-box;
}

html,
body {
    width: 100%;
    height: 100%;
    margin: 0;
}

body {
    overflow: hidden;
    color: var(--text);
    font-family: Inter, system-ui, sans-serif;
    background:
        radial-gradient(circle at 70% 0%, rgba(84,168,255,.13), transparent 30%),
        radial-gradient(circle at 0% 100%, rgba(75,228,217,.08), transparent 30%),
        linear-gradient(140deg, #061018 0%, #09151e 45%, #071018 100%);
}

/* Windows-like acrylic background */
.background {
    position: fixed;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
}

.background::before {
    content: "";
    position: absolute;
    inset: -30%;
    background:
        radial-gradient(circle at 30% 25%, rgba(75,228,217,.06), transparent 25%),
        radial-gradient(circle at 75% 20%, rgba(84,168,255,.08), transparent 28%),
        radial-gradient(circle at 50% 90%, rgba(154,124,255,.05), transparent 25%);
    filter: blur(45px);
}

/* subtle chip/grid decoration */
.background::after {
    content: "";
    position: absolute;
    inset: 0;
    opacity: .15;
    background-image:
        linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px);
    background-size: 36px 36px;
}

.app {
    position: relative;
    z-index: 1;
    width: 100%;
    height: 100vh;
    display: flex;
}

/* Sidebar */
.sidebar {
    width: 278px;
    min-width: 278px;
    padding: 18px;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    background: rgba(8,16,23,.65);
    backdrop-filter: blur(26px) saturate(145%);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 8px 22px;
}

.brand-icon {
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    font-family: "Space Grotesk", sans-serif;
    font-weight: 800;
    color: #061018;
    background: linear-gradient(145deg, #a9f5ff, #64a7ff);
    box-shadow:
        inset 0 1px 1px rgba(255,255,255,.5),
        0 8px 25px rgba(84,168,255,.22);
}

.brand-name {
    font-family: "Space Grotesk", sans-serif;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: .3px;
}

.brand-name span {
    color: var(--cyan);
}

.section-title {
    margin: 14px 8px 8px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: var(--muted2);
}

.glass-card {
    border: 1px solid var(--border);
    background: linear-gradient(
        180deg,
        rgba(255,255,255,.045),
        rgba(255,255,255,.018)
    );
    border-radius: 15px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
    backdrop-filter: blur(18px);
}

.status-card {
    padding: 13px;
}

.status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 15px rgba(86,219,132,.6);
}

.side-description {
    margin-top: 8px;
    color: var(--muted);
    font-size: 10.5px;
    line-height: 1.55;
}

.team-list {
    display: grid;
    gap: 7px;
}

.team-chip {
    padding: 10px 11px;
    display: flex;
    align-items: center;
    gap: 9px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: rgba(255,255,255,.025);
    color: var(--muted);
    font-size: 11px;
}

.team-chip strong {
    color: var(--text);
    font-size: 11px;
}

.team-chip .chip-icon {
    width: 25px;
    height: 25px;
    display: grid;
    place-items: center;
    border-radius: 8px;
    font-size: 11px;
    background: rgba(84,168,255,.10);
    border: 1px solid rgba(84,168,255,.16);
}

.bottom-info {
    margin-top: auto;
    padding: 8px;
    color: var(--muted2);
    font-size: 10px;
    line-height: 1.5;
}

/* Main window */
.main {
    min-width: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
}

.topbar {
    height: 72px;
    min-height: 72px;
    padding: 0 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    background: rgba(9,17,24,.42);
    backdrop-filter: blur(20px) saturate(140%);
}

.topbar-left strong {
    display: block;
    font-family: "Space Grotesk", sans-serif;
    font-size: 16px;
}

.topbar-left span {
    display: block;
    margin-top: 3px;
    font-size: 10px;
    color: var(--muted);
}

.connection {
    padding: 7px 11px;
    border-radius: 999px;
    border: 1px solid rgba(86,219,132,.15);
    background: rgba(86,219,132,.04);
    color: var(--green);
    font-size: 10px;
}

/* Chat */
#chatWrap {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
}

#chatWrap::-webkit-scrollbar {
    width: 8px;
}

#chatWrap::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,.08);
    border-radius: 10px;
}

#chat {
    width: 100%;
    max-width: 980px;
    margin: 0 auto;
    padding: 36px 22px 155px;
}

.welcome {
    min-height: 48vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.welcome-icon {
    width: 74px;
    height: 74px;
    display: grid;
    place-items: center;
    border-radius: 22px;
    color: #061018;
    font-family: "Space Grotesk", sans-serif;
    font-size: 25px;
    font-weight: 800;
    background: linear-gradient(145deg, #a9f5ff, #6fa8ff 62%, #9a7cff);
    box-shadow:
        0 20px 65px rgba(84,168,255,.20),
        inset 0 1px 1px rgba(255,255,255,.6);
}

.welcome h1 {
    margin: 18px 0 8px;
    font-family: "Space Grotesk", sans-serif;
    font-size: 31px;
    letter-spacing: -.5px;
}

.welcome p {
    max-width: 610px;
    margin: 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
}

.msg {
    display: flex;
    gap: 12px;
    margin-bottom: 22px;
    line-height: 1.7;
    font-size: 14px;
}

.avatar {
    width: 32px;
    height: 32px;
    min-width: 32px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 700;
}

.user-avatar {
    background: rgba(255,255,255,.08);
    border: 1px solid var(--border);
}

.ai-avatar {
    color: #061018;
    background: linear-gradient(145deg, #a9f5ff, #6fa8ff);
}

.message-content {
    min-width: 0;
    flex: 1;
}

.message-box {
    padding: 17px 18px;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: rgba(14,24,32,.68);
    backdrop-filter: blur(18px);
    box-shadow: 0 14px 45px rgba(0,0,0,.22);
}

.user .message-box {
    background: rgba(255,255,255,.025);
    box-shadow: none;
}

.meta {
    margin-top: 7px;
    color: var(--muted2);
    font-family: "JetBrains Mono", monospace;
    font-size: 9.5px;
}

pre {
    margin: 12px 0;
    padding: 15px;
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.08);
    background: #05090d;
    color: #d6edff;
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    line-height: 1.65;
}

.image-preview-chat {
    max-width: min(500px, 100%);
    max-height: 360px;
    display: block;
    margin-bottom: 12px;
    border-radius: 13px;
    border: 1px solid var(--border);
}

/* Composer */
#inputArea {
    position: relative;
    z-index: 3;
    padding: 14px 20px 20px;
}

#inputBar {
    width: 100%;
    max-width: 980px;
    margin: 0 auto;
    padding: 12px;
    border: 1px solid var(--border2);
    border-radius: 20px;
    background: rgba(14,24,32,.82);
    box-shadow: 0 22px 70px rgba(0,0,0,.38);
    backdrop-filter: blur(24px) saturate(145%);
}

#dropZone.dragging {
    border-color: var(--cyan);
    box-shadow:
        0 0 0 2px rgba(75,228,217,.08),
        0 22px 70px rgba(0,0,0,.38);
}

#promptInput {
    width: 100%;
    min-height: 44px;
    max-height: 170px;
    padding: 9px 10px;
    border: 0;
    outline: none;
    resize: none;
    background: transparent;
    color: var(--text);
    font: 14px/1.6 Inter, sans-serif;
}

#promptInput::placeholder {
    color: #70808d;
}

.preview-row {
    display: none;
    align-items: center;
    gap: 10px;
    margin: 0 6px 8px;
    padding: 8px;
    border: 1px solid var(--border);
    border-radius: 13px;
    background: rgba(255,255,255,.025);
}

.preview-row.visible {
    display: flex;
}

#imagePreview {
    width: 60px;
    height: 60px;
    object-fit: cover;
    border-radius: 10px;
    border: 1px solid var(--border);
}

.preview-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--muted);
    font-size: 11px;
}

.remove-image {
    border: 0;
    background: rgba(255,255,255,.06);
    color: var(--muted);
    width: 28px;
    height: 28px;
    border-radius: 9px;
    cursor: pointer;
}

#toolRow {
    display: flex;
    gap: 8px;
    align-items: center;
    justify-content: space-between;
    margin-top: 7px;
}

#toolLeft {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
}

.control-chip {
    min-height: 34px;
    padding: 7px 10px;
    display: flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--border);
    border-radius: 11px;
    background: rgba(255,255,255,.03);
    color: var(--muted);
    font-size: 10.5px;
}

.control-chip:hover {
    background: rgba(255,255,255,.05);
}

.control-chip.active {
    color: var(--cyan);
    border-color: rgba(75,228,217,.22);
    background: rgba(75,228,217,.06);
}

.control-chip input {
    margin: 0;
    width: 13px;
    height: 13px;
    accent-color: var(--blue);
}

#modeSelect,
#modelSelect {
    height: 34px;
    max-width: 300px;
    padding: 0 10px;
    border: 1px solid var(--border);
    border-radius: 11px;
    outline: none;
    color: var(--text);
    background: rgba(255,255,255,.03);
    font: 10.5px Inter, sans-serif;
}

#modeSelect option,
#modelSelect option {
    background: #10202c;
    color: white;
}

#sendBtn {
    height: 35px;
    padding: 0 18px;
    border: 0;
    border-radius: 11px;
    color: #061018;
    background: linear-gradient(145deg, #a9f5ff, #6fa8ff);
    font: 700 11px Inter, sans-serif;
    cursor: pointer;
    box-shadow:
        inset 0 1px 1px rgba(255,255,255,.55),
        0 8px 24px rgba(84,168,255,.16);
    transition: transform .15s ease, opacity .15s ease;
}

#sendBtn:hover {
    transform: translateY(-1px);
}

#sendBtn:disabled {
    opacity: .45;
    cursor: not-allowed;
    transform: none;
}

.drop-help {
    text-align: center;
    margin-top: 7px;
    color: var(--muted2);
    font-size: 9px;
}

/* Mobile */
@media (max-width: 900px) {
    .sidebar {
        display: none;
    }
}

@media (max-width: 700px) {
    .topbar {
        padding: 0 12px;
    }

    #chat {
        padding: 20px 12px 170px;
    }

    #inputArea {
        padding: 10px;
    }

    #toolRow {
        flex-direction: column;
        align-items: stretch;
    }

    #toolLeft {
        width: 100%;
    }

    #modeSelect,
    #modelSelect {
        flex: 1;
        max-width: none;
    }

    #sendBtn {
        width: 100%;
    }
}
</style>
</head>

<body>

<div class="background"></div>

<div class="app">

    <aside class="sidebar">

        <div class="brand">
            <div class="brand-icon">M</div>
            <div class="brand-name">MULTI<span>AI</span></div>
        </div>

        <div class="section-title">Connection</div>

        <div class="glass-card status-card">
            <div class="status">
                <span class="status-dot"></span>
                <span>Ollama Local</span>
            </div>

            <div class="side-description">
                Your local AI models run directly through Ollama.
            </div>
        </div>

        <div class="section-title">AI Modes</div>

        <div class="team-list">

            <div class="team-chip">
                <div class="chip-icon">⚡</div>
                <div>
                    <strong>Fastest</strong><br>
                    quickest available responses
                </div>
            </div>

            <div class="team-chip">
                <div class="chip-icon">◎</div>
                <div>
                    <strong>AI Team</strong><br>
                    multiple models + refinement
                </div>
            </div>

            <div class="team-chip">
                <div class="chip-icon">{ }</div>
                <div>
                    <strong>Coding Team</strong><br>
                    coding models collaborate
                </div>
            </div>

            <div class="team-chip">
                <div class="chip-icon">◉</div>
                <div>
                    <strong>Single AI</strong><br>
                    direct model response
                </div>
            </div>

        </div>

        <div class="section-title">Models</div>

        <div class="glass-card status-card">
            <div class="side-description" id="modelCount">
                Detecting installed models...
            </div>
        </div>

        <div class="bottom-info">
            MULTIAI<br>
            Fluent / Acrylic workspace
        </div>

    </aside>

    <main class="main">

        <header class="topbar">
            <div class="topbar-left">
                <strong>AI Workspace</strong>
                <span>Compare, collaborate, refine.</span>
            </div>

            <div class="connection">
                ● LOCAL AI
            </div>
        </header>

        <div id="chatWrap">
            <div id="chat">

                <div class="welcome" id="welcome">

                    <div class="welcome-icon">AI</div>

                    <h1>Build answers together.</h1>

                    <p>
                        Pick a mode, choose a model, ask your question,
                        or drop an image into the box below.
                    </p>

                </div>

            </div>
        </div>

        <div id="inputArea">

            <div id="inputBar">

                <div class="preview-row" id="previewRow">

                    <img id="imagePreview" alt="Image preview">

                    <div class="preview-name" id="previewName">
                        image
                    </div>

                    <button
                        class="remove-image"
                        onclick="removeImage()"
                        title="Remove image">
                        ×
                    </button>

                </div>

                <div id="dropZone">

                    <textarea
                        id="promptInput"
                        rows="1"
                        placeholder="Ask anything, compare ideas, build code..."
                    ></textarea>

                </div>

                <div id="toolRow">

                    <div id="toolLeft">

                        <label class="control-chip" id="searchLabel">
                            <input type="checkbox" id="searchToggle">
                            🌐 Web
                        </label>

                        <select id="modeSelect" title="AI mode">
                            <option value="single">Single AI</option>
                            <option value="fastest">⚡ Fastest</option>
                            <option value="team">◎ AI Team</option>
                            <option value="coding">{} Coding Team</option>
                        </select>

                        <select id="modelSelect" title="AI model">
                            <option>Loading models...</option>
                        </select>

                        <input
                            id="imageInput"
                            type="file"
                            accept="image/png,image/jpeg,image/webp,image/gif"
                            style="display:none"
                        >

                        <button
                            class="control-chip"
                            id="imageButton"
                            type="button">
                            🖼 Image
                        </button>

                    </div>

                    <button id="sendBtn" onclick="send()">
                        Send
                    </button>

                </div>

                <div class="drop-help">
                    Drag & drop an image here
                </div>

            </div>

        </div>

    </main>

</div>

<script>
const chat = document.getElementById("chat");
const chatWrap = document.getElementById("chatWrap");
const input = document.getElementById("promptInput");
const modeSelect = document.getElementById("modeSelect");
const modelSelect = document.getElementById("modelSelect");
const searchToggle = document.getElementById("searchToggle");
const searchLabel = document.getElementById("searchLabel");
const sendBtn = document.getElementById("sendBtn");
const welcome = document.getElementById("welcome");
const modelCount = document.getElementById("modelCount");

const imageInput = document.getElementById("imageInput");
const imageButton = document.getElementById("imageButton");
const previewRow = document.getElementById("previewRow");
const imagePreview = document.getElementById("imagePreview");
const previewName = document.getElementById("previewName");
const dropZone = document.getElementById("dropZone");
const inputBar = document.getElementById("inputBar");

let selectedImage = null;
let sending = false;

searchToggle.addEventListener("change", () => {
    searchLabel.classList.toggle("active", searchToggle.checked);
});

imageButton.addEventListener("click", () => {
    imageInput.click();
});

imageInput.addEventListener("change", () => {
    if (imageInput.files.length) {
        setImage(imageInput.files[0]);
    }
});

function setImage(file) {

    if (!file || !file.type.startsWith("image/")) {
        return;
    }

    selectedImage = file;

    const reader = new FileReader();

    reader.onload = () => {

        imagePreview.src = reader.result;
        previewName.textContent = file.name;
        previewRow.classList.add("visible");
        inputBar.classList.add("dragging");

        setTimeout(() => {
            inputBar.classList.remove("dragging");
        }, 450);
    };

    reader.readAsDataURL(file);
}

function removeImage() {

    selectedImage = null;
    imageInput.value = "";
    imagePreview.removeAttribute("src");
    previewRow.classList.remove("visible");
}

["dragenter", "dragover"].forEach(eventName => {

    inputBar.addEventListener(eventName, (e) => {

        e.preventDefault();
        e.stopPropagation();

        inputBar.classList.add("dragging");
    });
});

["dragleave", "drop"].forEach(eventName => {

    inputBar.addEventListener(eventName, (e) => {

        e.preventDefault();
        e.stopPropagation();

        inputBar.classList.remove("dragging");
    });
});

inputBar.addEventListener("drop", (e) => {

    if (e.dataTransfer.files.length) {

        const file = e.dataTransfer.files[0];

        setImage(file);
    }
});

input.addEventListener("keydown", (e) => {

    if (e.key === "Enter" && !e.shiftKey) {

        e.preventDefault();
        send();
    }
});

input.addEventListener("input", () => {

    input.style.height = "auto";
    input.style.height =
        Math.min(input.scrollHeight, 170) + "px";
});

function escapeHtml(text) {

    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatText(text) {

    return escapeHtml(text).replace(
        /```([\s\S]*?)```/g,
        "<pre>$1</pre>"
    );
}

function scrollToBottom() {

    chatWrap.scrollTop = chatWrap.scrollHeight;
}

function addUserMessage(text, imageData) {

    const div = document.createElement("div");

    div.className = "msg user";

    let imageHTML = "";

    if (imageData) {

        imageHTML = `
            <img
                class="image-preview-chat"
                src="${imageData}"
                alt="Uploaded image">
        `;
    }

    div.innerHTML = `
        <div class="avatar user-avatar">YOU</div>

        <div class="message-content">

            <div class="message-box">
                ${imageHTML}
                ${text ? escapeHtml(text) : "Image uploaded"}
            </div>

        </div>
    `;

    chat.appendChild(div);

    if (welcome) {
        welcome.style.display = "none";
    }

    scrollToBottom();
}

function addAiMessage(initialText, metaText) {

    const div = document.createElement("div");

    div.className = "msg";

    div.innerHTML = `
        <div class="avatar ai-avatar">AI</div>

        <div class="message-content">

            <div class="message-box answer-box">
                <div class="answer">
                    ${formatText(initialText)}
                </div>
            </div>

            <div class="meta">
                ${escapeHtml(metaText)}
            </div>

        </div>
    `;

    chat.appendChild(div);

    scrollToBottom();

    return div;
}

async function loadModels() {

    try {

        const response = await fetch("/models");
        const data = await response.json();

        modelSelect.innerHTML = "";

        if (!data.models.length) {

            const option =
                document.createElement("option");

            option.value = "";
            option.textContent =
                "No Ollama models found";

            modelSelect.appendChild(option);

            modelCount.textContent =
                "No models detected";

            return;
        }

        data.models.forEach(model => {

            const option =
                document.createElement("option");

            option.value = model.name;
            option.textContent = model.label;

            modelSelect.appendChild(option);
        });

        const defaultOption =
            [...modelSelect.options].find(
                option => option.value === "qwen2.5-coder:7b"
            );

        if (defaultOption) {
            modelSelect.value = "qwen2.5-coder:7b";
        }

        modelCount.textContent =
            `${data.models.length} installed model${data.models.length === 1 ? "" : "s"}`;

    } catch (error) {

        modelSelect.innerHTML =
            "<option value=''>Ollama unavailable</option>";

        modelCount.textContent =
            "Could not connect to Ollama";
    }
}

function readImageAsBase64(file) {

    return new Promise((resolve, reject) => {

        const reader = new FileReader();

        reader.onload = () => {

            const result = reader.result;

            if (typeof result !== "string") {
                reject(new Error("Could not read image"));
                return;
            }

            const comma =
                result.indexOf(",");

            resolve(
                result.substring(comma + 1)
            );
        };

        reader.onerror = () =>
            reject(new Error("Could not read image"));

        reader.readAsDataURL(file);
    });
}

async function send() {

    if (sending) return;

    const text = input.value.trim();
    const model = modelSelect.value;
    const mode = modeSelect.value;

    if (!text && !selectedImage) {
        return;
    }

    if (!model && mode === "single") {
        return;
    }

    sending = true;
    sendBtn.disabled = true;

    const start = performance.now();

    let imageBase64 = null;
    let imageMime = null;
    let imageDataUrl = null;

    try {

        if (selectedImage) {

            imageMime = selectedImage.type;

            imageDataUrl =
                await new Promise((resolve, reject) => {

                    const reader =
                        new FileReader();

                    reader.onload = () =>
                        resolve(reader.result);

                    reader.onerror = () =>
                        reject(new Error("Image read failed"));

                    reader.readAsDataURL(selectedImage);
                });

            imageBase64 =
                await readImageAsBase64(selectedImage);
        }

        addUserMessage(text, imageDataUrl);

        input.value = "";
        input.style.height = "auto";

        const modeNames = {
            single: "Single AI",
            fastest: "Fastest",
            team: "AI Team",
            coding: "Coding Team"
        };

        const modeName =
            modeNames[mode] || "AI";

        const aiDiv = addAiMessage(
            mode === "fastest"
                ? "⚡ Getting the fastest response..."
                : mode === "team"
                    ? "◎ The AI team is working on it..."
                    : mode === "coding"
                        ? "{} Coding team is analyzing the problem..."
                        : "Thinking...",
            modeName
        );

        const answerElement =
            aiDiv.querySelector(".answer");

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                prompt: text,

                model: model,

                mode: mode,

                web_search: searchToggle.checked,

                image: imageBase64,

                image_mime: imageMime
            })
        });

        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(errorText);
        }

        const reader =
            response.body.getReader();

        const decoder =
            new TextDecoder();

        let full = "";

        while (true) {

            const {
                value,
                done
            } = await reader.read();

            if (done) break;

            full +=
                decoder.decode(
                    value,
                    { stream: true }
                );

            answerElement.innerHTML =
                formatText(full);

            scrollToBottom();
        }

        const elapsed =
            ((performance.now() - start) / 1000).toFixed(1);

        aiDiv.querySelector(".meta").textContent =
            `${modeName} • ${elapsed}s`;

    } catch (error) {

        const errorDiv =
            addAiMessage(
                "Error: " + error.message,
                "ERROR"
            );

        const answer =
            errorDiv.querySelector(".answer");

        answer.style.color = "#ff9aac";

    } finally {

        removeImage();

        sending = false;
        sendBtn.disabled = false;
        sendBtn.textContent = "Send";

        input.focus();
    }
}

loadModels();
</script>

</body>
</html>
"""


def get_installed_models():
    response = requests.get(
        OLLAMA_TAGS_URL,
        timeout=6
    )
    response.raise_for_status()

    data = response.json()

    models = []

    for item in data.get("models", []):
        name = item.get("name", "").strip()

        if not name:
            continue

        lower = name.lower()
        label = name

        if "qwen2.5-coder" in lower:
            label = f"{name} — fast coding"
        elif "deepseek-coder" in lower:
            label = f"{name} — coding specialist"
        elif "deepseek-r1" in lower:
            label = f"{name} — reasoning"
        elif "llama" in lower:
            label = f"{name} — general"
        elif "qwen" in lower:
            label = f"{name} — general"
        elif "mistral" in lower:
            label = f"{name} — fast general"
        elif "phi" in lower:
            label = f"{name} — small / fast"
        elif "gemma" in lower:
            label = f"{name} — general"

        models.append({
            "name": name,
            "label": label
        })

    return models


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/models")
def models():

    try:
        models = get_installed_models()

        return jsonify({
            "models": models
        })

    except Exception as e:

        return jsonify({
            "models": [],
            "error": str(e)
        }), 503


def is_coding_model(name):
    lower = name.lower()

    return any(
        hint in lower
        for hint in CODING_MODEL_HINTS
    )


def is_fast_model(name):
    lower = name.lower()

    return any(
        hint in lower
        for hint in FAST_MODEL_HINTS
    )


def is_vision_model(name):
    lower = name.lower()

    return any(
        hint in lower
        for hint in VISION_MODEL_HINTS
    )


def web_search(query, max_results=5):

    try:

        results = DDGS().text(
            query,
            max_results=max_results
        )

        formatted = ""

        for result in results:

            formatted += (
                f"Title: {result.get('title', '')}\n"
                f"Snippet: {result.get('body', '')}\n"
                f"URL: {result.get('href', '')}\n\n"
            )

        return formatted

    except Exception as e:

        return f"[web search failed: {e}]"


def call_model(
    model,
    prompt,
    image=None,
    system_prompt=None
):

    message = {
        "role": "user",
        "content": prompt
    }

    if image:
        message["images"] = [image]

    payload = {
        "model": model,
        "messages": [message],
        "stream": False
    }

    if system_prompt:
        payload["messages"].insert(
            0,
            {
                "role": "system",
                "content": system_prompt
            }
        )

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    return (
        data
        .get("message", {})
        .get("content", "")
        .strip()
    )


def choose_fast_models(models, limit=3):

    fast = [
        m["name"]
        for m in models
        if is_fast_model(m["name"])
    ]

    if not fast:
        fast = [
            m["name"]
            for m in models
        ]

    # Keep the number limited so fastest mode
    # does not overwhelm a machine.
    return fast[:limit]


def choose_coding_models(models, limit=4):

    coding = [
        m["name"]
        for m in models
        if is_coding_model(m["name"])
    ]

    if not coding:
        coding = [
            m["name"]
            for m in models
        ]

    return coding[:limit]


def choose_team_models(models, limit=5):

    names = [
        m["name"]
        for m in models
    ]

    return names[:limit]


def best_synthesis_model(models):

    preferred = [
        "qwen2.5-coder:7b",
        "deepseek-r1:7b",
        "qwen2.5",
        "llama3.1",
        "mistral"
    ]

    names = {
        m["name"]
        for m in models
    }

    for model in preferred:

        if model in names:
            return model

    if names:
        return list(names)[0]

    return DEFAULT_MODEL


def single_answer(model, prompt, image):

    return call_model(
        model=model,
        prompt=prompt,
        image=image
    )


def fastest_answer(models, prompt, image):

    chosen = choose_fast_models(models)

    if not chosen:
        raise RuntimeError(
            "No installed models are available."
        )

    def worker(model):
        try:
            answer = call_model(
                model=model,
                prompt=prompt,
                image=image
            )

            return {
                "model": model,
                "answer": answer
            }

        except Exception as e:

            return {
                "model": model,
                "answer": "",
                "error": str(e)
            }

    # All selected models start at approximately
    # the same time. The first successful result wins.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(chosen)
    ) as executor:

        futures = [
            executor.submit(worker, model)
            for model in chosen
        ]

        for future in concurrent.futures.as_completed(futures):

            result = future.result()

            if result.get("answer"):

                return (
                    f"{result['answer']}\n\n"
                    f"— Fastest response from {result['model']}"
                )

    raise RuntimeError(
        "Every fast model failed to return an answer."
    )


def collaborative_answer(
    models,
    prompt,
    image,
    coding=False
):

    if coding:
        worker_models = choose_coding_models(models)
    else:
        worker_models = choose_team_models(models)

    if not worker_models:
        raise RuntimeError(
            "No models are available for collaboration."
        )

    role_instruction = (
        """
You are one member of a coding team.
Analyze the programming problem carefully.
Focus on correctness, bugs, edge cases, architecture,
security, and practical implementation.
Return a concrete proposed solution for another AI
to review and improve.
"""
        if coding
        else
        """
You are one member of an AI answer team.
Solve the user's request independently.
Focus on correctness, useful detail, and practical advice.
Your output will be reviewed by another AI.
"""
    )

    def worker(model):

        try:

            answer = call_model(
                model=model,
                prompt=prompt,
                image=image,
                system_prompt=role_instruction
            )

            return {
                "model": model,
                "answer": answer
            }

        except Exception as e:

            return {
                "model": model,
                "answer": "",
                "error": str(e)
            }

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(worker_models)
    ) as executor:

        futures = [
            executor.submit(worker, model)
            for model in worker_models
        ]

        for future in concurrent.futures.as_completed(futures):

            result = future.result()

            if result.get("answer"):
                results.append(result)

    if not results:
        raise RuntimeError(
            "The AI team could not produce any answers."
        )

    synthesis_model = best_synthesis_model(models)

    evidence = "\n\n".join(
        [
            (
                f"MODEL: {result['model']}\n"
                f"PROPOSAL:\n{result['answer']}"
            )
            for result in results
        ]
    )

    if coding:

        synth_prompt = f"""
You are the senior engineer reviewing a team of AI developers.

USER REQUEST:
{prompt}

TEAM PROPOSALS:
{evidence}

Create one improved final answer.

Your job:
1. Compare the proposals.
2. Identify mistakes or weak approaches.
3. Keep the strongest ideas.
4. Fix bugs and contradictions.
5. Produce a practical implementation.
6. For code, provide complete code where appropriate.
7. Do not mention internal team deliberations unless useful.

Return only the refined answer for the user.
"""

    else:

        synth_prompt = f"""
You are the senior AI reviewer.

USER REQUEST:
{prompt}

ANSWERS FROM THE AI TEAM:
{evidence}

Create one superior final answer.

Your job:
1. Compare the answers.
2. Correct errors.
3. Resolve contradictions.
4. Keep the strongest useful ideas.
5. Add missing important details.
6. Make the final response clear and practical.
7. Do not merely paste the individual responses together.

Return one polished answer for the user.
"""

    final = call_model(
        model=synthesis_model,
        prompt=synth_prompt
    )

    return (
        final
        + "\n\n"
        + f"— Refined by the AI team using {len(results)} "
          f"models • final review: {synthesis_model}"
    )


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    prompt = str(
        data.get("prompt", "")
    ).strip()

    selected_model = str(
        data.get("model", DEFAULT_MODEL)
    ).strip()

    mode = str(
        data.get("mode", "single")
    ).strip()

    use_search = bool(
        data.get("web_search", False)
    )

    image = data.get("image")

    if not prompt and not image:
        return Response(
            "Please enter a question or upload an image.",
            status=400,
            mimetype="text/plain"
        )

    try:

        models = get_installed_models()

    except Exception as e:

        return Response(
            f"Could not read Ollama models: {e}",
            status=503,
            mimetype="text/plain"
        )

    model_names = {
        m["name"]
        for m in models
    }

    if mode == "single":

        if selected_model not in model_names:

            return Response(
                f"Model '{selected_model}' is not installed.",
                status=400,
                mimetype="text/plain"
            )

        selected_models = [selected_model]

    elif mode == "fastest":

        selected_models = choose_fast_models(models)

    elif mode == "coding":

        selected_models = choose_coding_models(models)

    elif mode == "team":

        selected_models = choose_team_models(models)

    else:

        return Response(
            "Unknown AI mode.",
            status=400,
            mimetype="text/plain"
        )

    # Vision warning.
    if image:

        if mode == "single":

            if not is_vision_model(selected_model):

                return Response(
                    "An image was uploaded, but the selected model "
                    "does not appear to be a vision model. "
                    "Install/use a vision-capable Ollama model "
                    "such as a model in your Ollama list that supports images.",
                    status=400,
                    mimetype="text/plain"
                )

        else:

            vision_models = [
                model
                for model in selected_models
                if is_vision_model(model)
            ]

            if not vision_models:

                return Response(
                    "You uploaded an image, but none of the selected "
                    "models appears to support vision.",
                    status=400,
                    mimetype="text/plain"
                )

            # Collaborative modes should only send an image to
            # models that can actually process it.
            selected_models = vision_models

    final_prompt = prompt

    if use_search and prompt:

        results = web_search(prompt)

        final_prompt = f"""
Use these current web search results when helpful.

SEARCH RESULTS:
{results}

USER QUESTION:
{prompt}

Answer accurately and distinguish information supported by the
search results from your own reasoning.
"""

    def generate():

        try:

            if mode == "single":

                answer = single_answer(
                    selected_model,
                    final_prompt,
                    image
                )

                yield answer

                return

            if mode == "fastest":

                answer = fastest_answer(
                    models,
                    final_prompt,
                    image
                )

                yield answer

                return

            if mode == "team":

                answer = collaborative_answer(
                    models,
                    final_prompt,
                    image,
                    coding=False
                )

                yield answer

                return

            if mode == "coding":

                answer = collaborative_answer(
                    models,
                    final_prompt,
                    image,
                    coding=True
                )

                yield answer

                return

        except requests.RequestException as e:

            yield f"Ollama request failed: {e}"

        except Exception as e:

            yield f"Error: {e}"

    return Response(
        generate(),
        mimetype="text/plain"
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )