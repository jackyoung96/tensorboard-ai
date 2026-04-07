/**
 * Chat side-panel with SSE streaming, markdown rendering, and metric links.
 */

import { buildFullDataContext, getSelectedTagNames } from "./index.js";

const CHAT_ENDPOINT = "./chat";

let panelEl = null;
let messagesEl = null;
let inputEl = null;
let sendBtn = null;
let titleEl = null;
let historySelect = null;
let conversationHistory = [];
let knownTags = [];
let markedReady = null;
let chatSettings = { language: "", provider: "", model: "", useReasoning: false, debug: false };
let isStreaming = false;
let isFirstMessage = true;

/** Analysis history: array of { id, title, conversationHistory, messagesHTML, dataContext, isFirstMessage } */
let analysisHistory = [];
let activeHistoryId = null;
let nextHistoryId = 1;

/**
 * Load marked.js for markdown rendering.
 */
function loadMarked() {
  if (markedReady) return markedReady;
  markedReady = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "./marked.min.js";
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load marked.js"));
    document.head.appendChild(s);
  });
  return markedReady;
}

function showEmptyState() {
  messagesEl.innerHTML = `
    <div class="tb-ai-empty-state">
      <div class="tb-ai-empty-icon">AI</div>
      <div>Click <strong>Analyze All</strong> or a chart's <strong>AI</strong> button to start analysis</div>
    </div>
  `;
}

export function initChatPanel(container) {
  panelEl = container;

  panelEl.innerHTML = `
    <div class="tb-ai-chat-header">
      <span class="tb-ai-chat-title">AI Analysis</span>
      <div class="tb-ai-chat-header-right">
        <button class="tb-ai-new-btn">+ New</button>
        <select class="tb-ai-history-select"></select>
      </div>
    </div>
    <div class="tb-ai-chat-messages"></div>
    <div class="tb-ai-chat-input-row">
      <input type="text" placeholder="Analyze all training metrics, identify trends, issues, and cross-metric correlations." />
      <button class="tb-ai-chat-send">Send</button>
    </div>
  `;

  titleEl = panelEl.querySelector(".tb-ai-chat-title");
  historySelect = panelEl.querySelector(".tb-ai-history-select");
  messagesEl = panelEl.querySelector(".tb-ai-chat-messages");
  inputEl = panelEl.querySelector("input");
  sendBtn = panelEl.querySelector(".tb-ai-chat-send");

  const newBtn = panelEl.querySelector(".tb-ai-new-btn");
  newBtn.addEventListener("click", () => {
    openChat({ title: "AI Analysis" });
  });

  historySelect.addEventListener("change", () => {
    const id = parseInt(historySelect.value, 10);
    if (id) restoreHistory(id);
  });
  sendBtn.addEventListener("click", sendMessage);

  // Show empty state
  showEmptyState();
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  // Handle metric link clicks via event delegation
  messagesEl.addEventListener("click", (e) => {
    const link = e.target.closest(".tb-ai-metric-link");
    if (link) {
      const tag = link.dataset.tag;
      document.dispatchEvent(new CustomEvent("tb-ai-scroll-to-chart", { detail: { tag } }));
    }
  });

  loadMarked();
}

/**
 * Set the list of known metric tags (for linkification).
 */
export function setKnownTags(tags) {
  knownTags = tags;
}

/**
 * Update analysis settings (language, model, reasoning, debug).
 */
export function setChatSettings(settings) {
  const prevDebug = chatSettings.debug;
  chatSettings = { ...chatSettings, ...settings };
  // Toggle visibility of debug messages when debug flag changes
  if (chatSettings.debug !== prevDebug && messagesEl) {
    messagesEl.querySelectorAll("[data-debug-msg]").forEach((el) => {
      el.style.display = chatSettings.debug ? "" : "none";
    });
  }
}

/**
 * Save current session to history (if it has content).
 */
function saveCurrentToHistory() {
  if (conversationHistory.length === 0) return;
  const existing = analysisHistory.find((h) => h.id === activeHistoryId);
  if (existing) {
    existing.conversationHistory = [...conversationHistory];
    existing.messagesHTML = messagesEl.innerHTML;
    existing.isFirstMessage = isFirstMessage;
  } else {
    const entry = {
      id: nextHistoryId++,
      title: titleEl.textContent || "Analysis",
      conversationHistory: [...conversationHistory],
      messagesHTML: messagesEl.innerHTML,
      dataContext: panelEl.dataset.context || "",
      isFirstMessage,
    };
    analysisHistory.push(entry);
    activeHistoryId = entry.id;
  }
  rebuildHistorySelect();
}

/**
 * Restore a session from history.
 */
function restoreHistory(id) {
  // Save current first
  saveCurrentToHistory();

  const entry = analysisHistory.find((h) => h.id === id);
  if (!entry) return;

  activeHistoryId = entry.id;
  conversationHistory = [...entry.conversationHistory];
  messagesEl.innerHTML = entry.messagesHTML;
  titleEl.textContent = entry.title;
  panelEl.dataset.context = entry.dataContext;
  isFirstMessage = entry.isFirstMessage;
  isStreaming = false;
  setInputEnabled(true);

  // Re-apply debug visibility
  messagesEl.querySelectorAll("[data-debug-msg]").forEach((el) => {
    el.style.display = chatSettings.debug ? "" : "none";
  });

  rebuildHistorySelect();
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/**
 * Rebuild the history dropdown options.
 */
function rebuildHistorySelect() {
  historySelect.innerHTML = "";
  if (analysisHistory.length === 0) {
    historySelect.style.display = "none";
    return;
  }
  historySelect.style.display = "";

  // Show current (new) session indicator when not viewing a saved session
  if (!activeHistoryId) {
    const cur = document.createElement("option");
    cur.value = "0";
    cur.textContent = "— Current —";
    cur.selected = true;
    historySelect.appendChild(cur);
  }

  for (const entry of analysisHistory) {
    const opt = document.createElement("option");
    opt.value = String(entry.id);
    opt.textContent = entry.title.length > 30 ? entry.title.slice(0, 28) + "..." : entry.title;
    opt.title = entry.title;
    if (entry.id === activeHistoryId) opt.selected = true;
    historySelect.appendChild(opt);
  }
}

/**
 * Start a new analysis (called externally). Saves current session to history first.
 */
export function openChat({ title, dataContext, autoMessage }) {
  // Save current session before starting a new one
  saveCurrentToHistory();

  // Start fresh
  activeHistoryId = null;
  conversationHistory = [];
  messagesEl.innerHTML = "";
  titleEl.textContent = title || "AI Analysis";
  panelEl.dataset.context = dataContext || "";
  isFirstMessage = true;
  rebuildHistorySelect();

  if (autoMessage) {
    inputEl.value = autoMessage;
    sendMessage();
  } else {
    setInputEnabled(true);
    inputEl.focus();
  }
}

function setInputEnabled(enabled) {
  inputEl.disabled = !enabled;
  sendBtn.disabled = !enabled;
  if (enabled) {
    inputEl.classList.remove("tb-ai-disabled");
    sendBtn.classList.remove("tb-ai-disabled");
  } else {
    inputEl.classList.add("tb-ai-disabled");
    sendBtn.classList.add("tb-ai-disabled");
  }
}

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = role === "user" ? "tb-ai-msg tb-ai-msg-user" : "tb-ai-msg tb-ai-msg-assistant";
  if (role === "user") {
    el.textContent = text;
  }
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

/**
 * Append a debug-only message block showing the raw prompt sent to the API.
 * Hidden by default; shown when debug checkbox is on. Toggled reactively.
 */
function appendDebugBlock(content) {
  const el = document.createElement("div");
  el.className = "tb-ai-debug-block";
  el.dataset.debugMsg = "1";

  const header = document.createElement("div");
  header.className = "tb-ai-debug-header";
  header.textContent = "Raw User Message (sent to API)";
  el.appendChild(header);

  const pre = document.createElement("pre");
  pre.className = "tb-ai-debug-pre";
  pre.textContent = content;
  el.appendChild(pre);

  if (!chatSettings.debug) {
    el.style.display = "none";
  }
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/**
 * Format token count with K/M units.
 */
function formatTokens(n) {
  if (n == null) return "?";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

/**
 * Append a token usage bar after the assistant message.
 */
function appendUsageInfo(usage) {
  if (!usage) return;
  const el = document.createElement("div");
  el.className = "tb-ai-usage-info";
  el.textContent =
    "Tokens: " + formatTokens(usage.input_tokens) + " in / " +
    formatTokens(usage.output_tokens) + " out";
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/**
 * Render markdown into an element and linkify known metric names.
 */
function renderMarkdown(el, markdown) {
  /* global marked */
  if (typeof marked !== "undefined") {
    try {
      // Escape single tildes that aren't part of ~~ strikethrough
      // e.g., "Step ~5000" → "Step \~5000", but keep "~~deleted~~" intact
      const escaped = markdown.replace(/(?<![~\\])~(?!~)/g, "\\~");
      el.innerHTML = marked.parse(escaped, { breaks: true, gfm: true });
    } catch {
      el.textContent = markdown;
      return;
    }
  } else {
    el.textContent = markdown;
    return;
  }
  linkifyMetrics(el);
}

/**
 * Walk text nodes in the element and wrap known tag names with clickable spans.
 */
function linkifyMetrics(container) {
  if (!knownTags.length) return;

  // Sort longest first to prevent partial matches (e.g., "eval/loss" before "loss")
  const sorted = [...knownTags].sort((a, b) => b.length - a.length);
  const escaped = sorted.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const regex = new RegExp("(?<=\\s|^|\\`)(" + escaped.join("|") + ")(?=\\s|$|\\`|[,.:;!?)])", "g");

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  for (const node of textNodes) {
    // Skip nodes inside code blocks, links, or already-linked spans
    if (node.parentElement?.closest("pre, a, .tb-ai-metric-link")) continue;

    const text = node.textContent;
    regex.lastIndex = 0;
    if (!regex.test(text)) continue;
    regex.lastIndex = 0;

    const frag = document.createDocumentFragment();
    let lastIdx = 0;
    let match;
    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIdx) {
        frag.appendChild(document.createTextNode(text.slice(lastIdx, match.index)));
      }
      const span = document.createElement("span");
      span.className = "tb-ai-metric-link";
      span.dataset.tag = match[1];
      span.textContent = match[1];
      span.title = 'Click to view "' + match[1] + '" chart';
      frag.appendChild(span);
      lastIdx = regex.lastIndex;
    }
    if (lastIdx < text.length) {
      frag.appendChild(document.createTextNode(text.slice(lastIdx)));
    }
    node.parentNode.replaceChild(frag, node);
  }
}

const DEFAULT_PROMPT =
  "Provide a comprehensive analysis of all training metrics. " +
  "Identify key trends, potential issues, and actionable insights. " +
  "Pay special attention to cross-metric correlations at event steps — " +
  "e.g., did grad_norm spike when reward dropped? Did loss plateau when learning_rate decayed?";

async function sendMessage() {
  let text = inputEl.value.trim();
  if (isStreaming) return;
  // If empty and first message, use default prompt
  if (!text) {
    if (conversationHistory.length > 0) return;
    text = DEFAULT_PROMPT;
  }
  inputEl.value = "";

  const isFirst = conversationHistory.length === 0;

  let content = text;
  let ctx = panelEl.dataset.context;
  // If no context yet, auto-build from selected metrics
  if (isFirst && !ctx) {
    ctx = buildFullDataContext();
    panelEl.dataset.context = ctx;
  }
  // Auto-generate title from selected metrics on first message
  if (isFirst) {
    const tags = getSelectedTagNames();
    if (tags.length > 0) {
      const MAX_LEN = 40;
      let title = tags[0];
      let i = 1;
      while (i < tags.length && (title + ", " + tags[i]).length <= MAX_LEN) {
        title += ", " + tags[i];
        i++;
      }
      if (i < tags.length) title += " +" + (tags.length - i);
      titleEl.textContent = title;
    }
  }
  if (isFirst && ctx) {
    content = "Here is the data I want you to analyze:\n\n" + ctx + "\n\nUser request: " + text;
  }

  if (isFirst) {
    appendDebugBlock(content);
  } else {
    appendMessage("user", text);
  }

  // Disable input during first analysis
  isStreaming = true;
  if (isFirstMessage) {
    setInputEnabled(false);
  }

  conversationHistory.push({ role: "user", content });

  const assistantEl = appendMessage("assistant", "");
  await streamResponse(assistantEl);

  isStreaming = false;
  if (isFirstMessage) {
    isFirstMessage = false;
    setInputEnabled(true);
    inputEl.focus();
  }

  // Auto-save to history after each response
  saveCurrentToHistory();
}

async function streamResponse(assistantEl) {
  try {
    const response = await fetch(CHAT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: conversationHistory,
        provider: chatSettings.provider || undefined,
        language: chatSettings.language || undefined,
        model: chatSettings.model || undefined,
        use_reasoning: chatSettings.useReasoning || undefined,
      }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let event;
        try {
          event = JSON.parse(line.slice(6));
        } catch {
          continue;
        }

        if (event.type === "text") {
          fullText += event.content;
          renderMarkdown(assistantEl, fullText);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (event.type === "error") {
          assistantEl.textContent = "Error: " + event.content;
        } else if (event.type === "done") {
          if (event.usage) {
            appendUsageInfo(event.usage);
          }
        }
      }
    }

    if (fullText) {
      conversationHistory.push({ role: "assistant", content: fullText });
    }
  } catch (err) {
    assistantEl.textContent = "Connection error: " + err.message;
  }
}
