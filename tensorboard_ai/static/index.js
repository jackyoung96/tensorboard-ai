/**
 * TensorBoard AI Plugin — dashboard matching TIME SERIES layout.
 */
import { initChatPanel, openChat, setKnownTags, setChatSettings } from "./chat.js";
import {
  loadPlotly,
  renderChart,
  assignRunColors,
  summarizeSeries,
  summarizeTagGroups,
  buildCrossMetricContext,
  buildRunAliases,
} from "./chart_renderer.js";

const SCALARS_ENDPOINT = "./scalars";

/** State */
let allScalars = {};
let tagGroups = {};
let runColors = {};
let enabledRuns = new Set();
let tagFilter = "";
let runFilter = "";
let smoothing = 0.6;
let xAxisMode = "step";

export async function render() {
  const root = document.createElement("div");
  root.id = "tb-ai-root";
  document.body.appendChild(root);

  const style = document.createElement("style");
  style.textContent = CSS;
  document.head.appendChild(style);

  root.innerHTML = LAYOUT_HTML;
  rootEl = root;

  initChatPanel(root.querySelector("#tb-ai-chat-panel"));

  // Scroll-to-chart from chat metric links
  document.addEventListener("tb-ai-scroll-to-chart", (e) => {
    console.log("[tb-ai] scroll-to-chart event:", e.detail.tag);
    scrollToChart(e.detail.tag);
  });

  // Analyze All
  root.querySelector("#tb-ai-analyze-all").addEventListener("click", () => {
    const visibleGroups = getVisibleTagGroups();
    // Build run aliases to compress long run names
    const allRuns = new Set();
    for (const series of Object.values(visibleGroups)) {
      for (const { run } of series) allRuns.add(run);
    }
    const { aliases, legend } = buildRunAliases([...allRuns]);
    const parts = [];
    if (legend) parts.push(legend);
    parts.push(summarizeTagGroups(visibleGroups, aliases));
    const crossMetric = buildCrossMetricContext(visibleGroups, aliases);
    if (crossMetric) parts.push(crossMetric);
    if (legend) parts.push("NOTE: The data above uses short aliases for run names. In your response, always use the original full run names, not the aliases.");
    const dataContext = parts.join("\n\n");
    openChat({
      title: "Full Training Analysis",
      dataContext,
      autoMessage:
        "Provide a comprehensive analysis of all training metrics. " +
        "Identify key trends, potential issues, and actionable insights. " +
        "Pay special attention to cross-metric correlations at event steps — " +
        "e.g., did grad_norm spike when reward dropped? Did loss plateau when learning_rate decayed?",
    });
  });

  // Settings: Smoothing
  const smoothSlider = root.querySelector("#tb-ai-smoothing");
  const smoothValue = root.querySelector("#tb-ai-smoothing-val");
  smoothSlider.value = smoothing;
  smoothValue.textContent = smoothing.toFixed(2);
  smoothSlider.addEventListener("input", () => {
    smoothing = parseFloat(smoothSlider.value);
    smoothValue.textContent = smoothing.toFixed(2);
    rerenderCharts();
  });

  // Settings: X-axis
  root.querySelectorAll('input[name="tb-ai-xaxis"]').forEach((radio) => {
    if (radio.value === xAxisMode) radio.checked = true;
    radio.addEventListener("change", () => {
      xAxisMode = radio.value;
      rerenderCharts();
    });
  });

  // Analysis settings
  const langSelect = root.querySelector("#tb-ai-language");
  const modelSelect = root.querySelector("#tb-ai-model");
  const providerRadios = root.querySelectorAll('input[name="tb-ai-provider"]');
  const reasoningCb = root.querySelector("#tb-ai-reasoning");
  const debugCb = root.querySelector("#tb-ai-debug");

  // Detect browser language for "Auto" option
  const browserLang = (() => {
    const lang = (navigator.language || "").split("-")[0].toLowerCase();
    const map = { en: "English", ko: "Korean", ja: "Japanese", zh: "Chinese", es: "Spanish", fr: "French", de: "German" };
    return map[lang] || "";
  })();

  // Provider config will be populated from /config endpoint
  let providerConfig = {};
  let activeProvider = "";

  function populateModelDropdown(providerName) {
    modelSelect.innerHTML = "";
    const info = providerConfig[providerName];
    if (!info) return;
    for (const m of info.models) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      if (m === info.default_model) opt.selected = true;
      modelSelect.appendChild(opt);
    }
  }

  function syncAnalysisSettings() {
    setChatSettings({
      language: langSelect.value || browserLang,
      provider: activeProvider,
      model: modelSelect.value,
      useReasoning: reasoningCb.checked,
      debug: debugCb.checked,
    });
  }

  providerRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
      activeProvider = radio.value;
      populateModelDropdown(activeProvider);
      syncAnalysisSettings();
    });
  });
  langSelect.addEventListener("change", syncAnalysisSettings);
  modelSelect.addEventListener("change", syncAnalysisSettings);
  reasoningCb.addEventListener("change", syncAnalysisSettings);
  debugCb.addEventListener("change", syncAnalysisSettings);

  // Tag filter
  const tagInput = root.querySelector("#tb-ai-tag-filter");
  tagInput.addEventListener("input", () => {
    tagFilter = tagInput.value.toLowerCase();
    rerenderCharts();
  });

  // Run filter
  const runFilterInput = root.querySelector("#tb-ai-run-filter");
  runFilterInput.addEventListener("input", () => {
    runFilter = runFilterInput.value;
    filterRunCheckboxes();
  });

  // Select All / None
  root.querySelector("#tb-ai-runs-all").addEventListener("click", () => {
    toggleFilteredRuns(true);
  });
  root.querySelector("#tb-ai-runs-none").addEventListener("click", () => {
    toggleFilteredRuns(false);
  });

  // Resizable sidebar
  initResizeHandle(
    root.querySelector("#tb-ai-sidebar-resize"),
    root.querySelector(".tb-ai-sidebar"),
    "width",
    { min: 140, max: 500, side: "left" },
  );

  // Resizable chat panel
  initResizeHandle(
    root.querySelector("#tb-ai-chat-resize"),
    root.querySelector("#tb-ai-chat-panel"),
    "width",
    { min: 280, max: 800, side: "right" },
  );

  // Load config (provider/model info, debug flag)
  const debugRow = root.querySelector("#tb-ai-debug-row");
  try {
    const config = await fetch("./config").then((r) => r.ok ? r.json() : {});
    if (!config.debug) {
      debugRow.style.display = "none";
    }
    // Populate provider/model settings from server config
    if (config.providers) {
      providerConfig = config.providers;
      activeProvider = config.default_provider || "anthropic";
      // Check the matching radio
      providerRadios.forEach((radio) => {
        radio.checked = radio.value === activeProvider;
      });
      // If a specific model was set via env, override the default in the config
      if (config.default_model && providerConfig[activeProvider]) {
        providerConfig[activeProvider].default_model = config.default_model;
      }
      populateModelDropdown(activeProvider);
    }
  } catch {
    debugRow.style.display = "none";
  }
  syncAnalysisSettings();

  // Load data
  await loadPlotly();
  await fetchAndRender(root);
}

async function fetchAndRender(root) {
  try {
    const data = await fetch(SCALARS_ENDPOINT).then((r) => {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
    allScalars = data;

    const allRuns = Object.keys(data);
    // Preserve existing run colors, assign new ones for new runs
    runColors = assignRunColors(allRuns);
    // Add new runs to enabledRuns (keep disabled runs disabled)
    const prevRuns = new Set(allRunLabels.map((r) => r.run));
    if (prevRuns.size === 0) {
      allRuns.forEach((r) => enabledRuns.add(r));
    } else {
      for (const r of allRuns) {
        if (!prevRuns.has(r)) enabledRuns.add(r);
      }
    }

    const allTags = new Set();
    for (const tags of Object.values(data)) {
      Object.keys(tags).forEach((t) => allTags.add(t));
    }
    setKnownTags([...allTags]);

    buildRunCheckboxes(root.querySelector("#tb-ai-runs-list"), allRuns);

    rebuildTagGroups();
    rerenderCharts();
  } catch (err) {
    (rootEl || document).querySelector("#tb-ai-grid").innerHTML =
      '<div class="tb-ai-empty">Failed to load data: ' + (err.message || err) + "</div>";
  }
}

/** Reload data from the server and re-render charts. */
export async function reload() {
  if (rootEl) await fetchAndRender(rootEl);
}

/* ── Resize handle ── */

function initResizeHandle(handle, target, prop, opts) {
  if (!handle || !target) return;
  const { min, max, side } = opts;

  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = target.getBoundingClientRect().width;

    // Disable CSS transitions during drag so they don't fight with mouse
    target.classList.add("resizing");

    function onMove(ev) {
      let delta = ev.clientX - startX;
      if (side === "right") delta = -delta;
      const newW = Math.min(max, Math.max(min, startW + delta));
      target.style.width = newW + "px";
      target.style.minWidth = newW + "px";
    }

    function onUp() {
      target.classList.remove("resizing");
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}

/* ── State helpers ── */

function rebuildTagGroups() {
  tagGroups = {};
  for (const [run, tags] of Object.entries(allScalars)) {
    if (!enabledRuns.has(run)) continue;
    for (const [tag, points] of Object.entries(tags)) {
      if (!tagGroups[tag]) tagGroups[tag] = [];
      tagGroups[tag].push({ run, points, color: runColors[run] });
    }
  }
}

function getVisibleTagGroups() {
  if (!tagFilter) return tagGroups;
  const filtered = {};
  for (const [tag, series] of Object.entries(tagGroups)) {
    if (tag.toLowerCase().includes(tagFilter)) filtered[tag] = series;
  }
  return filtered;
}

let rootEl = null;

function scrollToChart(tag) {
  const container = rootEl || document;
  const main = container.querySelector(".tb-ai-main");
  if (!main) {
    console.warn("[tb-ai] .tb-ai-main not found");
    return;
  }
  const cards = main.querySelectorAll(".tb-ai-card");
  console.log("[tb-ai] looking for tag:", tag, "among", cards.length, "cards");
  for (const c of cards) {
    if (c.dataset.tag === tag) {
      const mainRect = main.getBoundingClientRect();
      const cardRect = c.getBoundingClientRect();
      const targetTop = main.scrollTop + (cardRect.top - mainRect.top) - 12;
      console.log("[tb-ai] scrolling .tb-ai-main to", targetTop,
        "| main.scrollHeight:", main.scrollHeight,
        "| main.clientHeight:", main.clientHeight,
        "| main.scrollTop:", main.scrollTop);
      main.scrollTo({ top: targetTop, behavior: "smooth" });
      c.classList.add("tb-ai-card-highlight");
      setTimeout(() => c.classList.remove("tb-ai-card-highlight"), 2000);
      return;
    }
  }
  console.warn("[tb-ai] card not found for tag:", tag);
}

/* ── Runs ── */

let allRunLabels = []; // { run, label, checkbox }

function buildRunCheckboxes(container, runs) {
  container.innerHTML = "";
  allRunLabels = [];
  const sorted = [...runs].sort();
  for (const run of sorted) {
    const label = document.createElement("label");
    label.className = "tb-ai-run-label";
    label.dataset.run = run;

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", () => {
      if (cb.checked) enabledRuns.add(run);
      else enabledRuns.delete(run);
      rerenderCharts();
    });

    const swatch = document.createElement("span");
    swatch.className = "tb-ai-run-swatch";
    swatch.style.background = runColors[run];

    const name = document.createElement("span");
    name.className = "tb-ai-run-name";
    name.textContent = run;

    label.appendChild(cb);
    label.appendChild(swatch);
    label.appendChild(name);
    container.appendChild(label);

    allRunLabels.push({ run, label, checkbox: cb });
  }
}

function filterRunCheckboxes() {
  let regex = null;
  try {
    if (runFilter) regex = new RegExp(runFilter, "i");
  } catch {
    // invalid regex, treat as plain substring
  }

  for (const { run, label } of allRunLabels) {
    const visible = !runFilter || (regex ? regex.test(run) : run.toLowerCase().includes(runFilter.toLowerCase()));
    label.style.display = visible ? "" : "none";
  }
}

function toggleFilteredRuns(enable) {
  let regex = null;
  try {
    if (runFilter) regex = new RegExp(runFilter, "i");
  } catch {}

  for (const { run, label, checkbox } of allRunLabels) {
    const visible = !runFilter || (regex ? regex.test(run) : run.toLowerCase().includes(runFilter.toLowerCase()));
    if (!visible) continue;
    checkbox.checked = enable;
    if (enable) enabledRuns.add(run);
    else enabledRuns.delete(run);
  }
  rerenderCharts();
}

/* ── Chart rendering ── */

function rerenderCharts() {
  rebuildTagGroups();
  const grid = (rootEl || document).querySelector("#tb-ai-grid");
  grid.innerHTML = "";

  const visible = getVisibleTagGroups();
  const tags = Object.keys(visible).sort();

  if (tags.length === 0) {
    grid.innerHTML = '<div class="tb-ai-empty">No matching metrics.</div>';
    return;
  }

  for (const tag of tags) {
    const series = visible[tag];
    const card = document.createElement("div");
    card.className = "tb-ai-card";
    card.dataset.tag = tag;

    const chartDiv = document.createElement("div");
    chartDiv.className = "tb-ai-card-chart";
    card.appendChild(chartDiv);

    const bar = document.createElement("div");
    bar.className = "tb-ai-card-bar";
    const aiBtn = document.createElement("button");
    aiBtn.className = "tb-ai-card-btn";
    aiBtn.textContent = "AI";
    aiBtn.title = 'Analyze "' + tag + '"';
    aiBtn.addEventListener("click", () => {
      const runs = series.map((s) => s.run);
      const { aliases: a, legend: l } = buildRunAliases(runs);
      const ctxParts = [];
      if (l) ctxParts.push(l);
      ctxParts.push(summarizeSeries(tag, series, a));
      if (l) ctxParts.push("NOTE: The data above uses short aliases for run names. In your response, always use the original full run names, not the aliases.");
      const ctx = ctxParts.join("\n\n");
      openChat({
        title: tag,
        dataContext: ctx,
        autoMessage:
          'Analyze the "' + tag + '" metric. Describe the trend, identify any anomalies, and suggest whether training looks healthy.',
      });
    });
    bar.appendChild(aiBtn);
    card.appendChild(bar);
    grid.appendChild(card);

    renderChart(chartDiv, tag, series, { smoothing, xAxisMode });
  }
}

/* ── Layout HTML ── */
const LAYOUT_HTML = `
  <div class="tb-ai-layout">
    <aside class="tb-ai-sidebar">
      <!-- Chart Settings -->
      <div class="tb-ai-sidebar-section">
        <div class="tb-ai-sidebar-title">Chart Settings</div>
        <label class="tb-ai-setting-row">
          <span class="tb-ai-setting-label">Smoothing</span>
          <input id="tb-ai-smoothing" type="range" min="0" max="0.999" step="0.001" />
          <span id="tb-ai-smoothing-val" class="tb-ai-setting-value">0.60</span>
        </label>
        <div class="tb-ai-setting-group">
          <span class="tb-ai-setting-label">Horizontal Axis</span>
          <label class="tb-ai-radio"><input type="radio" name="tb-ai-xaxis" value="step" /> Step</label>
          <label class="tb-ai-radio"><input type="radio" name="tb-ai-xaxis" value="relative" /> Relative</label>
          <label class="tb-ai-radio"><input type="radio" name="tb-ai-xaxis" value="wall" /> Wall</label>
        </div>
      </div>

      <!-- Analysis Settings -->
      <div class="tb-ai-sidebar-section">
        <div class="tb-ai-sidebar-title">Analysis Settings</div>
        <div class="tb-ai-setting-group">
          <span class="tb-ai-setting-label">Language</span>
          <select id="tb-ai-language" class="tb-ai-select">
            <option value="">Auto (browser)</option>
            <option value="English">English</option>
            <option value="Korean">Korean</option>
            <option value="Japanese">Japanese</option>
            <option value="Chinese">Chinese</option>
            <option value="Spanish">Spanish</option>
            <option value="French">French</option>
            <option value="German">German</option>
          </select>
        </div>
        <div class="tb-ai-setting-group">
          <span class="tb-ai-setting-label">Provider</span>
          <label class="tb-ai-radio"><input type="radio" name="tb-ai-provider" value="anthropic" /> Anthropic</label>
          <label class="tb-ai-radio"><input type="radio" name="tb-ai-provider" value="openai" /> OpenAI</label>
          <label class="tb-ai-radio"><input type="radio" name="tb-ai-provider" value="bedrock" /> Bedrock</label>
        </div>
        <div class="tb-ai-setting-group">
          <span class="tb-ai-setting-label">Model</span>
          <select id="tb-ai-model" class="tb-ai-select"></select>
        </div>
        <label class="tb-ai-checkbox-row">
          <input id="tb-ai-reasoning" type="checkbox" />
          <span>Extended thinking</span>
        </label>
        <label id="tb-ai-debug-row" class="tb-ai-checkbox-row" style="margin-top:4px;">
          <input id="tb-ai-debug" type="checkbox" />
          <span>Debug (show raw prompt)</span>
        </label>
      </div>

      <!-- Runs -->
      <div class="tb-ai-sidebar-section tb-ai-runs-section">
        <div class="tb-ai-sidebar-title">
          Runs
          <span class="tb-ai-runs-actions">
            <button id="tb-ai-runs-all" class="tb-ai-link-btn">All</button>
            <button id="tb-ai-runs-none" class="tb-ai-link-btn">None</button>
          </span>
        </div>
        <input id="tb-ai-run-filter" type="text" placeholder="Filter runs (regex)"
               class="tb-ai-tag-input" style="margin-bottom:6px;" />
        <div id="tb-ai-runs-list" class="tb-ai-runs-list"></div>
      </div>

      <!-- Tag filter -->
      <div class="tb-ai-sidebar-section">
        <div class="tb-ai-sidebar-title">Tag Filter</div>
        <input id="tb-ai-tag-filter" type="text" placeholder="Filter tags"
               class="tb-ai-tag-input" />
      </div>
    </aside>

    <div id="tb-ai-sidebar-resize" class="tb-ai-resize-handle"></div>

    <main class="tb-ai-main">
      <div class="tb-ai-toolbar">
        <button id="tb-ai-analyze-all" class="tb-ai-analyze-all">Analyze All</button>
      </div>
      <div id="tb-ai-grid" class="tb-ai-grid">
        <div class="tb-ai-empty">Loading metrics...</div>
      </div>
    </main>

    <div id="tb-ai-chat-resize" class="tb-ai-resize-handle tb-ai-resize-handle-chat"></div>
    <div class="tb-ai-chat-panel" id="tb-ai-chat-panel"></div>
  </div>
`;

/* ── Styles ── */
const CSS = `
  html, body {
    height: 100%;
    margin: 0;
    overflow: hidden;
  }
  #tb-ai-root {
    height: 100%;
    font-family: 'Roboto', 'Google Sans', Arial, sans-serif;
    font-size: 13px;
    color: #333;
  }

  .tb-ai-layout {
    display: flex;
    height: 100%;
    overflow: hidden;
  }

  /* ── Resize handles ── */
  .tb-ai-resize-handle {
    width: 5px;
    cursor: col-resize;
    background: transparent;
    flex-shrink: 0;
    position: relative;
    z-index: 10;
    transition: background 0.15s;
  }
  .tb-ai-resize-handle:hover,
  .tb-ai-resize-handle:active {
    background: #ff6f00;
    opacity: 0.5;
  }
  .tb-ai-resize-handle-chat {
  }

  /* ── Sidebar ── */
  .tb-ai-sidebar {
    width: 220px;
    min-width: 140px;
    background: #fff;
    border-right: 1px solid #e0e0e0;
    overflow-y: auto;
    padding: 12px 0;
    flex-shrink: 0;
  }
  .tb-ai-sidebar.resizing { transition: none; }
  .tb-ai-sidebar-section {
    padding: 0 14px 12px;
    border-bottom: 1px solid #eee;
    margin-bottom: 4px;
  }
  .tb-ai-sidebar-section:last-child { border-bottom: none; }
  .tb-ai-sidebar-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: #666;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    padding-top: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  /* Settings controls */
  .tb-ai-setting-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
  }
  .tb-ai-setting-label {
    font-size: 12px;
    color: #555;
    display: block;
    margin-bottom: 4px;
  }
  .tb-ai-setting-value {
    font-size: 11px;
    color: #888;
    min-width: 32px;
    text-align: right;
  }
  #tb-ai-smoothing { flex: 1; height: 2px; accent-color: #ff6f00; }
  .tb-ai-setting-group { margin-bottom: 8px; }
  .tb-ai-radio {
    display: block;
    font-size: 12px;
    color: #555;
    padding: 2px 0;
    cursor: pointer;
  }
  .tb-ai-radio input { margin-right: 4px; accent-color: #ff6f00; }
  .tb-ai-select {
    width: 100%;
    padding: 4px 6px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 12px;
    background: #fff;
    outline: none;
  }
  .tb-ai-select:focus { border-color: #ff6f00; }
  .tb-ai-text-input {
    width: 100%;
    box-sizing: border-box;
    padding: 4px 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 12px;
    outline: none;
  }
  .tb-ai-text-input:focus { border-color: #ff6f00; }
  .tb-ai-checkbox-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #555;
    cursor: pointer;
  }
  .tb-ai-checkbox-row input { accent-color: #ff6f00; }

  /* Runs */
  .tb-ai-runs-section { flex: 1; min-height: 0; display: flex; flex-direction: column; }
  .tb-ai-runs-actions { display: flex; gap: 4px; }
  .tb-ai-link-btn {
    background: none;
    border: none;
    color: #ff6f00;
    font-size: 11px;
    cursor: pointer;
    padding: 0;
    font-weight: 500;
    text-transform: none;
    letter-spacing: 0;
  }
  .tb-ai-link-btn:hover { text-decoration: underline; }
  .tb-ai-runs-list {
    display: flex;
    flex-direction: column;
    gap: 3px;
    overflow-y: auto;
    flex: 1;
  }
  .tb-ai-run-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    cursor: pointer;
    padding: 2px 0;
  }
  .tb-ai-run-label input { margin: 0; accent-color: #ff6f00; }
  .tb-ai-run-swatch {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .tb-ai-run-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .tb-ai-tag-input {
    width: 100%;
    box-sizing: border-box;
    padding: 5px 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 12px;
    outline: none;
  }
  .tb-ai-tag-input:focus { border-color: #ff6f00; }

  /* ── Main ── */
  .tb-ai-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #f5f5f5;
    min-width: 0;
    overflow-y: auto;
  }
  .tb-ai-toolbar {
    display: flex;
    justify-content: flex-end;
    padding: 8px 16px;
    background: #fff;
    border-bottom: 1px solid #e0e0e0;
    flex-shrink: 0;
  }
  .tb-ai-analyze-all {
    padding: 6px 16px;
    background: #ff6f00;
    color: #fff;
    border: none;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
  }
  .tb-ai-analyze-all:hover { background: #e65100; }
  .tb-ai-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 12px;
    padding: 12px 16px;
    align-content: start;
  }
  .tb-ai-empty {
    grid-column: 1 / -1;
    text-align: center;
    padding: 48px 0;
    color: #999;
    font-size: 13px;
  }
  .tb-ai-card {
    background: #fff;
    border-radius: 4px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: box-shadow 0.3s, outline 0.3s;
  }
  .tb-ai-card-highlight {
    outline: 2px solid #ff6f00;
    box-shadow: 0 0 12px rgba(255,111,0,0.3);
  }
  .tb-ai-card-chart { padding: 6px 6px 0; min-height: 280px; }
  .tb-ai-card-bar { display: flex; justify-content: flex-end; padding: 2px 8px 6px; }
  .tb-ai-card-btn {
    padding: 3px 12px;
    background: #fff3e0;
    color: #e65100;
    border: 1px solid #ffcc80;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }
  .tb-ai-card-btn:hover { background: #ffe0b2; }

  /* ── Chat panel — always visible flex child ── */
  .tb-ai-chat-panel {
    width: 400px;
    min-width: 280px;
    background: #fff;
    border-left: 1px solid #e0e0e0;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
  }
  /* Disable transition during drag resize */
  .tb-ai-chat-panel.resizing {
    transition: none;
  }

  .tb-ai-chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid #e0e0e0;
    background: #f5f5f5;
    flex-shrink: 0;
    min-height: 20px;
  }
  .tb-ai-chat-title {
    font-weight: 600; font-size: 13px; color: #333;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .tb-ai-chat-header-right {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }
  .tb-ai-history-select {
    font-size: 11px;
    padding: 3px 6px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: #fff;
    max-width: 180px;
    outline: none;
  }
  .tb-ai-history-select:focus { border-color: #ff6f00; }
  .tb-ai-chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .tb-ai-chat-input-row {
    display: flex;
    padding: 8px 10px;
    border-top: 1px solid #e0e0e0;
    background: #f8f9fa;
    flex-shrink: 0;
  }
  .tb-ai-chat-input-row input {
    flex: 1; padding: 7px 10px; border: 1px solid #ccc;
    border-radius: 5px; font-size: 12px; outline: none;
  }
  .tb-ai-chat-input-row input:focus { border-color: #ff6f00; }
  .tb-ai-chat-input-row input.tb-ai-disabled {
    background: #f0f0f0; color: #999; cursor: not-allowed;
  }
  .tb-ai-chat-send {
    margin-left: 6px; padding: 7px 14px; background: #ff6f00; color: #fff;
    border: none; border-radius: 5px; cursor: pointer; font-size: 12px; font-weight: 500;
  }
  .tb-ai-chat-send:hover { background: #e65100; }
  .tb-ai-chat-send.tb-ai-disabled {
    background: #ccc; cursor: not-allowed;
  }

  /* Chat empty state */
  .tb-ai-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    color: #999;
    font-size: 12px;
    text-align: center;
    gap: 10px;
    padding: 20px;
  }
  .tb-ai-empty-icon {
    width: 40px; height: 40px;
    border-radius: 50%;
    background: #fff3e0;
    color: #ff6f00;
    font-weight: 700;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* Chat messages */
  .tb-ai-msg {
    max-width: 92%; padding: 8px 12px; border-radius: 10px;
    font-size: 13px; line-height: 1.5; word-break: break-word;
    flex-shrink: 0;
  }
  .tb-ai-msg-user { align-self: flex-end; background: #e3f2fd; white-space: pre-wrap; }
  .tb-ai-msg-assistant { align-self: flex-start; background: #f5f5f5; }

  /* Token usage info */
  .tb-ai-usage-info {
    align-self: flex-start;
    font-size: 10px;
    color: #999;
    padding: 2px 8px;
    margin-top: -6px;
    flex-shrink: 0;
  }

  /* Debug block */
  .tb-ai-debug-block {
    align-self: stretch;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    overflow: hidden;
    font-size: 11px;
    background: #fafafa;
    flex-shrink: 0;
  }
  .tb-ai-debug-header {
    background: #f0f0f0;
    padding: 4px 10px;
    font-weight: 600;
    color: #666;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    border-bottom: 1px solid #e0e0e0;
  }
  .tb-ai-debug-pre {
    margin: 0;
    padding: 8px 10px;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 300px;
    overflow-y: auto;
    font-family: 'Roboto Mono', monospace;
    font-size: 10px;
    line-height: 1.5;
    color: #444;
    background: #fafafa;
  }

  /* Markdown in assistant messages */
  .tb-ai-msg-assistant h1, .tb-ai-msg-assistant h2, .tb-ai-msg-assistant h3 {
    margin: 8px 0 4px; color: #222;
  }
  .tb-ai-msg-assistant h1 { font-size: 16px; }
  .tb-ai-msg-assistant h2 { font-size: 14px; }
  .tb-ai-msg-assistant h3 { font-size: 13px; }
  .tb-ai-msg-assistant p { margin: 4px 0; }
  .tb-ai-msg-assistant ul, .tb-ai-msg-assistant ol { margin: 4px 0; padding-left: 20px; }
  .tb-ai-msg-assistant li { margin: 2px 0; }
  .tb-ai-msg-assistant code {
    background: #e8e8e8; padding: 1px 4px; border-radius: 3px;
    font-size: 12px; font-family: 'Roboto Mono', monospace;
  }
  .tb-ai-msg-assistant pre {
    background: #263238; color: #eee; padding: 10px 12px;
    border-radius: 6px; overflow-x: auto; font-size: 12px; margin: 6px 0;
  }
  .tb-ai-msg-assistant pre code { background: none; padding: 0; color: inherit; }
  .tb-ai-msg-assistant strong { font-weight: 600; }
  .tb-ai-msg-assistant blockquote {
    border-left: 3px solid #ff6f00; margin: 6px 0; padding: 2px 10px; color: #555;
  }
  .tb-ai-msg-assistant table { border-collapse: collapse; margin: 6px 0; font-size: 12px; }
  .tb-ai-msg-assistant th, .tb-ai-msg-assistant td { border: 1px solid #ddd; padding: 4px 8px; }
  .tb-ai-msg-assistant th { background: #f0f0f0; font-weight: 600; }

  /* Metric links */
  .tb-ai-metric-link {
    color: #e65100; font-weight: 500; cursor: pointer;
    border-bottom: 1px dashed #ffcc80; transition: color 0.15s;
  }
  .tb-ai-metric-link:hover { color: #ff6f00; border-bottom-color: #ff6f00; }
`;
