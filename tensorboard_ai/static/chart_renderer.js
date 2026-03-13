/**
 * Chart renderer — Plotly chart rendering with smoothing and axis options.
 */

let plotlyReady = null;

export function loadPlotly() {
  if (plotlyReady) return plotlyReady;
  plotlyReady = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "./plotly-basic.min.js";
    script.onload = resolve;
    script.onerror = () => reject(new Error("Failed to load Plotly library"));
    document.head.appendChild(script);
  });
  return plotlyReady;
}

// TensorBoard-like color palette
const COLORS = [
  "#e377c2", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
  "#9467bd", "#8c564b", "#7f7f7f", "#bcbd22", "#17becf",
];

/**
 * Apply exponential moving average smoothing (same algorithm as TensorBoard).
 */
function smooth(values, weight) {
  if (weight <= 0) return values;
  const smoothed = [];
  let last = values[0];
  let numAccum = 0;
  for (let i = 0; i < values.length; i++) {
    const nextVal = values[i];
    last = last * weight + (1 - weight) * nextVal;
    numAccum++;
    // Bias correction
    const debiased = last / (1 - Math.pow(weight, numAccum));
    smoothed.push(debiased);
  }
  return smoothed;
}

/**
 * Get x-axis values based on the selected mode.
 */
function getXValues(points, xAxisMode) {
  if (xAxisMode === "relative") {
    const t0 = points[0]?.wall_time || 0;
    return points.map((p) => (p.wall_time - t0) / 3600); // hours
  }
  if (xAxisMode === "wall") {
    return points.map((p) => new Date(p.wall_time * 1000));
  }
  return points.map((p) => p.step);
}

function getXAxisTitle(xAxisMode) {
  if (xAxisMode === "relative") return "Time (hours)";
  if (xAxisMode === "wall") return "Wall Time";
  return "Step";
}

/**
 * Render a chart into the given container.
 * @param {HTMLElement} container
 * @param {string} tag
 * @param {Array<{run: string, points: Array, color: string}>} series
 * @param {object} opts - { smoothing, xAxisMode }
 */
export function renderChart(container, tag, series, opts = {}) {
  const { smoothing = 0, xAxisMode = "step" } = opts;
  const traces = [];

  for (const { run, points, color } of series) {
    if (!points.length) continue;
    const xVals = getXValues(points, xAxisMode);
    const yVals = points.map((p) => p.value);
    const ySmoothed = smooth(yVals, smoothing);

    // Original (faded) when smoothing is applied
    if (smoothing > 0) {
      traces.push({
        x: xVals,
        y: yVals,
        type: "scatter",
        mode: "lines",
        name: run,
        line: { color, width: 1 },
        opacity: 0.25,
        showlegend: false,
        hoverinfo: "skip",
      });
    }

    // Smoothed (or original if no smoothing)
    traces.push({
      x: xVals,
      y: smoothing > 0 ? ySmoothed : yVals,
      type: "scatter",
      mode: "lines",
      name: run,
      line: { color, width: 1.5 },
    });
  }

  const layout = {
    title: { text: tag, font: { size: 13, color: "#333" }, x: 0.02, xanchor: "left" },
    xaxis: {
      title: { text: getXAxisTitle(xAxisMode), font: { size: 11 } },
      gridcolor: "#ececec",
      zeroline: false,
    },
    yaxis: {
      gridcolor: "#ececec",
      zeroline: false,
      tickfont: { size: 10 },
    },
    margin: { t: 32, r: 12, b: 36, l: 52 },
    legend: { orientation: "h", y: -0.22, font: { size: 10 } },
    height: 280,
    plot_bgcolor: "#fff",
    paper_bgcolor: "#fff",
    hovermode: "x unified",
  };

  /* global Plotly */
  Plotly.newPlot(container, traces, layout, {
    responsive: true,
    displayModeBar: false,
  });
}

/**
 * Assign colors to runs (consistent ordering).
 */
export function assignRunColors(runs) {
  const map = {};
  const sorted = [...runs].sort();
  sorted.forEach((run, i) => {
    map[run] = COLORS[i % COLORS.length];
  });
  return map;
}

/**
 * Smart sampling: captures overall trend, spikes/drops, AND slope transitions.
 *
 * Three detection layers:
 *  1. Uniform grid (~20 pts) — overall trend skeleton
 *  2. Value anomalies — spikes/drops deviating from local mean
 *  3. Curvature / slope-change — points where the gradient changes sharply
 *     (e.g., loss starts plateauing, reward begins diverging)
 *
 * All layers merged, sorted by step, capped at MAX_TOTAL.
 */
const UNIFORM_COUNT = 20;
const MAX_TOTAL = 50;
const ANOMALY_THRESHOLD = 1.5; // std deviations for value anomalies
const CURVATURE_THRESHOLD = 1.5; // std deviations for slope changes

function smartSample(points) {
  if (points.length <= MAX_TOTAL) return points;

  const values = points.map((p) => p.value);
  const steps = points.map((p) => p.step);
  const n = values.length;

  // ── 1. Uniform grid ──
  const uniformIdx = new Set([0, n - 1]);
  const uStep = (n - 1) / (UNIFORM_COUNT - 1);
  for (let i = 1; i < UNIFORM_COUNT - 1; i++) {
    uniformIdx.add(Math.round(i * uStep));
  }

  // ── 2. Value anomalies (deviation from local moving average) ──
  const windowHalf = 2;
  const residuals = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - windowHalf);
    const hi = Math.min(n - 1, i + windowHalf);
    let sum = 0;
    for (let j = lo; j <= hi; j++) sum += values[j];
    residuals[i] = Math.abs(values[i] - sum / (hi - lo + 1));
  }

  const rMean = residuals.reduce((a, b) => a + b, 0) / n;
  const rStd = Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / n - rMean * rMean) || 1;
  const valThreshold = rMean + ANOMALY_THRESHOLD * rStd;

  // ── 3. Slope-change / curvature detection ──
  // Smooth values first with EMA to suppress noise before differentiation.
  // Without this, raw high-frequency noise makes every point look like a
  // slope change, defeating the purpose of curvature detection.
  const emaWeight = Math.min(0.85, 2 / (Math.max(10, n / 20) + 1));
  const smoothed = new Float64Array(n);
  smoothed[0] = values[0];
  for (let i = 1; i < n; i++) {
    smoothed[i] = emaWeight * smoothed[i - 1] + (1 - emaWeight) * values[i];
  }

  // First derivative on smoothed curve
  const grad = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    const dx = steps[i] - steps[i - 1] || 1;
    grad[i] = (smoothed[i] - smoothed[i - 1]) / dx;
  }
  grad[0] = grad[1];

  // Second derivative (curvature) on smoothed curve
  const curvature = new Float64Array(n);
  for (let i = 1; i < n - 1; i++) {
    const dx = (steps[i + 1] - steps[i - 1]) / 2 || 1;
    curvature[i] = Math.abs(grad[i + 1] - grad[i]) / dx;
  }

  const cMean = curvature.reduce((a, b) => a + b, 0) / n;
  const cStd = Math.sqrt(curvature.reduce((a, b) => a + b * b, 0) / n - cMean * cMean) || 1;
  const curveThreshold = cMean + CURVATURE_THRESHOLD * cStd;

  // ── Collect important indices ──
  const importantIdx = new Set();
  for (let i = 1; i < n - 1; i++) {
    // Value spike/drop
    if (residuals[i] > valThreshold) {
      importantIdx.add(i);
      continue;
    }
    // Significant local peak/valley
    const isPeak = values[i] > values[i - 1] && values[i] > values[i + 1];
    const isValley = values[i] < values[i - 1] && values[i] < values[i + 1];
    if ((isPeak || isValley) && residuals[i] > rMean + 0.5 * rStd) {
      importantIdx.add(i);
      continue;
    }
    // Slope transition (high curvature)
    if (curvature[i] > curveThreshold) {
      importantIdx.add(i);
    }
  }

  // ── Merge and cap ──
  const allIdx = new Set([...uniformIdx, ...importantIdx]);

  if (allIdx.size <= MAX_TOTAL) {
    return [...allIdx].sort((a, b) => a - b).map((i) => points[i]);
  }

  // Too many: rank important points by a combined score, keep top ones
  const importantArr = [...importantIdx].map((i) => ({
    i,
    score: residuals[i] / (rStd || 1) + curvature[i] / (cStd || 1),
  }));
  importantArr.sort((a, b) => b.score - a.score);
  const kept = importantArr.slice(0, MAX_TOTAL - uniformIdx.size).map((x) => x.i);
  const capped = new Set([...uniformIdx, ...kept]);
  return [...capped].sort((a, b) => a - b).map((i) => points[i]);
}

/**
 * Detect notable events (anomalies, peaks/valleys, slope changes) in a series.
 * Returns array of { step, type, value, score }.
 */
export function detectEvents(points) {
  if (!points.length) return [];

  const values = points.map((p) => p.value);
  const steps = points.map((p) => p.step);
  const n = values.length;
  if (n < 5) return [];

  // Value anomalies (deviation from local moving average)
  const windowHalf = 2;
  const residuals = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - windowHalf);
    const hi = Math.min(n - 1, i + windowHalf);
    let sum = 0;
    for (let j = lo; j <= hi; j++) sum += values[j];
    residuals[i] = Math.abs(values[i] - sum / (hi - lo + 1));
  }

  const rMean = residuals.reduce((a, b) => a + b, 0) / n;
  const rStd = Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / n - rMean * rMean) || 1;
  const valThreshold = rMean + ANOMALY_THRESHOLD * rStd;

  // EMA smoothing before differentiation
  const emaWeight = Math.min(0.85, 2 / (Math.max(10, n / 20) + 1));
  const smoothed = new Float64Array(n);
  smoothed[0] = values[0];
  for (let i = 1; i < n; i++) {
    smoothed[i] = emaWeight * smoothed[i - 1] + (1 - emaWeight) * values[i];
  }

  const grad = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    const dx = steps[i] - steps[i - 1] || 1;
    grad[i] = (smoothed[i] - smoothed[i - 1]) / dx;
  }
  grad[0] = grad[1];

  const curvature = new Float64Array(n);
  for (let i = 1; i < n - 1; i++) {
    const dx = (steps[i + 1] - steps[i - 1]) / 2 || 1;
    curvature[i] = Math.abs(grad[i + 1] - grad[i]) / dx;
  }

  const cMean = curvature.reduce((a, b) => a + b, 0) / n;
  const cStd = Math.sqrt(curvature.reduce((a, b) => a + b * b, 0) / n - cMean * cMean) || 1;
  const curveThreshold = cMean + CURVATURE_THRESHOLD * cStd;

  const events = [];
  for (let i = 1; i < n - 1; i++) {
    if (residuals[i] > valThreshold) {
      events.push({ step: steps[i], type: "anomaly", value: values[i], score: residuals[i] / rStd });
    } else if (curvature[i] > curveThreshold) {
      events.push({ step: steps[i], type: "slope_change", value: values[i], score: curvature[i] / cStd });
    } else {
      const isPeak = values[i] > values[i - 1] && values[i] > values[i + 1];
      const isValley = values[i] < values[i - 1] && values[i] < values[i + 1];
      if ((isPeak || isValley) && residuals[i] > rMean + 0.5 * rStd) {
        events.push({
          step: steps[i],
          type: isPeak ? "peak" : "valley",
          value: values[i],
          score: residuals[i] / rStd,
        });
      }
    }
  }
  return events;
}

/**
 * Interpolate a metric's value at a given step (nearest-neighbor from sorted points).
 */
function valueAtStep(points, targetStep) {
  if (!points.length) return null;
  let best = 0;
  let bestDist = Math.abs(points[0].step - targetStep);
  for (let i = 1; i < points.length; i++) {
    const d = Math.abs(points[i].step - targetStep);
    if (d < bestDist) { best = i; bestDist = d; }
    if (points[i].step > targetStep) break;
  }
  // Only match if within 5% of the step range
  const range = points[points.length - 1].step - points[0].step || 1;
  if (bestDist > range * 0.05) return null;
  return points[best].value;
}

/**
 * Build cross-metric correlation context for AI analysis.
 * Collects events from all metrics, clusters nearby steps, then creates a
 * snapshot table showing what every metric was doing at each event cluster.
 *
 * @param {Object<string, Array<{run: string, points: Array}>>} tagGroups
 * @returns {string} Text context for the AI prompt
 */
export function buildCrossMetricContext(tagGroups, runAliases) {
  const tags = Object.keys(tagGroups);
  if (tags.length < 2) return "";

  const alias = (r) => (runAliases && runAliases[r]) || r;

  // Collect all events across all metrics
  const allEvents = [];
  for (const tag of tags) {
    const series = tagGroups[tag];
    for (const { run, points } of series) {
      const events = detectEvents(points);
      for (const ev of events) {
        allEvents.push({ ...ev, tag, run });
      }
    }
  }

  if (!allEvents.length) return "";

  // Cluster nearby steps (within 3% of overall step range)
  allEvents.sort((a, b) => a.step - b.step);
  const allSteps = allEvents.map((e) => e.step);
  const globalMin = allSteps[0];
  const globalMax = allSteps[allSteps.length - 1];
  const clusterRadius = (globalMax - globalMin) * 0.03 || 1;

  const clusters = [];
  let currentCluster = { steps: [allEvents[0].step], events: [allEvents[0]] };
  for (let i = 1; i < allEvents.length; i++) {
    const ev = allEvents[i];
    const clusterMean = currentCluster.steps.reduce((a, b) => a + b, 0) / currentCluster.steps.length;
    if (ev.step - clusterMean <= clusterRadius) {
      currentCluster.steps.push(ev.step);
      currentCluster.events.push(ev);
    } else {
      clusters.push(currentCluster);
      currentCluster = { steps: [ev.step], events: [ev] };
    }
  }
  clusters.push(currentCluster);

  // Only keep clusters with events from 2+ different tags or high-score events
  const interestingClusters = clusters.filter((c) => {
    const uniqueTags = new Set(c.events.map((e) => e.tag));
    return uniqueTags.size >= 2 || c.events.some((e) => e.score > 2.5);
  });

  if (!interestingClusters.length) return "";

  // Cap at 15 most interesting clusters
  const scored = interestingClusters.map((c) => ({
    ...c,
    totalScore: c.events.reduce((s, e) => s + e.score, 0),
    centerStep: Math.round(c.steps.reduce((a, b) => a + b, 0) / c.steps.length),
  }));
  scored.sort((a, b) => b.totalScore - a.totalScore);
  const topClusters = scored.slice(0, 15).sort((a, b) => a.centerStep - b.centerStep);

  const lines = ["Cross-Metric Events:"];

  for (const cluster of topClusters) {
    const step = cluster.centerStep;
    const triggerDescs = cluster.events
      .map((e) => `${e.tag}/${alias(e.run)}:${e.type}(${fmt(e.value)})`)
      .join("; ");
    lines.push(`s${step} | ${triggerDescs}`);

    const snapshots = [];
    for (const tag of tags) {
      for (const { run, points } of tagGroups[tag]) {
        const val = valueAtStep(points, step);
        if (val !== null) {
          snapshots.push(`${tag}/${alias(run)}=${fmt(val)}`);
        }
      }
    }
    lines.push("  " + snapshots.join(" "));
  }

  return lines.join("\n");
}

/**
 * Format a number to 3 significant figures, compactly.
 */
function fmt(v) {
  return Number(v.toPrecision(3));
}

/**
 * Check if a series is effectively constant (all values within 0.1% of mean).
 */
function isConstant(values) {
  if (values.length <= 1) return true;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (mean === 0) return values.every((v) => v === 0);
  return values.every((v) => Math.abs(v - mean) / Math.abs(mean) < 0.001);
}

/**
 * Build short aliases for run names to save tokens.
 *
 * Pipeline:
 *  1. Strip common prefix/suffix shared by ALL runs (separator-aware)
 *  2. Remove noise segments: hex hashes (>=6 chars), UUIDs, timestamps,
 *     base64-ish blobs, long numeric IDs
 *  3. Collapse repeated separators and trim
 *  4. If still >20 chars, keep only first+last meaningful segments
 *  5. Deduplicate: on collision append shortest disambiguator from original
 *  6. If total alias length isn't shorter than original, skip aliasing
 *
 * Returns { aliases: {originalName → shortName}, legend: string }
 */
export function buildRunAliases(runs) {
  if (runs.length <= 1) {
    const map = {};
    runs.forEach((r) => { map[r] = r; });
    return { aliases: map, legend: "" };
  }

  const sorted = [...runs].sort();

  // ── 1. Common prefix (up to last separator) ──
  let prefix = sorted[0];
  for (const r of sorted) {
    while (prefix && !r.startsWith(prefix)) prefix = prefix.slice(0, -1);
  }
  // Snap to separator boundary
  const pSep = Math.max(prefix.lastIndexOf("/"), prefix.lastIndexOf("\\"),
    prefix.lastIndexOf("-"), prefix.lastIndexOf("_"), prefix.lastIndexOf("."));
  prefix = pSep > 0 ? prefix.slice(0, pSep + 1) : "";

  // ── Common suffix (up to first separator) ──
  let suffix = sorted[0];
  for (const r of sorted) {
    while (suffix && !r.endsWith(suffix)) suffix = suffix.slice(1);
  }
  const sSep = [suffix.indexOf("/"), suffix.indexOf("\\"),
    suffix.indexOf("-"), suffix.indexOf("_"), suffix.indexOf(".")]
    .filter((i) => i >= 0);
  if (sSep.length > 0) suffix = suffix.slice(Math.min(...sSep));
  else suffix = "";
  // Don't strip suffix if it would overlap with prefix
  if (suffix.length && prefix.length && prefix.length + suffix.length >= sorted[0].length) suffix = "";

  // ── 2–3. Strip noise from each run ──
  const noisePatterns = [
    /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, // UUID
    /[0-9a-f]{7,}/gi,          // hex hash (git-like, 7+ chars)
    /\d{8,10}T?\d{0,6}/g,      // timestamps: 20240115, 20240115T143022
    /\d{10,}/g,                 // long numeric IDs (unix timestamps etc)
    /[A-Za-z0-9+/]{20,}={0,2}/g, // base64-ish blobs
  ];

  const cleaned = {};
  for (const run of sorted) {
    let s = run;
    if (prefix) s = s.slice(prefix.length);
    if (suffix && s.endsWith(suffix)) s = s.slice(0, -suffix.length);
    for (const pat of noisePatterns) {
      s = s.replace(pat, "");
    }
    // Collapse separators, trim
    s = s.replace(/[-_/\\.]{2,}/g, "-").replace(/^[-_/\\.]+|[-_/\\.]+$/g, "");
    cleaned[run] = s || null;
  }

  // ── 4. Truncate long names: keep first + last segment ──
  const MAX_ALIAS = 20;
  for (const run of sorted) {
    let s = cleaned[run];
    if (!s || s.length <= MAX_ALIAS) continue;
    const parts = s.split(/[-_/\\.]+/);
    if (parts.length >= 3) {
      s = parts[0] + "-" + parts[parts.length - 1];
    }
    if (s.length > MAX_ALIAS) s = s.slice(0, MAX_ALIAS);
    cleaned[run] = s;
  }

  // ── 5. Deduplicate ──
  const aliases = {};
  const used = new Map(); // alias → run
  for (const run of sorted) {
    let candidate = cleaned[run] || "R";
    if (used.has(candidate)) {
      // Find shortest disambiguator from original segments
      const origParts = run.replace(prefix, "").split(/[-_/\\.]+/).filter(Boolean);
      const otherParts = used.get(candidate).replace(prefix, "").split(/[-_/\\.]+/).filter(Boolean);
      let disambig = "";
      for (const p of origParts) {
        if (!otherParts.includes(p) && p.length <= 8) { disambig = p; break; }
      }
      if (disambig) {
        candidate = candidate + "-" + disambig;
      }
      // Still duplicate? append index
      if (used.has(candidate)) {
        let idx = 2;
        while (used.has(candidate + idx)) idx++;
        candidate = candidate + idx;
      }
    }
    aliases[run] = candidate;
    used.set(candidate, run);
  }

  // ── 6. Check if aliasing saves anything ──
  const totalOriginal = sorted.reduce((s, r) => s + r.length, 0);
  const totalAlias = sorted.reduce((s, r) => s + aliases[r].length, 0);
  // Legend cost: each line is ~alias.length + run.length + 6 overhead
  const legendCost = sorted.reduce((s, r) => s + aliases[r].length + r.length + 6, 0);
  // aliases are used N times per metric, legend once; estimate N=5 uses per run
  const estimatedUses = 5;
  const savingsPerUse = totalOriginal - totalAlias;
  const allSame = sorted.every((r) => aliases[r] === r);

  if (allSame || savingsPerUse * estimatedUses < legendCost) {
    sorted.forEach((r) => { aliases[r] = r; });
    return { aliases, legend: "" };
  }

  const legendLines = sorted.map((r) => `  ${aliases[r]} = "${r}"`);
  return { aliases, legend: "Run aliases:\n" + legendLines.join("\n") };
}

/**
 * Build a text summary of a metric's series data for AI context.
 * Uses smart sampling: uniform trend points + anomaly points (spikes, drops).
 */
export function summarizeSeries(tag, series, runAliases) {
  const lines = [`Metric: ${tag}`];
  _appendSeriesData(lines, series, runAliases);
  return lines.join("\n");
}

/**
 * Append per-run data lines for a set of series.
 */
function _appendSeriesData(lines, series, runAliases) {
  for (const { run, points } of series) {
    if (!points.length) continue;
    const alias = (runAliases && runAliases[run]) || run;
    const values = points.map((p) => p.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const first = values[0];
    const last = values[values.length - 1];
    const steps = points.map((p) => p.step);
    lines.push(
      `  [${alias}] ${points.length}pts s${steps[0]}-${steps[steps.length - 1]} ` +
        `[${fmt(min)},${fmt(max)}] first=${fmt(first)} last=${fmt(last)}`,
    );
    // Constant detection: skip sampling if values don't change
    if (isConstant(values)) {
      lines.push(`  constant=${fmt(values[0])}`);
    } else {
      const sampled = smartSample(points);
      const pairs = sampled.map((p) => `(${p.step},${fmt(p.value)})`);
      lines.push(`  Data: ${pairs.join(" ")}`);
    }
  }
}

/**
 * Build a fingerprint string for a tag's series data (for deduplication).
 * Two tags with identical run→points data will produce the same fingerprint.
 */
function _seriesFingerprint(series) {
  return series
    .map(({ run, points }) => {
      if (!points.length) return run + ":empty";
      const vals = points.map((p) => p.step + "," + fmt(p.value));
      return run + ":" + vals.join(";");
    })
    .sort()
    .join("|");
}

/**
 * Summarize multiple tags, deduplicating tags with identical data.
 * Tags with the same series data are grouped: "Metrics: tag1, tag2, tag3"
 * with the data listed only once.
 */
export function summarizeTagGroups(tagGroups, runAliases) {
  // Group tags by their data fingerprint
  const fpMap = new Map(); // fingerprint → { tags: string[], series }
  for (const [tag, series] of Object.entries(tagGroups)) {
    const fp = _seriesFingerprint(series);
    if (!fpMap.has(fp)) {
      fpMap.set(fp, { tags: [], series });
    }
    fpMap.get(fp).tags.push(tag);
  }

  const blocks = [];
  for (const { tags, series } of fpMap.values()) {
    const lines = [];
    if (tags.length === 1) {
      lines.push(`Metric: ${tags[0]}`);
    } else {
      lines.push(`Metrics (identical data): ${tags.join(", ")}`);
    }
    _appendSeriesData(lines, series, runAliases);
    blocks.push(lines.join("\n"));
  }
  return blocks.join("\n\n");
}
