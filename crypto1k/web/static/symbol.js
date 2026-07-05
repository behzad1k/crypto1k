/**
 * Symbol analysis page — v2
 * Handles horizon selection, analysis, chart, market data, and result rendering.
 */

const HORIZON_DESCRIPTIONS = {
  short: "Scalping / intraday  ·  5m, 15m, 30m",
  mid: "Intraday swing / day trading  ·  1h, 2h, 4h",
  long: "Swing / position trading  ·  6h, 12h, 1d, 1w",
};

const CATEGORY_LABELS = {
  momentum: "Momentum",
  trend: "Trend",
  volume: "Volume",
  volatility: "Volatility",
  structure: "Market Structure",
  smc: "Smart Money (SMC)",
  pattern: "Price Patterns",
};

// Level line colors per direction/grade
const LEVEL_COLORS = {
  bullish: { A: "#22c55e", B: "#16a34a", C: "#166534" },
  bearish: { A: "#ef4444", B: "#dc2626", C: "#991b1b" },
  neutral: { A: "#94a3b8", B: "#64748b", C: "#475569" },
};

let chartInstance = null;
let candleSeries = null;
let currentSymbol = "";
let currentChartTf = "1h";
let lastResult = null;
let lastCandles = [];

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  currentSymbol = document.getElementById("symbolInput").value.trim().toUpperCase();

  document.getElementById("symbolInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") analyze();
  });

  document.getElementById("analyzeBtn").addEventListener("click", analyze);

  document.getElementById("chartTfSelect").addEventListener("change", (e) => {
    currentChartTf = e.target.value;
    loadChart(currentSymbol, currentChartTf);
  });

  fetchMarketData(currentSymbol);
});

// ── Analyze ───────────────────────────────────────────────────────────────────

async function fetchHorizon(symbol, horizon) {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, horizon }),
  });
  const data = await res.json();
  if (!data.success) throw new Error(data.error || `${horizon} analysis failed`);
  return data;
}

async function analyze() {
  const symbol = document.getElementById("symbolInput").value.trim().toUpperCase();
  if (!symbol) return;

  currentSymbol = symbol;
  setLoading(true);
  clearError();
  fetchMarketData(symbol);

  try {
    const [short, mid, long] = await Promise.all([
      fetchHorizon(symbol, "short"),
      fetchHorizon(symbol, "mid"),
      fetchHorizon(symbol, "long"),
    ]);

    lastResult = { symbol, short, mid, long };
    renderAllResults(short, mid, long);
    fetchPersistence(symbol);
    fetchLiquidations(symbol);
  } catch (err) {
    showError(err.message || "Network error — could not reach server");
    console.error(err);
  } finally {
    setLoading(false);
  }
}

// ── Market data ───────────────────────────────────────────────────────────────

async function fetchMarketData(symbol) {
  setMarketLoading();
  try {
    const res = await fetch(
      `/api/market-data/${encodeURIComponent(symbol + "-usdt")}`,
    );
    const data = await res.json();
    if (data.success) renderMarketData(data);
  } catch (err) {
    console.warn("Market data fetch failed:", err);
    setMarketUnavailable();
  }
}

function setMarketLoading() {
  ["mdFundingRate", "mdOI", "mdImbalance"].forEach((id) => {
    const el = document.getElementById(id);
    el.textContent = "loading…";
    el.className = "market-item-value loading";
  });
  document.getElementById("mdNextFunding").textContent = "";
  document.getElementById("mdOIChange").textContent = "";
  document.getElementById("mdImbalanceSub").textContent = "";
}

function setMarketUnavailable() {
  ["mdFundingRate", "mdOI", "mdImbalance"].forEach((id) => {
    const el = document.getElementById(id);
    el.textContent = "N/A";
    el.className = "market-item-value neutral";
  });
}

function renderMarketData(data) {
  // ── Funding rate ───────────────────────────────────────────────────────────
  const frEl = document.getElementById("mdFundingRate");
  const frSub = document.getElementById("mdNextFunding");

  if (data.funding_rate !== null) {
    const fr = data.funding_rate;
    const pct = (fr * 100).toFixed(4) + "%";
    frEl.textContent = pct;

    // Color: neutral if |fr| < 0.01%, elevated if < 0.05%, extreme otherwise
    const abs = Math.abs(fr * 100);
    if (abs < 0.01) {
      frEl.className = "market-item-value neutral";
    } else if (fr > 0) {
      frEl.className =
        abs < 0.05 ? "market-item-value" : "market-item-value bear";
      // Positive funding = longs paying = crowded long = slightly bearish signal
    } else {
      frEl.className =
        abs < 0.05 ? "market-item-value" : "market-item-value bull";
    }

    if (data.next_funding_in !== null) {
      frSub.textContent = `Next in ${data.next_funding_in}m`;
    }
  } else {
    frEl.textContent = "No perp";
    frEl.className = "market-item-value neutral";
    frSub.textContent = "Spot only";
  }

  // ── Open interest ──────────────────────────────────────────────────────────
  const oiEl = document.getElementById("mdOI");
  const oiSub = document.getElementById("mdOIChange");

  if (data.open_interest !== null) {
    oiEl.textContent = formatLargeNumber(data.open_interest);
    oiEl.className = "market-item-value";

    if (data.oi_change_pct !== null) {
      const chg = data.oi_change_pct;
      const arrow = chg >= 0 ? "↑" : "↓";
      const cls = chg >= 0 ? "bull" : "bear";
      oiSub.innerHTML = `<span style="color:var(--${cls})">${arrow} ${Math.abs(chg).toFixed(2)}% (1h)</span>`;
    }
  } else {
    oiEl.textContent = "No perp";
    oiEl.className = "market-item-value neutral";
    oiSub.textContent = "Spot only";
  }

  // ── Order book imbalance ───────────────────────────────────────────────────
  const imEl = document.getElementById("mdImbalance");
  const imSub = document.getElementById("mdImbalanceSub");

  if (data.imbalance_ratio !== null) {
    const ratio = data.imbalance_ratio;
    const side = data.imbalance_side;
    imEl.textContent = `${ratio}:1`;
    imEl.className = `market-item-value ${side === "bullish" ? "bull" : side === "bearish" ? "bear" : "neutral"}`;

    const bidK = formatLargeNumber(data.bid_depth);
    const askK = formatLargeNumber(data.ask_depth);
    imSub.textContent = `Bid $${bidK}  /  Ask $${askK}`;
  } else {
    imEl.textContent = "N/A";
    imEl.className = "market-item-value neutral";
    imSub.textContent = "";
  }
}

// ── Chart ─────────────────────────────────────────────────────────────────────

function initChart() {
  const container = document.getElementById("chartContainer");
  container.innerHTML = "";

  chartInstance = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: 320,
    layout: {
      background: { color: "#161a23" },
      textColor: "#64748b",
    },
    grid: {
      vertLines: { color: "#1e2330" },
      horzLines: { color: "#1e2330" },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: "#2a3040" },
    timeScale: {
      borderColor: "#2a3040",
      timeVisible: true,
      secondsVisible: false,
    },
  });

  candleSeries = chartInstance.addCandlestickSeries({
    upColor: "#22c55e",
    downColor: "#ef4444",
    borderVisible: false,
    wickUpColor: "#22c55e",
    wickDownColor: "#ef4444",
  });

  // Resize observer
  new ResizeObserver(() => {
    chartInstance.applyOptions({ width: container.clientWidth });
  }).observe(container);
}

async function loadChart(symbol, timeframe) {
  document.getElementById("chartTitle").textContent =
    `${symbol}  ·  ${timeframe}`;

  if (!chartInstance) initChart();

  try {
    const res = await fetch(
      `/api/ohlcv/${encodeURIComponent(symbol)}/${timeframe}`,
    );
    const data = await res.json();
    if (!data.success || !data.candles.length) return;

    candleSeries.setData(data.candles);
    chartInstance.timeScale().fitContent();
    lastCandles = data.candles;
    drawVPVR();
  } catch (err) {
    console.warn("Chart load failed:", err);
  }
}

function drawChartLevels(chartLevels) {
  if (!candleSeries || !chartLevels) return;

  // Remove existing price lines (recreate series is simplest)
  const existingData = candleSeries.data ? candleSeries.data() : null;

  chartLevels.forEach((level) => {
    if (!level.price) return;
    const color =
      (LEVEL_COLORS[level.direction] || LEVEL_COLORS.neutral)[level.grade] ||
      "#64748b";
    try {
      candleSeries.createPriceLine({
        price: level.price,
        color: color,
        lineWidth: level.grade === "A" ? 2 : 1,
        lineStyle:
          level.grade === "A"
            ? LightweightCharts.LineStyle.Solid
            : LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: level.grade === "A",
        title: level.label,
      });
    } catch (e) {
      /* price line already exists or out of range */
    }
  });

  // Build legend
  const legend = document.getElementById("chartLegend");
  legend.innerHTML = "";
  const seen = new Set();
  chartLevels.forEach((l) => {
    if (!l.price || seen.has(l.label)) return;
    seen.add(l.label);
    const color = (LEVEL_COLORS[l.direction] || LEVEL_COLORS.neutral)[l.grade];
    const item = document.createElement("div");
    item.className = "chart-legend-item";
    item.innerHTML = `<span class="chart-legend-dot" style="background:${color}"></span>${l.label}: ${formatPrice(l.price)}`;
    legend.appendChild(item);
  });
}

// ── Results rendering ─────────────────────────────────────────────────────────

function renderAllResults(short, mid, long) {
  document.getElementById("priceDisplay").textContent = formatPrice(mid.price);
  document.getElementById("lastUpdated").textContent =
    "Updated " + new Date(mid.timestamp).toLocaleTimeString();

  // Chart uses mid timeframes
  const tfs = mid.timeframes || [];
  const tfSelect = document.getElementById("chartTfSelect");
  tfSelect.innerHTML = tfs
    .map((tf) => `<option value="${tf}"${tf === mid.chart_tf ? " selected" : ""}>${tf}</option>`)
    .join("");
  currentChartTf = mid.chart_tf;
  document.getElementById("chartSection").style.display = "block";
  loadChart(mid.symbol, currentChartTf).then(() => drawChartLevels(mid.chart_levels));

  // Build the three stacked horizon sections
  const container = document.getElementById("horizonSections");
  container.innerHTML = "";
  for (const data of [short, mid, long]) {
    container.appendChild(buildHorizonSection(data));
  }

  document.getElementById("emptyState").style.display = "none";
  document.getElementById("results").style.display = "block";
  document.getElementById("exportBtns").style.display = "flex";
}

// ── Horizon section builder ───────────────────────────────────────────────────

function buildHorizonSection(data) {
  const h     = data.horizon;
  const bias  = data.bias;
  const pct   = bias.score === 0 ? 50 : Math.round(((bias.score + 1) / 2) * 100);

  const section = document.createElement("div");
  section.className = `horizon-section ${h}`;

  // Data-source badge
  const sourceBadges = {
    cex: { label: "CEX · KuCoin/Binance", color: "var(--muted)" },
    dexscreener: { label: "DexScreener · on-chain", color: "#a78bfa" },
    dexscreener_snapshot: { label: "DexScreener · no candles (limited)", color: "#f59e0b" },
  };
  const sb = sourceBadges[data.source];
  const sourceHtml = sb
    ? `<span style="font-size:10px;font-weight:600;color:${sb.color};border:1px solid ${sb.color};border-radius:4px;padding:1px 6px">${sb.label}</span>`
    : "";

  // Header
  const header = document.createElement("div");
  header.className = "horizon-section-header";
  header.innerHTML = `
    <span class="horizon-label ${h}">${data.horizon_label}</span>
    <span style="font-size:11px;color:var(--muted)">${data.timeframes.join(" · ")}</span>
    <span class="family-bias-badge ${bias.direction}">${bias.direction.toUpperCase()}</span>
    <span style="font-size:11px;color:var(--muted)">score <strong style="color:var(--text)">${bias.score.toFixed(2)}</strong></span>
    ${sourceHtml}
    <span style="margin-left:auto;font-size:11px;color:var(--muted)">${formatPrice(data.price)} · ${new Date(data.timestamp).toLocaleTimeString()}</span>
  `;
  section.appendChild(header);

  // Body
  const body = document.createElement("div");
  body.className = "horizon-section-body";

  // Bias banner
  const biasEl = document.createElement("div");
  biasEl.className = `bias-banner-el`;
  const biasColors = {
    bullish: { bg: "linear-gradient(135deg,rgba(34,197,94,0.08),rgba(34,197,94,0.03))", border: "var(--bull-dim)" },
    bearish: { bg: "linear-gradient(135deg,rgba(239,68,68,0.08),rgba(239,68,68,0.03))", border: "var(--bear-dim)" },
    neutral: { bg: "rgba(148,163,184,0.05)", border: "var(--border)" },
  };
  const bc = biasColors[bias.direction] || biasColors.neutral;
  biasEl.style.cssText = `background:${bc.bg};border:1px solid ${bc.border};border-radius:8px;padding:12px 16px;margin-bottom:12px`;
  biasEl.innerHTML = `
    <div class="bias-row">
      <div class="bias-direction ${bias.direction}">${bias.direction.toUpperCase()}</div>
      <div class="bias-bar-wrap">
        <div class="bias-bar-track"><div class="bias-bar-fill" style="width:${pct}%"></div></div>
        <div class="bias-labels"><span>Bear</span><span>Bull</span></div>
      </div>
      <div class="bias-meta">
        <span style="color:var(--text);font-weight:600">${bias.bull_score}</span> bull &nbsp;/&nbsp;
        <span style="color:var(--text);font-weight:600">${bias.bear_score}</span> bear
        &nbsp;·&nbsp;
        <span style="color:var(--text);font-weight:600">${data.all_signals.length}</span> signals
      </div>
    </div>
  `;
  body.appendChild(biasEl);

  // Degraded mode: no OHLCV anywhere — show non-OHLCV DexScreener signals only
  if (data.degraded && data.snapshot) {
    body.appendChild(buildSnapshotEl(data.snapshot));
    section.appendChild(body);
    return section;
  }

  // Scalp dashboard
  if (data.scalp_metrics) body.appendChild(buildScalpEl(data.scalp_metrics, bias));

  // Primary strip
  if (data.primary_signals && data.primary_signals.length) {
    body.appendChild(buildPrimaryEl(data.primary_signals));
  }

  // Grade legend
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = `
    <div class="legend-item"><span class="legend-grade A">A</span> Primary</div>
    <div class="legend-item"><span class="legend-grade B">B</span> Confirmation</div>
    <div class="legend-item"><span class="legend-grade C">C</span> Context</div>
  `;
  body.appendChild(legend);

  // Family cards
  body.appendChild(buildCategoriesEl(data.families, data.category_order));

  section.appendChild(body);
  return section;
}

// Degraded panel: DexScreener aggregate signals when no OHLCV candles exist
function buildSnapshotEl(snapshot) {
  const el = document.createElement("div");
  const p = snapshot.pair || {};
  const dirColor = { bullish: "var(--bull)", bearish: "var(--bear)", neutral: "var(--muted)" };

  const notice = `
    <div style="background:rgba(245,158,11,0.08);border:1px solid #f59e0b;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:var(--text)">
      ⚠️ No OHLCV candles available for this token (not on a CEX, and no on-chain
      candle history). Showing <strong>non-OHLCV signals</strong> derived from
      DexScreener aggregates only — technical indicators are unavailable.
    </div>`;

  const meta = `
    <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-bottom:10px">
      <span>Pair: <strong style="color:var(--text)">${p.symbol || "?"}</strong></span>
      <span>Chain: <strong style="color:var(--text)">${p.chain || "?"}</strong></span>
      <span>DEX: <strong style="color:var(--text)">${p.dex || "?"}</strong></span>
      <span>Price: <strong style="color:var(--text)">${p.price_usd != null ? formatPrice(p.price_usd) : "?"}</strong></span>
      <span>Liquidity: <strong style="color:var(--text)">$${Math.round(p.liquidity_usd || 0).toLocaleString()}</strong></span>
      ${p.url ? `<a href="${p.url}" target="_blank" rel="noopener" style="color:#a78bfa">DexScreener ↗</a>` : ""}
    </div>`;

  const rows = (snapshot.signals || [])
    .map((s) => `
      <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border)">
        <span style="width:8px;height:8px;border-radius:50%;background:${dirColor[s.direction] || "var(--muted)"}"></span>
        <span style="font-weight:600;color:var(--text);font-size:12px">${s.name}</span>
        <span style="margin-left:auto;font-size:11px;color:var(--muted)">${s.detail || ""}</span>
      </div>`)
    .join("");

  el.innerHTML = notice + meta +
    `<div class="strip-label">DexScreener signals</div>` +
    (rows || `<div style="font-size:12px;color:var(--muted);padding:6px 0">No notable signals.</div>`);
  return el;
}

function buildPrimaryEl(primarySignals) {
  const el = document.createElement("div");
  el.style.marginBottom = "12px";
  el.innerHTML = `<div class="strip-label">⭐ Grade A — Primary signals</div>`;
  const pills = document.createElement("div");
  pills.className = "primary-pills";
  primarySignals.forEach((s) => {
    const pill = document.createElement("div");
    pill.className = "primary-pill";
    const levelHtml = s.price_level ? `<span class="pill-level">@ ${formatPrice(s.price_level)}</span>` : "";
    pill.innerHTML = `
      <span class="pill-dir ${s.direction}"></span>
      <span>${formatSignalName(s.signal_name)}</span>
      <span class="pill-tf">${s.timeframe}</span>
      ${levelHtml}
    `;
    pills.appendChild(pill);
  });
  el.appendChild(pills);
  return el;
}

function buildCategoriesEl(families, categoryOrder) {
  const container = document.createElement("div");
  const byCategory = {};
  Object.entries(families).forEach(([, fam]) => {
    if (!byCategory[fam.category]) byCategory[fam.category] = {};
    byCategory[fam.category][fam.name] = fam;
  });
  const ordered = categoryOrder ? categoryOrder.filter((c) => byCategory[c]) : Object.keys(byCategory);
  ordered.forEach((cat) => {
    const catFamilies = byCategory[cat];
    if (!catFamilies) return;
    const section = document.createElement("div");
    section.className = "category-section";
    const heading = document.createElement("div");
    heading.className = "category-heading";
    heading.textContent = CATEGORY_LABELS[cat] || cat;
    section.appendChild(heading);
    const grid = document.createElement("div");
    grid.className = "family-grid";
    Object.values(catFamilies).forEach((fam) => grid.appendChild(buildFamilyCard(fam)));
    section.appendChild(grid);
    container.appendChild(section);
  });
  return container;
}

function buildScalpEl(m, bias) {
  const dir   = bias ? bias.direction : "neutral";
  const label = m.setup_label || "WEAK";
  const score = Math.round(m.setup_score ?? 0);
  const tfA   = m.tf_alignment || {};

  const el = document.createElement("div");
  el.style.cssText = "background:var(--surface);border:1px solid var(--border);border-radius:10px;margin-bottom:12px;overflow:hidden";

  const pipsHtml = Array.from({ length: 10 }, (_, i) =>
    `<div class="setup-score-pip${i < score ? ` filled ${label}` : ""}"></div>`
  ).join("");

  const aligned = tfA.aligned_count ?? 0;
  const total   = tfA.total_count ?? 0;
  const strength = tfA.strength === "full" ? 1 : tfA.strength === "partial" ? 0.5 : 0;
  const tfCls  = strength >= 0.7 ? "good" : strength >= 0.4 ? "warn" : "bad";
  const tfDirHtml = Object.entries(tfA.timeframes || {}).map(([tf, d]) => {
    const arrow = d === "bullish" ? "↑" : d === "bearish" ? "↓" : "·";
    const cls   = d === "bullish" ? "bull" : d === "bearish" ? "bear" : "neutral";
    return `<span style="color:var(--${cls})">${tf}${arrow}</span>`;
  }).join(" ");

  const vr  = m.volume_ratio;
  const vCls = vr >= 2 ? "good" : vr >= 1.2 ? "warn" : "neutral";
  const vSub = m.volume_trend === "rising"
    ? `<span style="color:var(--bull)">↑ rising</span>`
    : m.volume_trend === "falling"
    ? `<span style="color:var(--bear)">↓ falling</span>`
    : "avg";

  const atrPct = m.atr_pct;
  const atrCls = atrPct >= 0.8 ? "good" : atrPct >= 0.3 ? "warn" : "bad";
  const atrs   = m.atrs_for_2pct;
  const atrSub = atrs != null
    ? `<span style="color:${m.target_realistic ? "var(--bull)" : "var(--bear)"}">${atrs.toFixed(1)} ATRs for 2%</span>`
    : "";

  const ra = m.room_above; const rb = m.room_below;
  const raCls = ra?.pct >= 2 ? "good" : ra?.pct >= 1 ? "warn" : "bad";
  const rbCls = rb?.pct >= 2 ? "good" : rb?.pct >= 1 ? "warn" : "bad";

  el.innerHTML = `
    <div class="scalp-header">
      <span class="scalp-header-label">⚡ Setup Quality</span>
      <div class="setup-score-badge">
        <span class="setup-score-label ${label}">${label}</span>
        <div class="setup-score-bar">${pipsHtml}</div>
        <span style="font-size:11px;color:var(--muted)">${score}/10</span>
      </div>
    </div>
    <div class="scalp-metrics-row">
      <div class="scalp-metric">
        <div class="scalp-metric-label">TF Alignment</div>
        <div class="scalp-metric-value ${tfCls}">${aligned}/${total}</div>
        <div class="scalp-metric-sub">${tfDirHtml}</div>
      </div>
      <div class="scalp-metric">
        <div class="scalp-metric-label">Volume</div>
        <div class="scalp-metric-value ${vCls}">${vr != null ? vr.toFixed(2) + "×" : "—"}</div>
        <div class="scalp-metric-sub">${vSub}</div>
      </div>
      <div class="scalp-metric">
        <div class="scalp-metric-label">ATR Context</div>
        <div class="scalp-metric-value ${atrCls}">${atrPct != null ? atrPct.toFixed(2) + "% ATR" : "—"}</div>
        <div class="scalp-metric-sub">${atrSub}</div>
      </div>
      <div class="scalp-metric">
        <div class="scalp-metric-label">Room Above</div>
        <div class="scalp-metric-value ${ra?.pct != null ? raCls : "neutral"}">${ra?.pct != null ? ra.pct.toFixed(2) + "%" : "—"}</div>
        <div class="scalp-metric-sub">${dir === "bullish" ? "target zone" : "resistance"}</div>
      </div>
      <div class="scalp-metric">
        <div class="scalp-metric-label">Room Below</div>
        <div class="scalp-metric-value ${rb?.pct != null ? rbCls : "neutral"}">${rb?.pct != null ? rb.pct.toFixed(2) + "%" : "—"}</div>
        <div class="scalp-metric-sub">${dir === "bearish" ? "target zone" : "support"}</div>
      </div>
    </div>
  `;

  // Factors + key levels
  const body = document.createElement("div");
  body.className = "scalp-body";

  const factorsDiv = document.createElement("div");
  factorsDiv.className = "setup-factors";
  factorsDiv.innerHTML = `<div class="factor-label">Setup Quality Breakdown</div>`;
  (m.setup_factors || []).forEach((f) => {
    const ratio    = f.max > 0 ? (f.points ?? 0) / f.max : 0;
    const barColor = ratio >= 0.67 ? "var(--bull)" : ratio >= 0.34 ? "var(--grade-a)" : "var(--bear)";
    const row = document.createElement("div");
    row.innerHTML = `
      <div class="factor-row">
        <div class="factor-name">${f.name}</div>
        <div class="factor-bar-track"><div class="factor-bar-fill" style="width:${Math.round(ratio * 100)}%;background:${barColor}"></div></div>
        <div class="factor-pts">${f.points}/${f.max}</div>
      </div>
      ${f.detail ? `<div class="factor-detail">${f.detail}</div>` : ""}
    `;
    factorsDiv.appendChild(row);
  });
  body.appendChild(factorsDiv);

  const levelsDiv = document.createElement("div");
  levelsDiv.className = "key-levels-panel";
  levelsDiv.innerHTML = `<div class="key-levels-label">Key Levels — Distance from Current Price</div>`;
  (m.key_levels || []).slice(0, 8).forEach((l) => {
    const isAbove = l.pct != null && l.pct >= 0;
    const row = document.createElement("div");
    row.className = "key-level-row";
    row.innerHTML = `
      <div class="key-level-arrow">${isAbove ? "↑" : "↓"}</div>
      <div class="key-level-price">${formatPrice(l.price)}</div>
      <div class="key-level-pct ${isAbove ? "above" : "below"}">${l.pct != null ? (l.pct >= 0 ? "+" : "") + l.pct.toFixed(2) + "%" : ""}</div>
      <div class="key-level-label">${l.label}</div>
      ${l.grade ? `<div class="key-level-grade ${l.grade}">${l.grade}</div>` : ""}
    `;
    levelsDiv.appendChild(row);
  });
  body.appendChild(levelsDiv);
  el.appendChild(body);

  const stopAbove = m.stop_above ?? null;
  const stopBelow = m.stop_below ?? null;
  if (stopAbove !== null || stopBelow !== null) {
    const stopEl = document.createElement("div");
    stopEl.className = "stop-suggestion";
    if (stopBelow !== null) stopEl.innerHTML += `<div class="stop-item"><span style="color:var(--bull)">▲ Long</span> <strong>Stop: ${formatPrice(stopBelow)}</strong></div>`;
    if (stopAbove !== null) stopEl.innerHTML += `<div class="stop-item"><span style="color:var(--bear)">▼ Short</span> <strong>Stop: ${formatPrice(stopAbove)}</strong></div>`;
    el.appendChild(stopEl);
  }

  return el;
}

// Interpretation guide per indicator label — shown as a hint on value chips so
// you can judge a reading without memorizing each indicator's scale.
const INDICATOR_GUIDE = {
  ADX:   { hint: "<20 weak · >25 trending · >60 strong", rate: (v) => (v >= 60 ? "good" : v >= 25 ? "ok" : "bad") },
  RSI:   { hint: "<30 oversold · >70 overbought",        rate: (v) => (v <= 30 || v >= 70 ? "good" : "ok") },
  Stoch: { hint: "<20 oversold · >80 overbought",        rate: (v) => (v <= 20 || v >= 80 ? "good" : "ok") },
  MFI:   { hint: "<20 oversold · >80 overbought",        rate: (v) => (v <= 20 || v >= 80 ? "good" : "ok") },
  CCI:   { hint: "<-100 oversold · >+100 overbought",    rate: (v) => (Math.abs(v) >= 100 ? "good" : "ok") },
  "%R":  { hint: "<-80 oversold · >-20 overbought",      rate: (v) => (v <= -80 || v >= -20 ? "good" : "ok") },
  TSI:   { hint: "<-25 oversold · >+25 overbought",      rate: (v) => (Math.abs(v) >= 25 ? "good" : "ok") },
  ROC:   { hint: ">0 bullish · <0 bearish",              rate: (v) => (Math.abs(v) >= 5 ? "good" : "ok") },
  CMF:   { hint: ">+0.2 strong buying · <-0.2 strong selling", rate: (v) => (Math.abs(v) >= 0.2 ? "good" : Math.abs(v) >= 0.05 ? "ok" : "bad") },
  Hist:  { hint: ">0 bullish · <0 bearish · size = momentum",  rate: () => "ok" },
  ATR:   { hint: "higher = more volatile, bigger moves",       rate: () => "ok" },
};

function buildFamilyCard(fam) {
  const card = document.createElement("div");
  card.className = `family-card ${fam.bias}`;

  // Header
  card.innerHTML = `
    <div class="family-header">
      <span class="family-name">${fam.name}</span>
      <span class="family-bias-badge ${fam.bias}">${fam.bias}</span>
    </div>
    <div class="family-description">${fam.description}</div>
  `;

  // Indicator values (blue chips — raw readings like RSI: 22.8)
  const allValues = [
    ...Object.entries(fam.indicator_values || {}).map(([label, val]) => ({
      label,
      val,
      type: "indicator",
    })),
    ...Object.entries(fam.price_levels || {}).map(([label, val]) => ({
      label,
      val: formatPrice(val),
      type: "price_level",
    })),
  ];

  if (allValues.length) {
    const valWrap = document.createElement("div");
    valWrap.className = "family-values";
    allValues.forEach(({ label, val, type }) => {
      const chip = document.createElement("span");
      chip.className = `value-chip ${type}`;
      const guide = type === "indicator" ? INDICATOR_GUIDE[label] : null;
      const rating = guide && typeof val === "number" ? guide.rate(val) : null;
      chip.innerHTML =
        `<span class="chip-label">${label}</span>` +
        `<span class="chip-value${rating ? " " + rating : ""}">${typeof val === "number" ? formatValue(val) : val}</span>` +
        (guide ? `<span class="chip-hint">${guide.hint}</span>` : "");
      if (guide) chip.title = `${label}: ${guide.hint}`;
      valWrap.appendChild(chip);
    });
    card.appendChild(valWrap);
  }

  // Signal list
  if (fam.active_signals && fam.active_signals.length) {
    const list = document.createElement("div");
    list.className = "signal-list";

    const sorted = [...fam.active_signals].sort(
      (a, b) =>
        (({ A: 0, B: 1, C: 2 })[a.grade] ?? 3) -
        ({ A: 0, B: 1, C: 2 }[b.grade] ?? 3),
    );

    sorted.forEach((s) => {
      const item = document.createElement("div");
      item.className = "signal-item";

      const freshnessHtml = buildFreshnessHtml(s.candles_ago);
      const levelHtml = s.price_level
        ? `<span class="signal-price-level">${formatPrice(s.price_level)}</span>`
        : "";

      item.innerHTML = `
        <span class="grade-badge ${s.grade}">${s.grade}</span>
        <span class="signal-dir-dot ${s.direction}"></span>
        <span class="signal-name-text">${formatSignalName(s.signal_name)}</span>
        ${levelHtml}
        <span class="signal-tf-badge">${s.timeframe}</span>
        ${freshnessHtml}
        <span class="signal-role-tag">${s.role}</span>
      `;
      list.appendChild(item);
    });

    card.appendChild(list);
  }

  return card;
}

function buildFreshnessHtml(candles_ago) {
  if (candles_ago === null || candles_ago === undefined) return "";
  if (candles_ago <= 1)
    return `<span class="freshness-badge fresh">fresh</span>`;
  if (candles_ago <= 3)
    return `<span class="freshness-badge recent">${candles_ago}C ago</span>`;
  return `<span class="freshness-badge stale">${candles_ago}C ago</span>`;
}

// ── Scalp dashboard (legacy — kept for reference, replaced by buildScalpEl) ───

function renderScalpDashboard(m, bias) {
  if (!m) return;

  document.getElementById("scalpDashboard").style.display = "block";
  const dir = bias ? bias.direction : "neutral";

  // ── Setup score ──────────────────────────────────────────────────────────
  const scoreEl = document.getElementById("setupScoreLabel");
  const barEl = document.getElementById("setupScoreBar");
  const numEl = document.getElementById("setupScoreNum");

  scoreEl.textContent = m.setup_label || "—";
  scoreEl.className = "setup-score-label " + (m.setup_label || "");

  const score = Math.round(m.setup_score ?? 0);
  const label = m.setup_label || "WEAK";
  barEl.innerHTML = Array.from(
    { length: 10 },
    (_, i) =>
      `<div class="setup-score-pip${i < score ? ` filled ${label}` : ""}"></div>`,
  ).join("");
  numEl.textContent = `${score}/10`;

  // ── TF Alignment ─────────────────────────────────────────────────────────
  const tfA = m.tf_alignment || {};
  const aEl = document.getElementById("smAlignment");
  const aSub = document.getElementById("smAlignmentSub");

  const aligned = tfA.aligned_count ?? 0;
  const total = tfA.total_count ?? 0;
  // strength from backend is string: 'full' / 'partial' / 'split'
  const strength =
    tfA.strength === "full" ? 1 : tfA.strength === "partial" ? 0.5 : 0;

  aEl.textContent = `${aligned}/${total}`;
  aEl.className =
    "scalp-metric-value " +
    (strength >= 0.7 ? "good" : strength >= 0.4 ? "warn" : "bad");

  // Build per-TF direction summary  e.g. "5m↑ 15m↑ 30m↓"
  const tfDirs = tfA.timeframes || {};
  aSub.innerHTML = Object.entries(tfDirs)
    .map(([tf, d]) => {
      const arrow = d === "bullish" ? "↑" : d === "bearish" ? "↓" : "·";
      const cls =
        d === "bullish" ? "bull" : d === "bearish" ? "bear" : "neutral";
      return `<span style="color:var(--${cls})">${tf}${arrow}</span>`;
    })
    .join(" ");

  // ── Volume ────────────────────────────────────────────────────────────────
  const vRatio = m.volume_ratio ?? null;
  const vTrend = m.volume_trend || "";
  const vEl = document.getElementById("smVolume");
  const vSub = document.getElementById("smVolumeSub");

  if (vRatio !== null) {
    vEl.textContent = vRatio.toFixed(2) + "x";
    vEl.className =
      "scalp-metric-value " +
      (vRatio >= 2 ? "good" : vRatio >= 1.2 ? "warn" : "neutral");
    vSub.textContent =
      vTrend === "rising"
        ? "↑ rising"
        : vTrend === "falling"
          ? "↓ falling"
          : "avg";
    vSub.style.color =
      vTrend === "rising"
        ? "var(--bull)"
        : vTrend === "falling"
          ? "var(--bear)"
          : "";
  } else {
    vEl.textContent = "—";
    vEl.className = "scalp-metric-value neutral";
    vSub.textContent = "";
  }

  // ── ATR ───────────────────────────────────────────────────────────────────
  const atrPct = m.atr_pct ?? null;
  const atrVal = m.atr_value ?? null;
  const atr2pct = m.atrs_for_2pct ?? null;
  const atEl = document.getElementById("smAtr");
  const atSub = document.getElementById("smAtrSub");

  if (atrPct !== null) {
    atEl.textContent = atrPct.toFixed(2) + "% ATR";
    atEl.className =
      "scalp-metric-value " +
      (atrPct >= 0.8 ? "good" : atrPct >= 0.3 ? "warn" : "bad");
    if (atr2pct !== null) {
      const realistic = m.target_realistic;
      atSub.textContent = `${atr2pct.toFixed(1)} ATRs for 2%`;
      atSub.style.color = realistic ? "var(--bull)" : "var(--bear)";
    } else {
      atSub.textContent = atrVal ? formatPrice(atrVal) : "";
    }
  } else {
    atEl.textContent = "—";
    atEl.className = "scalp-metric-value neutral";
    atSub.textContent = "";
  }

  // ── Room above / below ────────────────────────────────────────────────────
  const raEl = document.getElementById("smRoomAbove");
  const raSub = document.getElementById("smRoomAboveSub");
  const rbEl = document.getElementById("smRoomBelow");
  const rbSub = document.getElementById("smRoomBelowSub");

  // room_above / room_below are objects: { price, label, pct, enough }
  const raObj = m.room_above;
  if (raObj && raObj.pct != null) {
    const ra = raObj.pct;
    raEl.textContent = ra.toFixed(2) + "%";
    raEl.className =
      "scalp-metric-value " + (ra >= 2 ? "good" : ra >= 1 ? "warn" : "bad");
    raSub.textContent = dir === "bullish" ? "target zone" : "resistance";
  } else {
    raEl.textContent = "—";
    raEl.className = "scalp-metric-value neutral";
    raSub.textContent = "";
  }

  const rbObj = m.room_below;
  if (rbObj && rbObj.pct != null) {
    const rb = rbObj.pct;
    rbEl.textContent = rb.toFixed(2) + "%";
    rbEl.className =
      "scalp-metric-value " + (rb >= 2 ? "good" : rb >= 1 ? "warn" : "bad");
    rbSub.textContent = dir === "bearish" ? "target zone" : "support";
  } else {
    rbEl.textContent = "—";
    rbEl.className = "scalp-metric-value neutral";
    rbSub.textContent = "";
  }

  // ── Setup factors ─────────────────────────────────────────────────────────
  const factorContainer = document.getElementById("factorRows");
  factorContainer.innerHTML = "";
  (m.setup_factors || []).forEach((f) => {
    const earned = f.points ?? 0;
    const max = f.max ?? 1;
    const ratio = max > 0 ? earned / max : 0;
    const barColor =
      ratio >= 0.67
        ? "var(--bull)"
        : ratio >= 0.34
          ? "var(--grade-a)"
          : "var(--bear)";

    const row = document.createElement("div");
    row.innerHTML = `
      <div class="factor-row">
        <div class="factor-name">${f.name}</div>
        <div class="factor-bar-track">
          <div class="factor-bar-fill" style="width:${Math.round(ratio * 100)}%;background:${barColor}"></div>
        </div>
        <div class="factor-pts">${earned}/${max}</div>
      </div>
      ${f.detail ? `<div class="factor-detail">${f.detail}</div>` : ""}
    `;
    factorContainer.appendChild(row);
  });

  // ── Key levels ────────────────────────────────────────────────────────────
  const levelContainer = document.getElementById("keyLevelRows");
  levelContainer.innerHTML = "";
  (m.key_levels || []).slice(0, 8).forEach((l) => {
    const dist = l.pct ?? null;  // backend field is 'pct'
    const isAbove = dist !== null && dist >= 0;
    const pctCls = isAbove ? "above" : "below";
    const pctText =
      dist !== null ? (dist >= 0 ? "+" : "") + dist.toFixed(2) + "%" : "";

    const row = document.createElement("div");
    row.className = "key-level-row";
    row.innerHTML = `
      <div class="key-level-arrow">${isAbove ? "↑" : "↓"}</div>
      <div class="key-level-price">${formatPrice(l.price)}</div>
      <div class="key-level-pct ${pctCls}">${pctText}</div>
      <div class="key-level-label">${l.label}</div>
      ${l.grade ? `<div class="key-level-grade ${l.grade}">${l.grade}</div>` : ""}
    `;
    levelContainer.appendChild(row);
  });

  // ── Stop suggestion ───────────────────────────────────────────────────────
  const stopEl = document.getElementById("stopSuggestion");
  const stopAbove = m.stop_above ?? null;
  const stopBelow = m.stop_below ?? null;

  if (stopAbove !== null || stopBelow !== null) {
    stopEl.innerHTML = "";
    if (stopBelow !== null) {
      stopEl.innerHTML += `
        <div class="stop-item">
          <span style="color:var(--bull)">▲ Long</span>
          <strong>Stop: ${formatPrice(stopBelow)}</strong>
        </div>`;
    }
    if (stopAbove !== null) {
      stopEl.innerHTML += `
        <div class="stop-item">
          <span style="color:var(--bear)">▼ Short</span>
          <strong>Stop: ${formatPrice(stopAbove)}</strong>
        </div>`;
    }
    stopEl.style.display = "flex";
  } else {
    stopEl.style.display = "none";
  }
}

// ── Formatters ────────────────────────────────────────────────────────────────

function formatSignalName(name) {
  return name
    .replace(/_bullish$/, "")
    .replace(/_bearish$/, "")
    .replace(/_up$/, "")
    .replace(/_down$/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatPrice(price) {
  if (!price && price !== 0) return "—";
  if (price >= 1000)
    return "$" + price.toLocaleString("en-US", { maximumFractionDigits: 2 });
  if (price >= 1) return "$" + parseFloat(price.toFixed(4)).toString();
  return "$" + parseFloat(price.toFixed(6)).toString();
}

function formatValue(val) {
  if (val === null || val === undefined) return "—";
  if (Math.abs(val) >= 1000)
    return val.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (Math.abs(val) >= 1) return parseFloat(val.toFixed(2)).toString();
  return parseFloat(val.toFixed(5)).toString();
}

function formatLargeNumber(n) {
  if (!n) return "0";
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(2) + "B";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toFixed(0);
}

// ── UI state ──────────────────────────────────────────────────────────────────

function setLoading(on) {
  document.getElementById("loadingState").style.display = on ? "block" : "none";
  document.getElementById("analyzeBtn").disabled = on;
  if (on) {
    document.getElementById("emptyState").style.display = "none";
    document.getElementById("results").style.display = "none";
  }
}

function showError(msg) {
  const el = document.getElementById("errorBanner");
  el.textContent = msg;
  el.style.display = "block";
}

function clearError() {
  document.getElementById("errorBanner").style.display = "none";
}

// ── VPVR ──────────────────────────────────────────────────────────────────────

function drawVPVR() {
  const overlay = document.getElementById("vpvrOverlay");
  overlay.innerHTML = "";
  if (!candleSeries || !lastCandles || lastCandles.length < 10) return;

  const W = 72, H = 320, N = 50;
  const minP = Math.min(...lastCandles.map((c) => c.low));
  const maxP = Math.max(...lastCandles.map((c) => c.high));
  const bSize = (maxP - minP) / N;
  if (bSize <= 0) return;

  const buckets = Array(N).fill(0);
  for (const c of lastCandles) {
    const range = c.high - c.low;
    if (range <= 0) {
      const bi = Math.min(Math.floor((c.close - minP) / bSize), N - 1);
      buckets[bi] += c.volume;
    } else {
      for (let i = 0; i < N; i++) {
        const bLow = minP + i * bSize;
        const overlap = Math.max(0, Math.min(c.high, bLow + bSize) - Math.max(c.low, bLow));
        buckets[i] += c.volume * (overlap / range);
      }
    }
  }

  const maxVol = Math.max(...buckets);
  const pocIdx = buckets.indexOf(maxVol);
  const vah = findVAH(buckets, maxVol, N);
  const val = findVAL(buckets, maxVol, N);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.style.cssText = "display:block";

  for (let i = 0; i < N; i++) {
    if (buckets[i] <= 0) continue;
    const bucketMid = minP + (i + 0.5) * bSize;
    const y = candleSeries.priceToCoordinate(bucketMid);
    if (y === null || y < 0 || y > H) continue;

    const yTop = candleSeries.priceToCoordinate(minP + (i + 1) * bSize);
    const yBot = candleSeries.priceToCoordinate(minP + i * bSize);
    const bh = Math.max(1, Math.abs((yBot ?? y + 3) - (yTop ?? y - 3)) - 1);
    const bw = Math.max(2, Math.round((buckets[i] / maxVol) * (W - 4)));

    const isPOC = i === pocIdx;
    const inVA = i >= val && i <= vah;
    let fill = "#2a3040";
    if (isPOC) fill = "#f59e0b";
    else if (inVA) fill = "#6366f1";

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", W - bw);
    rect.setAttribute("y", y - bh / 2);
    rect.setAttribute("width", bw);
    rect.setAttribute("height", bh);
    rect.setAttribute("fill", fill);
    rect.setAttribute("opacity", isPOC ? "1" : "0.65");
    svg.appendChild(rect);
  }

  // POC dashed line
  const pocY = candleSeries.priceToCoordinate(minP + (pocIdx + 0.5) * bSize);
  if (pocY !== null && pocY >= 0 && pocY <= H) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", "0"); line.setAttribute("y1", pocY);
    line.setAttribute("x2", W);  line.setAttribute("y2", pocY);
    line.setAttribute("stroke", "#f59e0b");
    line.setAttribute("stroke-width", "1");
    line.setAttribute("stroke-dasharray", "3,2");
    svg.appendChild(line);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", "1");
    label.setAttribute("y", Math.max(8, pocY - 2));
    label.setAttribute("fill", "#f59e0b");
    label.setAttribute("font-size", "8");
    label.setAttribute("font-family", "monospace");
    label.textContent = "POC";
    svg.appendChild(label);
  }

  overlay.appendChild(svg);
}

function findVAH(buckets, maxVol, N) {
  const pocIdx = buckets.indexOf(maxVol);
  const total = buckets.reduce((a, b) => a + b, 0);
  const target = total * 0.7;
  let cum = buckets[pocIdx];
  let hi = pocIdx, lo = pocIdx;
  while (cum < target && (hi < N - 1 || lo > 0)) {
    const addHi = hi < N - 1 ? buckets[hi + 1] : 0;
    const addLo = lo > 0 ? buckets[lo - 1] : 0;
    if (addHi >= addLo) hi++; else lo--;
    cum += Math.max(addHi, addLo);
  }
  return hi;
}

function findVAL(buckets, maxVol, N) {
  const pocIdx = buckets.indexOf(maxVol);
  const total = buckets.reduce((a, b) => a + b, 0);
  const target = total * 0.7;
  let cum = buckets[pocIdx];
  let hi = pocIdx, lo = pocIdx;
  while (cum < target && (hi < N - 1 || lo > 0)) {
    const addHi = hi < N - 1 ? buckets[hi + 1] : 0;
    const addLo = lo > 0 ? buckets[lo - 1] : 0;
    if (addHi >= addLo) hi++; else lo--;
    cum += Math.max(addHi, addLo);
  }
  return lo;
}

// ── Liquidation heatmap ───────────────────────────────────────────────────────

async function fetchLiquidations(symbol) {
  try {
    const res = await fetch(`/api/liquidations/${encodeURIComponent(symbol)}`);
    const data = await res.json();
    if (data.success && data.liq_levels && data.liq_levels.length) {
      renderLiquidations(data);
    }
  } catch (err) {
    console.warn("Liquidation fetch failed:", err);
  }
}

function renderLiquidations(data) {
  const panel = document.getElementById("liqPanel");
  const body  = document.getElementById("liqBody");
  const meta  = document.getElementById("liqHeaderMeta");

  const totalB = formatLargeNumber(data.total_oi_usd);
  const lPct   = Math.round(data.long_pct * 100);
  const sPct   = Math.round(data.short_pct * 100);
  meta.textContent = `OI $${totalB}  ·  ${lPct}% longs / ${sPct}% shorts (est.)`;

  body.innerHTML = "";

  // Separate and sort: shorts above price (descending by price), longs below (ascending proximity)
  const shorts = data.liq_levels.filter((l) => l.side === "short").sort((a, b) => b.price - a.price);
  const longs  = data.liq_levels.filter((l) => l.side === "long" ).sort((a, b) => a.price - b.price);

  const buildRow = (l) => {
    const row = document.createElement("div");
    row.className = "liq-row";
    const pct = l.pct_from_price;
    const side = l.side;
    const barW = Math.round(l.relative_size * 100);
    row.innerHTML = `
      <span class="liq-pct ${side}">${pct > 0 ? "+" : ""}${pct.toFixed(1)}%</span>
      <span class="liq-leverage">${l.leverage}×</span>
      <div class="liq-bar-track">
        <div class="liq-bar-fill ${side}" style="width:${barW}%"></div>
      </div>
      <span class="liq-size">$${formatLargeNumber(l.size_usd)}</span>
      <span class="liq-price-label">${formatPrice(l.price)}</span>
    `;
    return row;
  };

  shorts.forEach((l) => body.appendChild(buildRow(l)));

  const priceRow = document.createElement("div");
  priceRow.className = "liq-current-price";
  priceRow.textContent = `▶ Current price: ${formatPrice(data.price)}`;
  body.appendChild(priceRow);

  longs.forEach((l) => body.appendChild(buildRow(l)));

  panel.style.display = "block";
}

// ── Signal persistence tracker ────────────────────────────────────────────────

async function fetchPersistence(symbol) {
  try {
    const res  = await fetch(`/api/runs/${encodeURIComponent(symbol)}?limit=20`);
    const data = await res.json();
    if (data.success && data.runs && data.runs.length >= 2) {
      renderPersistence(data.runs);
    }
  } catch (err) {
    console.warn("Persistence fetch failed:", err);
  }
}

function renderPersistence(runs) {
  const panel  = document.getElementById("persistencePanel");
  const track  = document.getElementById("persistenceTrack");
  const streak = document.getElementById("persistenceStreak");
  track.innerHTML = "";

  // runs are newest-first; display oldest-to-newest left-to-right
  const ordered = [...runs].reverse();

  ordered.forEach((run, i) => {
    const dir   = run.bias_direction || "neutral";
    const score = Math.abs(run.bias_score || 0);
    const label = dir === "bullish" ? "▲" : dir === "bearish" ? "▼" : "—";
    const isCurrent = i === ordered.length - 1;

    const dot = document.createElement("div");
    dot.className = `persistence-dot ${dir}${isCurrent ? " current" : ""}`;
    dot.style.opacity = 0.35 + score * 0.65;
    dot.textContent = label;
    dot.title = `${new Date(run.created_at).toLocaleString()}  ·  ${dir}  ·  score ${(run.bias_score || 0).toFixed(2)}  ·  ${run.total_signals} signals  ·  ${formatPrice(run.price)}`;
    track.appendChild(dot);
  });

  // Compute current streak
  const latest = runs[0]?.bias_direction;
  let streakCount = 0;
  for (const r of runs) {
    if (r.bias_direction === latest) streakCount++;
    else break;
  }
  if (latest && latest !== "neutral" && streakCount > 1) {
    streak.innerHTML = `<span>${streakCount}</span>-run ${latest} streak`;
  } else {
    streak.textContent = "";
  }

  panel.style.display = "block";
}

// ── Export ────────────────────────────────────────────────────────────────────

function exportJSON() {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
  triggerDownload(blob, `${lastResult.symbol}_all_${exportTimestamp()}.json`);
}

// Compact markdown export for feeding to an LLM — keeps only grade A/B
// signals (drops the noisy grade-C tail) and states each fact once instead
// of the JSON export's all_signals/primary_signals/families triplication.
function exportSummary() {
  if (!lastResult) return;
  const lines = [`# ${lastResult.symbol} — Analysis Summary`, `_${new Date().toISOString()}_`, ""];

  for (const key of ["short", "mid", "long"]) {
    const d = lastResult[key];
    if (!d) continue;
    const sm = d.scalp_metrics || {};

    lines.push(`## ${d.horizon_label || key} (${(d.timeframes || []).join("/")})`);
    lines.push(`Price: $${d.price} · Bias: **${d.bias.direction}** (score ${d.bias.score}, bull ${d.bias.bull_score} / bear ${d.bias.bear_score})`);

    if (sm.setup_score != null) {
      lines.push(`Setup: **${sm.setup_label}** (${sm.setup_score}/10) — ` +
        (sm.setup_factors || []).map((f) => `${f.name} ${f.points}/${f.max} (${f.detail})`).join("; "));
    }
    if (sm.volume_ratio != null) {
      lines.push(`Volume: ${sm.volume_ratio}x avg, ${sm.volume_trend || "n/a"}` +
        (sm.atr_pct != null ? ` · ATR: ${sm.atr_pct}% (2% target = ${sm.atrs_for_2pct}x ATR, ${sm.target_realistic ? "realistic" : "stretched"})` : ""));
    }
    if (sm.tf_alignment) {
      const ta = sm.tf_alignment;
      lines.push(`TF alignment: ${ta.aligned_count}/${ta.total_count} ${ta.strength} ${ta.aligned_direction} (` +
        Object.entries(ta.timeframes || {}).map(([tf, dir]) => `${tf}:${dir}`).join(", ") + ")");
    }

    const sigs = (d.all_signals || []).filter((s) => s.grade === "A" || s.grade === "B");
    lines.push("", "**Signals (A/B grade):**");
    if (!sigs.length) {
      lines.push("_none_");
    } else {
      sigs
        .sort((a, b) => (a.grade === b.grade ? 0 : a.grade === "A" ? -1 : 1))
        .forEach((s) => {
          const val = s.price_level != null ? `level=$${s.price_level}` : s.indicator_value != null ? `value=${s.indicator_value}` : "";
          lines.push(`- [${s.grade}][${s.direction}][${s.timeframe}] ${s.signal_name}` +
            (val ? ` — ${val}` : "") + (s.candles_ago != null ? ` (${s.candles_ago} candles ago)` : ""));
        });
    }

    if ((sm.key_levels || []).length) {
      lines.push("", "**Key levels:**");
      sm.key_levels.forEach((l) => {
        lines.push(`- ${l.pct > 0 ? "+" : ""}${l.pct}% $${l.price} ${l.label} (${l.grade}, ${l.direction})`);
      });
    }
    lines.push("");
  }

  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  triggerDownload(blob, `${lastResult.symbol}_summary_${exportTimestamp()}.md`);
}

function exportCSV() {
  if (!lastResult) return;
  const rows = [["symbol","horizon","timestamp","price","bias_direction","bias_score","family","signal_name","grade","direction","timeframe","role","price_level","candles_ago"]];
  for (const key of ["short", "mid", "long"]) {
    const data = lastResult[key];
    if (!data) continue;
    const { symbol, horizon, timestamp, price, bias, families } = data;
    Object.entries(families || {}).forEach(([, fam]) => {
      (fam.active_signals || []).forEach((s) => {
        rows.push([
          symbol, horizon, timestamp, price,
          bias?.direction ?? "", bias?.score ?? "",
          fam.name, s.signal_name, s.grade, s.direction,
          s.timeframe, s.role, s.price_level ?? "", s.candles_ago ?? "",
        ]);
      });
    });
  }
  const csv = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  triggerDownload(blob, `${lastResult.symbol}_all_${exportTimestamp()}.csv`);
}

function exportTimestamp() {
  return new Date().toISOString().slice(0, 16).replace(/[T:]/g, "-");
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
