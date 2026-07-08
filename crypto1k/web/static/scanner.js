// Volume Scanner — front-end logic
'use strict';

const $ = (id) => document.getElementById(id);
let autoTimer = null;

// ── helpers ─────────────────────────────────────────────────────────────────
const fmtUsd = (n) => n == null ? '—'
  : '$' + (Math.abs(n) >= 1e6 ? (n/1e6).toFixed(2)+'M'
        : Math.abs(n) >= 1e3 ? (n/1e3).toFixed(1)+'K' : Number(n).toFixed(0));
const fmtPrice = (n) => {
  if (n == null) return '—';
  if (n >= 1000)    return '$' + n.toFixed(2);
  if (n >= 1)       return '$' + n.toFixed(4);
  if (n >= 0.01)    return '$' + n.toFixed(5);
  if (n >= 0.0001)  return '$' + n.toFixed(7);
  // For very small shitcoin prices: enough decimals to show 3 sig figs
  const dec = Math.max(8, Math.ceil(-Math.log10(n)) + 2);
  return '$' + n.toFixed(Math.min(dec, 12));
};
const pct = (n) => (n == null ? '—' : (n>=0?'+':'') + n.toFixed(1) + '%');
const cls = (n) => n > 0 ? 'pos' : n < 0 ? 'neg' : '';
const ago = (iso) => {
  if (!iso) return '—';
  const d = (Date.now() - new Date(iso).getTime()) / 1000;
  if (isNaN(d) || d < 0) return 'just now';
  if (d < 60) return Math.floor(d)+'s ago';
  if (d < 3600) return Math.floor(d/60)+'m ago';
  return Math.floor(d/3600)+'h ago';
};
// Opens the full signal-analyzer page for a coin (BASE-USDT convention)
const analyzerUrl = (sym) => '/symbol/' + encodeURIComponent((sym||'').toUpperCase() + '-USDT');

async function api(url, opts) {
  const res = await fetch(url, opts);
  return res.json();
}

// ── monitor ─────────────────────────────────────────────────────────────────
async function loadMonitor() {
  const d = await api('/api/scanner/monitor');
  const on = d.running;
  $('monStatus').innerHTML = `<span class="dot ${on?'on':'off'}"></span>${on?'Running':'Stopped'}`;
  $('monInterval').textContent = d.interval_seconds + 's';
  $('monLastRun').textContent = d.last_run ? ago(d.last_run) : 'never';
  $('monEmail').textContent = d.email_ready ? `Ready (${d.recipients})` : 'Not configured';
  $('monToggle').textContent = on ? 'Stop monitor' : 'Start monitor';
  $('monToggle').className = on ? 'btn-red' : 'btn-green';
}

$('monToggle').onclick = async () => {
  const running = $('monToggle').textContent.startsWith('Stop');
  await api('/api/scanner/monitor', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ action: running ? 'stop' : 'start' })
  });
  loadMonitor();
};

$('testEmailBtn').onclick = async () => {
  $('msg').textContent = 'Sending test email…';
  $('msg').className = 'loading';
  const d = await api('/api/scanner/test-email', { method:'POST' });
  $('msg').className = d.success ? 'pos' : 'neg';
  $('msg').textContent = d.success ? `✓ Test email sent to ${d.recipients} recipient(s).`
                                   : `✗ ${d.message}`;
};

// ── watchlist ───────────────────────────────────────────────────────────────
async function loadWatchlist() {
  const d = await api('/api/scanner/watchlist');
  renderChips(d.watchlist);
}

function renderChips(list) {
  const box = $('chips');
  if (!list || !list.length) { box.innerHTML = '<span class="empty">No coins yet — add some above.</span>'; return; }
  box.innerHTML = '';
  list.forEach(sym => {
    const c = document.createElement('div');
    c.className = 'chip';
    c.innerHTML = `<b>${sym}</b><span class="x" title="Remove">×</span>`;
    c.querySelector('.x').onclick = async () => {
      await api('/api/scanner/watchlist/' + encodeURIComponent(sym), { method:'DELETE' });
      loadWatchlist();
    };
    box.appendChild(c);
  });
}

async function addWatchlist() {
  const val = $('wlInput').value.trim();
  if (!val) return;
  await api('/api/scanner/watchlist', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ symbols: val })
  });
  $('wlInput').value = '';
  loadWatchlist();
}
$('wlAdd').onclick = addWatchlist;
$('wlInput').addEventListener('keydown', e => { if (e.key === 'Enter') addWatchlist(); });

// ── scan ────────────────────────────────────────────────────────────────────
async function scan() {
  $('scanMsg').textContent = 'Scanning…';
  $('scanBtn').disabled = true;
  try {
    const d = await api('/api/scanner/scan', { method:'POST',
      headers:{'Content-Type':'application/json'}, body:'{}' });
    renderResults(d.results || []);
    $('scanMsg').textContent = `Scanned ${d.count || 0} coin(s) · ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    $('scanMsg').textContent = 'Scan failed: ' + e;
  } finally {
    $('scanBtn').disabled = false;
  }
}
$('scanBtn').onclick = scan;

function renderResults(results) {
  const body = $('resultsBody');
  if (!results.length) {
    body.innerHTML = '<tr><td colspan="11" class="empty" style="padding:20px;text-align:center">Watchlist is empty or nothing returned.</td></tr>';
    return;
  }
  body.innerHTML = '';
  results.forEach((r, i) => {
    if (r.error) {
      body.insertAdjacentHTML('beforeend',
        `<tr class="row"><td></td><td class="sym">${r.symbol}</td>
         <td colspan="9" class="neg">${r.error}</td></tr>`);
      return;
    }
    const m = r.metrics || {};
    const sm = r.smart_money;
    const flags = [];
    if (r.is_alert) flags.push('<span class="alert-tag">ALERT' + (r.alert_path === 'smart' ? ' 🧠' : '') + '</span>');
    if (sm && sm.smart_buys > 0)
      flags.push(`<span class="smart-tag" title="tracked smart wallet bought ${fmtUsd(sm.smart_buy_usd)} in last ${sm.lookback_minutes}m">🧠 smart $</span>`);
    else if (sm && Math.abs(sm.net_flow_usd) >= 5000)
      flags.push(`<span class="whale-tag ${sm.net_flow_usd > 0 ? 'pos' : 'neg'}" title="whale net flow over last ${sm.lookback_minutes}m (trades ≥ ${fmtUsd(sm.min_trade_usd)})">🐳 ${sm.net_flow_usd > 0 ? '+' : ''}${fmtUsd(sm.net_flow_usd)}</span>`);
    if (r.wash_warning) flags.push('<span class="wash-tag">wash?</span>');
    if (r.thin_exit_warning) flags.push(`<span class="wash-tag" title="liquidity is only ${m.liq_mcap_pct}% of market cap — exit door is thin">thin exit</span>`);
    if (!r.passes_filters) flags.push(`<span class="sub" title="${(r.filter_fails||[]).join('; ')}">filtered</span>`);

    const row = document.createElement('tr');
    row.className = 'row';
    row.innerHTML = `
      <td><span class="caret" data-i="${i}">▸</span></td>
      <td><div class="sym">${r.symbol}</div><div class="sub">${r.chain||''} · ${r.dex||''}</div></td>
      <td class="num">${fmtPrice(r.price_usd)}</td>
      <td><span class="badge ${r.label}">${r.score} ${r.label}</span></td>
      <td class="num">${(m.vol_pace_1h ?? 0).toFixed(1)}×</td>
      <td class="num">${(m.vol_pace_5m ?? 0).toFixed(1)}×</td>
      <td class="num ${cls(r.price_change?.h1)}">${pct(r.price_change?.h1)}</td>
      <td class="num">${(m.buy_ratio_1h ?? 0)}:1</td>
      <td class="num">${fmtUsd(r.liquidity_usd)}</td>
      <td>${flags.join(' ') || '—'}</td>
      <td style="white-space:nowrap">
        <a href="${analyzerUrl(r.symbol)}" class="analyze-link" title="Open signal analyzer">analyze ↗</a>
        ${r.url ? `<br><a href="${r.url}" target="_blank" class="sub">chart ↗</a>` : ''}
      </td>`;
    body.appendChild(row);

    const sig = document.createElement('tr');
    sig.className = 'signals';
    const items = (r.signals || []).map(s =>
      `<li><b>${s.name.replace(/_/g,' ')}</b>
       <span class="dir-${s.direction}">(${s.direction})</span> — ${s.detail}</li>`).join('')
       || '<li class="sub">No individual signals fired.</li>';
    sig.innerHTML = `<td></td><td colspan="10">
       <div class="summary">${r.summary || ''}</div><ul>${items}</ul></td>`;
    body.appendChild(sig);

    row.querySelector('.caret').onclick = (e) => {
      sig.classList.toggle('open');
      e.target.textContent = sig.classList.contains('open') ? '▾' : '▸';
    };
  });
}

// ── live view ───────────────────────────────────────────────────────────────
// Reflects the latest scan — whether triggered by the background monitor or a
// manual "Scan now" — so you can see what's being watched in real time.
function updateLiveBadge(running, lastRun) {
  const b = $('liveBadge');
  if (running) {
    b.className = 'live-badge live';
    b.innerHTML = `<span class="dot on"></span>LIVE · monitor on · updated ${ago(lastRun)}`;
  } else {
    b.className = 'live-badge idle';
    b.innerHTML = `<span class="dot off"></span>${lastRun ? 'monitor off · last scan ' + ago(lastRun) : 'idle · monitor off'}`;
  }
}

async function loadLive() {
  try {
    const d = await api('/api/scanner/live');
    updateLiveBadge(d.running, d.last_run);
    if (d.results && d.results.length) {
      renderResults(d.results);
      $('scanMsg').textContent = `${d.results.length} coin(s) · ${d.running ? 'auto' : 'manual'} · ${d.last_run ? ago(d.last_run) : ''}`;
    }
  } catch (e) { /* keep last view on transient errors */ }
}

function startLive() { if (!autoTimer) { loadLive(); autoTimer = setInterval(loadLive, 20000); } }
function stopLive()  { clearInterval(autoTimer); autoTimer = null; }

$('autoRefresh').checked = true;
$('autoRefresh').onchange = (e) => e.target.checked ? startLive() : stopLive();

// ── alerts ──────────────────────────────────────────────────────────────────
async function loadAlerts() {
  const d = await api('/api/scanner/alerts?limit=30');
  const box = $('alertsBox');
  if (!d.alerts || !d.alerts.length) { box.innerHTML = '<span class="empty">No alerts yet.</span>'; return; }
  box.innerHTML = '';
  d.alerts.forEach(a => {
    const card = document.createElement('div');
    card.className = 'alert-card';
    card.innerHTML = `
      <span class="badge ${a.label}">${a.score} ${a.label}</span>
      <div class="meta">
        <b>${a.symbol}</b> · <span class="${cls(a.price_change_1h)}">${pct(a.price_change_1h)}</span>
        · vol ${(a.vol_pace_1h||0).toFixed(1)}× · liq ${fmtUsd(a.liquidity_usd)}
        <div class="sub">${a.summary || ''}</div>
      </div>
      <div style="text-align:right">
        <a href="${analyzerUrl(a.symbol)}" class="analyze-link">analyze ↗</a><br>
        ${a.url ? `<a href="${a.url}" target="_blank" class="sub">chart ↗</a><br>`:''}
        <span class="when">${ago(a.created_at)}</span>
      </div>`;
    box.appendChild(card);
  });
}

// ── init ────────────────────────────────────────────────────────────────────
loadMonitor();
loadWatchlist();
loadAlerts();
startLive();                       // live results poll (every 20s)
setInterval(loadMonitor, 30000);
setInterval(loadAlerts, 30000);
