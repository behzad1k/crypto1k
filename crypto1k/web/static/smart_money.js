// Smart Money — front-end logic
'use strict';

const $ = (id) => document.getElementById(id);

// ── helpers ─────────────────────────────────────────────────────────────────
const fmtUsd = (n) => n == null ? '—'
  : '$' + (Math.abs(n) >= 1e6 ? (n/1e6).toFixed(2)+'M'
        : Math.abs(n) >= 1e3 ? (n/1e3).toFixed(1)+'K' : Number(n).toFixed(0));
const fmtPrice = (n) => {
  if (n == null) return '—';
  if (n >= 1000)   return '$' + n.toFixed(2);
  if (n >= 1)      return '$' + n.toFixed(4);
  if (n >= 0.01)   return '$' + n.toFixed(5);
  if (n >= 0.0001) return '$' + n.toFixed(7);
  const dec = Math.max(8, Math.ceil(-Math.log10(n)) + 2);
  return '$' + n.toFixed(Math.min(dec, 12));
};
const pct = (n) => (n == null ? '—' : (n >= 0 ? '+' : '') + Number(n).toFixed(1) + '%');
const cls = (n) => n > 0 ? 'pos' : n < 0 ? 'neg' : '';
const shortAddr = (a) => !a ? '?' : a.length > 12 ? a.slice(0, 6) + '…' + a.slice(-4) : a;
const ago = (iso) => {
  if (!iso) return '—';
  // block_ts is naive UTC — force the Z so the browser doesn't read it as local
  const t = new Date(iso.includes('Z') || iso.includes('+') ? iso : iso + 'Z').getTime();
  const d = (Date.now() - t) / 1000;
  if (isNaN(d) || d < 0) return 'just now';
  if (d < 60) return Math.floor(d) + 's ago';
  if (d < 3600) return Math.floor(d / 60) + 'm ago';
  if (d < 86400) return Math.floor(d / 3600) + 'h ago';
  return Math.floor(d / 86400) + 'd ago';
};

async function api(url, opts) {
  const res = await fetch(url, opts);
  return res.json();
}

function walletCell(w) {
  const label = w.label ? `<b>${w.label}</b> <span class="sub mono">${shortAddr(w.wallet || w.address)}</span>`
                        : `<span class="mono">${shortAddr(w.wallet || w.address)}</span>`;
  return `<span title="${w.wallet || w.address}">${label}</span>`
       + (w.chain ? `<div class="sub">${w.chain}</div>` : '');
}

// ── overview ────────────────────────────────────────────────────────────────
async function loadOverview() {
  const d = await api('/api/smart-money/overview');
  $('stTrades').textContent  = d.trades_recorded ?? '—';
  $('stWallets').textContent = d.wallets_seen ?? '—';
  $('stScored').textContent  = d.buys_scored ?? '—';
  $('stManual').textContent  = d.manual_wallets ?? '—';
  $('stAuto').textContent    = d.auto_qualified ?? '—';
}

$('backfillBtn').onclick = async () => {
  $('msg').textContent = 'Scoring pending whale buys (one candle fetch per pool)…';
  $('msg').className = 'loading';
  $('backfillBtn').disabled = true;
  try {
    const d = await api('/api/smart-money/backfill', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
    });
    $('msg').className = d.success ? 'pos' : 'neg';
    // innerHTML, not textContent: the status icon is markup.
    $('msg').innerHTML = d.success
      ? icon('check-circle', 'ph-pos') + ` ${escapeAttr(d.pools)} pool(s) processed, ${escapeAttr(d.scored)} buy(s) scored.`
      : icon('x-circle', 'ph-neg') + ' backfill failed';
    refreshAll();
  } catch (e) {
    $('msg').className = 'neg';
    $('msg').innerHTML = icon('x-circle', 'ph-neg') + ' ' + escapeAttr(e);
  } finally {
    $('backfillBtn').disabled = false;
  }
};

// ── track a wallet ──────────────────────────────────────────────────────────
async function addWallet() {
  const address = $('walletAddr').value.trim();
  if (!address) return;
  await api('/api/smart-money/wallets', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      address,
      chain: $('walletChain').value.trim(),
      label: $('walletLabel').value.trim(),
    })
  });
  $('walletAddr').value = $('walletChain').value = $('walletLabel').value = '';
  refreshAll();
}
$('walletAdd').onclick = addWallet;
$('walletAddr').addEventListener('keydown', e => { if (e.key === 'Enter') addWallet(); });

// ── movements feed ──────────────────────────────────────────────────────────
async function loadFeed() {
  const d = await api('/api/smart-money/feed?limit=100');
  const body = $('feedBody');
  if (!d.feed || !d.feed.length) return; // keep the empty-state row
  body.innerHTML = '';
  d.feed.forEach(t => {
    body.insertAdjacentHTML('beforeend', `
      <tr class="row">
        <td class="sub">${ago(t.block_ts)}</td>
        <td>${walletCell(t)}</td>
        <td><span class="badge ${t.side}">${t.side.toUpperCase()}</span></td>
        <td><b>${t.symbol || '?'}</b> <span class="sub">${t.chain || ''}</span></td>
        <td class="num">${fmtUsd(t.amount_usd)}</td>
        <td class="num">${fmtPrice(t.price_usd)}</td>
        <td class="num ${cls(t.ret_1h_pct)}">${pct(t.ret_1h_pct)}</td>
        <td class="num ${cls(t.ret_24h_pct)}">${pct(t.ret_24h_pct)}</td>
      </tr>`);
  });
}

// ── leaderboard ─────────────────────────────────────────────────────────────
async function loadLeaderboard() {
  const d = await api('/api/smart-money/wallets?min_buys=1');
  const body = $('lbBody');
  if (!d.wallets || !d.wallets.length) return; // keep the empty-state row
  body.innerHTML = '';
  d.wallets.forEach((w, i) => {
    const status = w.manually_tracked
      ? `<span class="badge manual">${w.manual_source === 'auto' ? 'AUTO' : 'TRACKED'}</span>`
      : w.is_smart ? `<span class="badge auto">AUTO ${icon('check-circle')}</span>`
      : '<span class="sub">observed</span>';
    const trackBtn = w.manually_tracked
      ? `<button class="btn-ghost btn-mini" data-untrack="${w.wallet}" data-chain="${w.chain||''}">untrack</button>`
      : `<button class="btn-ghost btn-mini" data-track="${w.wallet}" data-chain="${w.chain||''}">track</button>`;

    const row = document.createElement('tr');
    row.className = 'row';
    row.innerHTML = `
      <td><span class="caret" data-i="${i}">▸</span></td>
      <td>${walletCell(w)}</td>
      <td>${status}</td>
      <td class="num">${w.buys ?? 0}</td>
      <td class="num">${w.scored_buys ?? 0}</td>
      <td class="num">${w.win_rate == null ? '—' : Math.round(w.win_rate * 100) + '%'}</td>
      <td class="num ${cls(w.avg_ret_1h)}">${pct(w.avg_ret_1h)}</td>
      <td class="num ${cls(w.avg_ret_24h)}">${pct(w.avg_ret_24h)}</td>
      <td class="num">${fmtUsd(w.total_usd)}</td>
      <td class="num">${w.pools ?? 0}</td>
      <td class="sub">${ago(w.last_seen)}</td>
      <td>${trackBtn}</td>`;
    body.appendChild(row);

    // expandable: this wallet's recent trades
    const detail = document.createElement('tr');
    detail.className = 'wallet-trades';
    detail.style.display = 'none';
    detail.innerHTML = `<td></td><td colspan="11"><span class="sub">loading…</span></td>`;
    body.appendChild(detail);

    row.querySelector('.caret').onclick = async (e) => {
      const open = detail.style.display !== 'none';
      detail.style.display = open ? 'none' : 'table-row';
      e.target.textContent = open ? '▸' : '▾';
      if (!open) {
        const dd = await api('/api/smart-money/wallet/' + encodeURIComponent(w.wallet));
        const items = (dd.trades || []).slice(0, 15).map(t =>
          `<li><span class="badge ${t.side}">${t.side}</span>
           <b>${t.symbol || '?'}</b> ${fmtUsd(t.amount_usd)} @ ${fmtPrice(t.price_usd)}
           <span class="sub">${ago(t.block_ts)}</span>
           · +1h <span class="${cls(t.ret_1h_pct)}">${pct(t.ret_1h_pct)}</span>
           · +24h <span class="${cls(t.ret_24h_pct)}">${pct(t.ret_24h_pct)}</span></li>`
        ).join('') || '<li class="sub">no trades recorded</li>';
        detail.innerHTML = `<td></td><td colspan="11"><ul>${items}</ul></td>`;
      }
    };
  });

  // track / untrack buttons
  body.querySelectorAll('[data-track]').forEach(b => b.onclick = async () => {
    const label = prompt('Label for this wallet (optional):') || '';
    await api('/api/smart-money/wallets', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address: b.dataset.track, chain: b.dataset.chain, label })
    });
    refreshAll();
  });
  body.querySelectorAll('[data-untrack]').forEach(b => b.onclick = async () => {
    await api('/api/smart-money/wallets/' + encodeURIComponent(b.dataset.untrack)
              + '?chain=' + encodeURIComponent(b.dataset.chain), { method: 'DELETE' });
    refreshAll();
  });
}

// ── init ────────────────────────────────────────────────────────────────────
function refreshAll() { loadOverview(); loadFeed(); loadLeaderboard(); }
refreshAll();
setInterval(refreshAll, 30000);
