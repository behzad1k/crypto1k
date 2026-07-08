"""
═══════════════════════════════════════════════════════════════════════════════
  SCANNER CONFIGURATION  —  edit everything here
═══════════════════════════════════════════════════════════════════════════════

This is the ONE place to tune the shitcoin volume scanner. It is split into
clearly labelled sections:

  1. EMAIL          — your SMTP credentials + who receives alerts   (FILL THIS IN)
  2. DATA SOURCE    — DexScreener settings
  3. FILTERS        — gates a coin must pass before we even score it
  4. SIGNAL THRESHOLDS — what counts as "unusual" volume / momentum
  5. SCORING        — how the 0–10 validity score is weighted
  6. ALERTING       — when an email actually fires (kept deliberately lenient)
  7. MONITOR        — background loop timing

Nothing in the scanner logic hard-codes a threshold; they all come from here.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. EMAIL  ─────────────────────────────────────────────────────────  FILL IN ↓
# ═══════════════════════════════════════════════════════════════════════════════

EMAIL = {
    # --- Your sending account (e.g. a Gmail App Password, NOT your login password) ---
    "enabled": False,  # flip to True once filled in
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,  # 587 = STARTTLS (Gmail), 465 = SSL
    "use_ssl": False,  # True only if you use port 465
    "username": "behzadkoohyani@gmail.com",  # <- your email / SMTP login
    "password": "vkoq iyvb wkjd gieg",  # <- app password (Gmail: myaccount.google.com/apppasswords)
    "from_name": "Crypto1k",
    "from_address": "behzadkoohyani@gmail.com",  # usually same as username
}

# --- Who gets alerted. Add as many as you like. ---
RECIPIENTS = [
    "bhzd1k@gmail.com"
    # 'you@example.com',
    # 'friend@example.com',
]


# --- Telegram channel alerts (free, sends over HTTPS — works even when SMTP is
#     blocked). See telegram_notify.py for the 3-step setup. ---
TELEGRAM = {
    "enabled": True,  # flip to True once token + chat_id are filled in
    "bot_token": "8458351245:AAGpkWV7gFm8acEvlyjWbAgEpfKvsXHVgBI",  # from @BotFather (e.g. 123456:ABC-DEF...)
    "chat_id": "@s_crypto1k",  # "@your_channel" (public) or -100123... (private)
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA SOURCE  (DexScreener — free, no API key)
# ═══════════════════════════════════════════════════════════════════════════════

DEXSCREENER = {
    "base_url": "https://api.dexscreener.com",
    "timeout": 8,  # seconds per request
    # When a symbol resolves to multiple pairs, prefer pairs quoted in these
    # tokens (most reliable volume), and pick the one with the deepest liquidity.
    "preferred_quotes": ["USDC", "USDT", "WETH", "SOL", "WBNB", "BNB", "DAI"],
    # Restrict resolution to these chains (lowercase). Empty list = allow all.
    "allowed_chains": [],  # e.g. ['ethereum', 'solana', 'bsc', 'base']
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FILTERS  —  a coin is ignored entirely unless it passes these gates
# ═══════════════════════════════════════════════════════════════════════════════

FILTERS = {
    "min_liquidity_usd": 15_000,  # below this = likely rug / untradeable
    "min_volume_24h_usd": 20_000,  # ignore totally dead coins
    "min_txns_1h": 15,  # need real trades, not a single whale print
    "min_pair_age_hours": 1,  # avoid brand-new pairs (set 0 to allow any)
    # Data-sanity cap: thin pools (often Meteora DLMM) sometimes report broken
    # priceUsd / priceChange — e.g. "+501,497% in 1h". Any pair whose |price
    # change| over a window exceeds this is treated as corrupted data: it's
    # skipped during symbol resolution and, if it slips through, fails filters
    # so it never scores or alerts. A real coin won't move 50× in an hour.
    "max_price_change_1h_pct": 5_000.0,
    # Wash-trade sanity: volume/liquidity above this is suspicious but NOT
    # auto-rejected — it only caps the score and adds a warning to the email.
    "wash_turnover_ratio": 75.0,  # vol_24h / liquidity
    # Exit-door sanity: liquidity as a % of market cap. Below this, the pool is
    # too shallow relative to the token's valuation — you (or any holder) can't
    # exit without cratering the price. Warning + score cap, not a hard gate.
    "min_liq_to_mcap_pct": 3.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL THRESHOLDS  —  what we consider "unusual"
# ═══════════════════════════════════════════════════════════════════════════════
#
# Volume "pace ratio" = recent volume vs. its own 24h average for that window.
#   ratio of 1.0  → exactly average.   ratio of 5.0 → 5× hotter than normal.
#
THRESHOLDS = {
    # --- Volume surge (the headline "unusual volume" detector) ---
    "vol_pace_1h_notable": 2.5,  # 1h volume this many × the 24h hourly average
    "vol_pace_1h_strong": 5.0,
    "vol_pace_5m_notable": 3.0,  # 5m volume vs 24h 5-min average (acceleration)
    "vol_pace_5m_strong": 6.0,
    # --- Buy/sell pressure (txn count imbalance) ---
    "buy_ratio_notable": 1.5,  # buys/sells in the last hour
    "buy_ratio_strong": 2.5,
    # --- Price momentum (percent) ---
    "price_move_1h_notable": 3.0,  # |price change| over 1h, percent
    "price_move_1h_strong": 8.0,
    # --- 5m price momentum (fast detection — scores a move while it's young,
    #     before the rolling 1h window has caught up) ---
    "price_move_5m_notable": 1.5,
    "price_move_5m_strong": 3.0,
    # --- Volume vs market cap (is the whole valuation actually trading?) ---
    # 24h volume as a multiple of market cap. High = the move is being actively
    # traded, not a few prints in a thin pool. Meme coins in a real trend
    # commonly run 2–10× mcap in a day.
    "vol_mcap_notable": 2.0,
    "vol_mcap_strong": 5.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SCORING  —  weights for the 0–10 validity score (mirrors the setup score)
# ═══════════════════════════════════════════════════════════════════════════════

SCORING = {
    "max_volume_surge": 4.0,  # the headline factor
    "max_momentum": 3.0,  # price move + multi-window alignment
    "max_buy_pressure": 2.0,
    "max_activity": 1.0,  # liquidity + trade-count health
    # Smart-money bonus (added on top by the enrichment pass, total still
    # clamped at 10): whale net flow and tracked-smart-wallet buys.
    "max_smart_money": 1.5,
    # Score → label bands (out of 10)
    "label_strong": 8.0,
    "label_good": 6.0,
    "label_fair": 4.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ALERTING  —  when an email actually goes out  (kept LENIENT on purpose)
# ═══════════════════════════════════════════════════════════════════════════════

ALERTING = {
    "min_validity_score": 7.5,  # 0–10; lenient. Raise to be pickier.
    # Volume gate: a coin qualifies if it shows a 1h surge OR a 5m acceleration.
    # The 5m path catches setups heating up *right now* even when the full hour
    # still looks average (e.g. PENGU: 1h 1.1× but 5m 6.4×).
    "min_vol_pace_1h": 2.5,  # 1h volume vs its 24h hourly average
    "min_vol_pace_5m": 4.0,  # 5m volume vs its 24h 5-min average (acceleration)
    "require_bullish": True,  # True = only alert on bullish-leaning spikes
    # Don't re-alert the same coin again until this many minutes have passed…
    "cooldown_minutes": 120,
    # …unless the new score beats the best score alerted inside that window by
    # this much — a weak early alert must not mask the real pump an hour later.
    "cooldown_rearm_score_jump": 2.0,
    # FAST PATH — alert immediately on an extreme 5-minute spike, regardless of
    # the validity score. The score is dominated by 1h rolling windows, which
    # dilute a fresh move; this path uses only 5m data (volume pace, price
    # move, buy/sell flow) so a spike can alert on the first scan after it
    # starts. Wash-trading suspects (see wash_turnover_ratio) never qualify.
    "fast_path": {
        "enabled": True,
        "min_vol_pace_5m": 6.0,  # 5m volume ≥ this × its 24h 5-min average
        "min_price_move_5m_pct": 2.0,  # |5m price change| ≥ this
        # Flow imbalance in the last 5m: buys/sells for an up-move,
        # sells/buys for a down-move.
        "min_flow_ratio_5m": 1.5,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MONITOR  —  background scanning loop
# ═══════════════════════════════════════════════════════════════════════════════

MONITOR = {
    "enabled_on_start": True,  # auto-start the background loop when the app boots
    # Re-scan every minute. DexScreener allows ~300 req/min and a scan costs one
    # request per watchlist symbol, so this is safe for watchlists up to ~100.
    # At the old 600s a spike could be 10 minutes stale before we even looked.
    "interval_seconds": 60,
    "per_request_pause": 0.4,  # polite pause between DexScreener calls
}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DIGEST  —  daily Telegram summary of how yesterday's buy alerts paid off
# ═══════════════════════════════════════════════════════════════════════════════

DIGEST = {
    "enabled": True,
    # Send once per day, at the first monitor tick after this local hour.
    # Covers every buy alert whose 24h outcome window completed since the last
    # digest (so nothing is judged on partial data, and downtime self-heals).
    "hour_local": 9,
    "retry_minutes": 10,   # wait this long before retrying a failed send
    # Take-profit calibration: the digest footer reports how many alerts
    # reached at least this % gain within 24h (would your TP have filled?).
    "tp_target_pct": 5.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BTC IMPACT  —  market-regime scoring around (not inside) the coin score
# ═══════════════════════════════════════════════════════════════════════════════

BTC_IMPACT = {
    "enabled": True,
    # Deep WBTC/USDC pool (Uniswap v3, Ethereum) — one DexScreener quote gives
    # BTC price plus its 5m/1h/24h changes, on the API budget we already pay.
    "pair": ("ethereum", "0x99ac8cA7087fA4A2A1FB6357269965A2014ABc35"),
    "cache_seconds": 60,  # one BTC fetch per scan cycle at most
    # Regime score = 5 + m5%*w + h1%*w + h24%*w, clamped 0–10. Fresh moves
    # weigh hardest: a BTC dump minutes ago kills alt pumps faster than a
    # slow daily drift. Flat BTC ≈ 5, -2% hour ≈ 3, +2% hour ≈ 7.
    "score_weights": {"m5": 0.8, "h1": 1.0, "h24": 0.15},
    "risk_on_score": 6.5,   # regime label thresholds
    "risk_off_score": 3.5,
    # ENTRY GATE — suppress buy alerts when the regime score is below the
    # cutoff. OFF until the digest's regime breakdown shows where (and
    # whether) alert performance actually collapses; flip it on with an
    # evidence-based cutoff instead of a guessed one.
    "gate": {
        "enabled": False,
        "min_regime_score": 3.5,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SMART MONEY  —  whale trades + tracked-wallet movements
# ═══════════════════════════════════════════════════════════════════════════════
#
# Data source: GeckoTerminal's per-pool trades endpoint (free, keyless) — every
# trade includes the buyer/seller wallet address. We record whale-sized trades
# on the pools the scanner watches, score each buy 1h/24h later from OHLCV
# candles, and auto-promote wallets whose buys consistently precede pumps.
# Coverage note: we see wallets on WATCHED pools only, not their activity on
# pools we never scan — that's the trade-off of staying keyless and free.
#
SMART_MONEY = {
    "enabled": True,

    # --- What counts as a whale trade (recorded + analyzed) ---
    "min_trade_usd": 500.0,     # ignore dust; meme-coin whale threshold
    "big_trade_usd": 2_500.0,   # single print this size gets its own signal

    # --- Per-coin enrichment during a scan ---
    "lookback_minutes": 60,        # window for whale net-flow metrics
    "min_score_to_fetch": 4.0,     # only enrich coins already worth a look
    "cache_ttl_seconds": 240,      # per-pool trade cache (rate-limit shield)
    "max_fetch_per_cycle": 6,      # GeckoTerminal free tier is ~30 req/min
    "per_request_pause": 2.2,      # polite pause after each trades fetch

    # --- Signal thresholds ---
    "net_flow_notable_usd": 5_000.0,   # |whale buys - sells| in the window
    "net_flow_strong_usd": 20_000.0,

    # --- Score bonus split (capped at SCORING['max_smart_money']) ---
    "pts_net_flow_notable": 0.5,
    "pts_net_flow_strong": 1.0,
    "pts_big_trade": 0.5,
    "pts_smart_wallet_buy": 1.5,   # a tracked winner bought — the whole bonus

    # --- Alerting: a smart-wallet buy can fire an alert on its own ---
    # (third alert path next to 'standard' and 'fast'; still requires the coin
    # to pass filters and not be a wash-trading suspect)
    "alert_on_smart_wallet_buy": True,

    # --- Auto-qualification: when does a wallet become "smart money"? ---
    # A wallet's buys are scored 1h/24h later; a buy is a WIN if the price is
    # up by either threshold. Enough scored buys + a high win rate = promoted.
    "qualify": {
        "min_scored_buys": 5,
        "min_win_rate": 0.55,
        "win_ret_1h_pct": 5.0,     # +5% one hour after the buy, or…
        "win_ret_24h_pct": 10.0,   # …+10% within a day
    },

    # --- Outcome backfill (piggybacks on the monitor loop) ---
    "backfill_every_minutes": 15,
    "backfill_max_pools": 4,        # one OHLCV fetch per pool per run
    "min_trade_age_minutes": 70,    # 1h return must be observable
    "give_up_after_hours": 48,      # dead pool: stop retrying, mark complete

    # --- Feed / UI ---
    "feed_limit": 100,
}
