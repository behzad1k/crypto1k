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
    "enabled": True,  # flip to True once filled in
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
    "enabled": False,  # flip to True once token + chat_id are filled in
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
    # Wash-trade sanity: volume/liquidity above this is suspicious but NOT
    # auto-rejected — it only caps the score and adds a warning to the email.
    "wash_turnover_ratio": 75.0,  # vol_24h / liquidity
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
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SCORING  —  weights for the 0–10 validity score (mirrors the setup score)
# ═══════════════════════════════════════════════════════════════════════════════

SCORING = {
    "max_volume_surge": 4.0,  # the headline factor
    "max_momentum": 3.0,  # price move + multi-window alignment
    "max_buy_pressure": 2.0,
    "max_activity": 1.0,  # liquidity + trade-count health
    # Score → label bands (out of 10)
    "label_strong": 8.0,
    "label_good": 6.0,
    "label_fair": 4.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ALERTING  —  when an email actually goes out  (kept LENIENT on purpose)
# ═══════════════════════════════════════════════════════════════════════════════

ALERTING = {
    "min_validity_score": 6.2,  # 0–10; lenient. Raise to be pickier.
    "min_vol_pace_1h": 2.5,  # must at least show this volume surge
    "require_bullish": False,  # True = only alert on bullish-leaning spikes
    # Don't re-alert the same coin again until this many minutes have passed.
    "cooldown_minutes": 120,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MONITOR  —  background scanning loop
# ═══════════════════════════════════════════════════════════════════════════════

MONITOR = {
    "enabled_on_start": True,  # auto-start the background loop when the app boots
    "interval_seconds": 180,  # how often to re-scan the watchlist (3 min)
    "per_request_pause": 0.4,  # polite pause between DexScreener calls
}
