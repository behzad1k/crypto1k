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
    # Raised 15k → 150k on 2026-07-26 outcome evidence. Across 598 bullish
    # alerts with complete 24h outcomes, sub-150k-liquidity alerts averaged
    # -13.8% at 24h with a 19.8% win rate, vs -4.0% / 41.0% above the line.
    # The effect was monotone across every cutoff tested and held in both
    # halves of a temporal train/test split. This is the single highest-impact
    # filter in the system.
    "min_liquidity_usd": 150_000,
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
    # Lowered 75 → 8: no alert in a month of data ever exceeded turnover 20, so
    # the old value was dead code. Turnover 5–20 averaged -13.4% at 24h vs
    # -4.5% below 5, so 8 is where the damage actually starts.
    "wash_turnover_ratio": 8.0,  # vol_24h / liquidity
    # Exit-door sanity: liquidity as a % of market cap. Below this, the pool is
    # too shallow relative to the token's valuation — you (or any holder) can't
    # exit without cratering the price. Warning + score cap, not a hard gate.
    "min_liq_to_mcap_pct": 3.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. OVER-EXTENSION  —  refuse to buy a move that has already happened
# ═══════════════════════════════════════════════════════════════════════════════
#
# The clearest, most monotone finding in the 2026-07-26 outcome review: the
# further price had ALREADY run when the alert fired, the worse the alert did.
# 24h mean return by 1h price change at alert time:
#     0–3% → -1.4% | 3–8% → -6.3% | 8–15% → -6.8% | 15–30% → -16.4% | 30%+ → -28.5%
# The same gradient shows up on the 5m and 6h windows, and the 8%+ buckets held
# up as losers in both halves of a train/test split. These are hard gates: past
# them we are buying someone else's exit.
#
EXTENSION = {
    "enabled": True,
    "max_price_change_1h_pct": 15.0,   # 15–30% bucket: -16.4% mean, 24% win
    "max_price_change_5m_pct": 6.0,    # 6%+ bucket: -10.8% mean, 29% win
    "max_price_change_6h_pct": 50.0,   # 50%+ bucket: -38.2% mean, 0% win (n=10)
    # Softer band that only costs score instead of blocking the alert.
    "penalty_price_change_1h_pct": 8.0,
    "penalty_points": 1.0,
    "penalty_price_change_6h_pct": 30.0,
    "penalty_points_6h": 1.5,
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
    # Outcome evidence (2026-07-26): vol_pace_1h is the ONLY factor with a
    # monotone, train/test-stable relationship to forward return. 1h return by
    # bucket (train | test):
    #   <2.5×  -0.5% / 45% win | -1.5% / 43% win
    #   2.5–5× +0.3% / 52%     | +0.6% / 56%
    #   5–10×  +7.8% / 62%     | +1.2% / 53%
    #   10×+  +11.9% / 60%     | +10.9% / 71%
    # Take-profit feasibility follows it too: at ≥5× pace, 44% of alerts touch
    # +5% within the hour vs 25% below 2.5×. This is the edge — weighted top.
    "vol_pace_1h_notable": 2.5,  # 1h volume this many × the 24h hourly average
    "vol_pace_1h_strong": 5.0,
    "vol_pace_1h_extreme": 10.0,  # the 60–71% win-rate bucket
    # 5m acceleration: only the extreme band separates. 6–12× was flat in both
    # halves; 12×+ ran +2.9% / +2.1% (52% / 61% win), so "strong" moved 6 → 12.
    "vol_pace_5m_notable": 6.0,
    "vol_pace_5m_strong": 12.0,
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
    # ── Re-weighted 2026-07-26 from 810 alerts with complete 24h outcomes ──
    # The old weights (vol 4 / mom 3 / buy 2 / act 1 / smart 1.5) produced a
    # score that was *inversely* useful at the top: STRONG (8+) alerts averaged
    # -6.0% at 24h with a 33.6% win rate while GOOD (6–8) managed -4.3% / 41.5%.
    # A high score mostly meant "every momentum signal fired at once", i.e. the
    # pump was already mature. Momentum is now the smallest component and
    # volume/liquidity carry the score.
    "max_volume_surge": 5.0,  # the one factor that survived out-of-sample
    "max_momentum": 1.5,      # was 3.0 — most reliably harmful signal family
    "max_buy_pressure": 1.0,  # was 2.0 — buy_ratio bands flipped sign in test
    "max_activity": 2.5,      # was 1.0 — liquidity depth is the #2 real factor
    # Smart-money bonus (added on top by the enrichment pass, total still
    # clamped at 10). Cut 1.5 → 0.5: tracked-wallet buys preceded a -10.4%
    # mean 24h return (33% win), and across 91,804 scored wallet buys the
    # tracker's own edge is +1.4% at 1h decaying to -2.4% at 24h. It is a
    # weak confirmation input, not a thesis.
    "max_smart_money": 0.5,
    # Score → label bands, recalibrated for the new weight distribution
    # (base components now sum to 10.0 with a 0.5 smart-money bonus on top).
    # Under the new model the observed 1h win rate by band is
    # <4 → 24%, 4–6 → 53%, 6–7 → 50%, 7–8 → 61%, so the junk floor moved to 4.
    "label_strong": 7.0,
    "label_good": 5.5,
    "label_fair": 4.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ALERTING  —  when an email actually goes out  (kept LENIENT on purpose)
# ═══════════════════════════════════════════════════════════════════════════════

ALERTING = {
    # Re-tuned for the new scoring scale (see SCORING). Simulated over the
    # historical alert set, the full proposed pipeline at this threshold keeps
    # ~33% of old alerts and turns the 1h return from +0.34% / 48.0% win into
    # +0.41% / 56.9% win, with the 24h mean improving from -5.5% to -0.8%.
    "min_validity_score": 5.5,
    # Volume gate: a coin qualifies if it shows a 1h surge OR a 5m acceleration.
    # The 5m path catches setups heating up *right now* even when the full hour
    # still looks average (e.g. PENGU: 1h 1.1× but 5m 6.4×).
    "min_vol_pace_1h": 2.5,  # 1h volume vs its 24h hourly average
    "min_vol_pace_5m": 6.0,  # 5m volume vs its 24h 5-min average (acceleration)
    "require_bullish": True,  # True = only alert on bullish-leaning spikes
    # Don't re-alert the same coin again until this many minutes have passed.
    # Raised 120 → 360. Repeat alerts on the same symbol degrade badly: DIH
    # fired 60 times and averaged -19.9% (22% win), CASHCAT 131 times at -7.9%.
    # Applying a 6h per-symbol cooldown to the filtered set lifted the 24h mean
    # from -0.93% to -0.63% and cost only ~19% of alerts.
    "cooldown_minutes": 360,
    # …unless the new score beats the best score alerted inside that window by
    # this much — a weak early alert must not mask the real pump an hour later.
    # Raised 2.0 → 2.5 to match the compressed new score scale.
    "cooldown_rearm_score_jump": 2.5,
    # Hard ceiling on how often one symbol may alert per rolling 24h, no matter
    # how much its score jumps. Nothing in the data justified the 20th alert on
    # a symbol; this stops a single manipulated coin owning the feed.
    "max_alerts_per_symbol_24h": 3,
    # FAST PATH — alert immediately on an extreme 5-minute spike, regardless of
    # the validity score. The score is dominated by 1h rolling windows, which
    # dilute a fresh move; this path uses only 5m data (volume pace, price
    # move, buy/sell flow) so a spike can alert on the first scan after it
    # starts. Wash-trading suspects (see wash_turnover_ratio) never qualify.
    #
    # KEEP AS IS — this was the best-performing path in the outcome review and
    # the only one positive in both halves of the split (1h +1.04% / 50% win
    # train, +0.74% / 54% win test; 24h -0.14%, the least-negative of any path).
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
# 6b. EXIT PLAN  —  an alert is a short trade, not a position
# ═══════════════════════════════════════════════════════════════════════════════
#
# The headline finding of the 2026-07-26 review. Bullish alerts are +0.55% at
# 15m and +0.34% at 1h, then decay to -1.3% at 4h and -5.5% at 24h (37.6% win).
# But 59% of them touch +5% at some point within 24h and 39% touch +10%, with a
# median time-to-peak of 5.6 hours. The alerts find real pumps; holding them
# gives the pump back. These numbers are attached to every alert so the exit is
# decided at entry, not improvised.
#
EXIT_PLAN = {
    "enabled": True,
    "take_profit_pct": 10.0,
    "stop_loss_pct": 10.0,
    "max_hold_hours": 8,
    # Measured by replaying the CURRENT filter/scoring stack over the month of
    # historical alerts (167 survive of 598), not over the old unfiltered feed
    # — quoting the old numbers here would overstate the target's hit rate.
    #
    #   reached +5%  within 24h: 44.3%      drew down -5%  within 24h: 46.7%
    #   reached +10% within 24h: 23.4%      drew down -10% within 24h: 21.0%
    #   reached +20% within 24h:  8.4%      drew down -20% within 24h:  4.2%
    #   median time to peak: 7.5h (51% of peaks land within 8h)
    #   ran +10% then closed the day red: 5.4% (was 17.3% pre-filter)
    #
    # CAVEAT, and it matters: these are unordered touch rates. The stored
    # outcome columns only kept the 24h high and low, so there is no way to
    # tell whether the target filled before the stop on the same alert. The
    # tp_hit_at / sl_hit_at columns added alongside this config now record the
    # first crossing of each level, so the next review can answer it properly
    # and this take_profit/stop_loss pair can be set on a real backtest rather
    # than on touch rates. Treat the levels below as a sane starting point.
    "observed": {
        "hit_tp_within_24h_pct": 23.4,
        "hit_sl_within_24h_pct": 21.0,
        "reached_5pct_within_24h_pct": 44.3,
        "median_hours_to_peak": 7.5,
        "basis": "replay of the current stack over 2026-06-24..07-25 (n=167)",
        "ordering_known": False,
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
    # ENTRY GATE — STAYS OFF. Checked properly on 2026-07-26 and there is no
    # stable cutoff to set, in either direction.
    #
    # It looks at first like there is: alerts whose 24h window contained a BTC
    # drop of 1%+ won only 21.2% of the time vs 49.4% when BTC rose. But that
    # is BTC's move measured *during* the outcome window — information from the
    # future of the alert. Gating on it is not possible.
    #
    # What we can actually see at fire time — regime score, BTC 1h change, BTC
    # 24h change — does not survive a train/test split. Splitting the month in
    # half, the worst regime bucket flips: in the first half regime <5.0 was
    # worst (-10.0%, 19% win) while in the second half regime ≥5.5 was worst
    # (-18.5%, 14% win). A gate fitted to either half would have hurt on the
    # other. Keep stamping BTC context on alerts for the digest, but do not
    # act on it until a cutoff holds up out-of-sample.
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

    # --- Score bonus split (capped at SCORING['max_smart_money'] = 0.5) ---
    # All four cut hard on 2026-07-26. Smart-money signals were the WORST
    # performing family in the review: alerts carrying smart_wallet_buy
    # averaged -10.4% at 24h (33% win) vs -2.9% without it, and the worst
    # signal pairs in the entire dataset were all momentum × smart_wallet_buy
    # combinations (-11% to -13%). whale_print and whale_net_flow were roughly
    # neutral once the two dominant symbols were excluded, so they keep a
    # token weight; the tracked-wallet buy no longer outweighs everything else.
    "pts_net_flow_notable": 0.1,
    "pts_net_flow_strong": 0.25,
    "pts_big_trade": 0.1,
    "pts_smart_wallet_buy": 0.25,  # was 1.5

    # --- Alerting: a smart-wallet buy can fire an alert on its own ---
    # DISABLED 2026-07-26. This path fired 232 alerts and was the worst in the
    # system by a wide margin: -10.4% mean at 24h with a 32.4% win rate, vs
    # -0.1% for the fast path. It was negative in both halves of a train/test
    # split, and negative even after excluding the two symbols (DIH, CASHCAT)
    # that dominated it. Raising its score bar does not rescue it — smart-path
    # alerts scoring ≥6.5 still averaged -8.7%. Smart money is now a
    # confirmation input only: it can nudge a score, never open an alert.
    "alert_on_smart_wallet_buy": False,

    # --- Auto-qualification: when does a wallet become "smart money"? ---
    #
    # TURNED OFF 2026-07-26 — not tightened, turned off. Auto-qualification was
    # tested properly for the first time: qualify wallets using only trades
    # before a cutoff date, then measure how their buys performed AFTER it.
    # It is reliably ANTI-predictive. Qualified wallets underperformed the
    # all-wallet baseline at 1h at every one of six cutoffs tested:
    #
    #   cutoff       qualified 1h / win      all wallets 1h / win
    #   2026-07-11   +0.23% / 51%            +1.04% / 51%
    #   2026-07-13   +0.25% / 51%            +1.22% / 51%
    #   2026-07-15   +0.10% / 48%            +2.05% / 51%
    #   2026-07-17   +3.18% / 57%            +3.22% / 53%
    #   2026-07-19   +0.37% / 53%            +3.56% / 53%
    #   2026-07-21   +0.21% / 52%            +5.00% / 53%
    #
    # At 24h they were worse in five of six (e.g. -12.3% vs -5.4%). Tightening
    # the bar does not fix it — 20 buys / 60% win / positive average 24h return
    # / multiple pools still forward-tested at -0.29% 1h and -10.3% 24h, worse
    # than picking wallets at random. Past win rate on meme-coin buys is
    # measuring luck and regime, not skill, and selecting on it concentrates
    # whichever wallets were most exposed to the coins that had already run.
    #
    # Whale trades are still recorded and the leaderboard still renders — the
    # data is worth having and the page is worth reading. What no longer
    # happens is a wallet being silently promoted to "smart money" on this
    # basis and then feeding score or alerts. Manually tracked wallets
    # (smart_wallets table, source='manual') are unaffected and still count.
    "auto_qualify": False,
    "qualify": {
        # Retained for the leaderboard's win-rate column and for the manual
        # tracking UI. With auto_qualify False these no longer promote anyone.
        "min_scored_buys": 20,
        "min_win_rate": 0.60,
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
