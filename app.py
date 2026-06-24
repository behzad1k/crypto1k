"""
Crypto Signal Analyzer v2
Minimal Flask app — one symbol page with family-based signal analysis.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from functools import wraps

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

import db
import emailer
import scanner
import telegram_notify
from scalp_signal_analyzer import ScalpSignalAnalyzer
from scanner_config import MONITOR
from signal_config import (
    CATEGORY_ORDER,
    GRADE_WEIGHTS,
    HORIZONS,
    SIGNAL_FAMILIES,
    SIGNAL_LOOKUP,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "sogol"


def _sanitize(obj):
    """Recursively convert numpy scalars to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


analyzer = ScalpSignalAnalyzer()
db.init_db()


# ── Auth ──────────────────────────────────────────────────────────────────────


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "logged_in" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        if (
            data.get("username") == "iheartsogol"
            and data.get("password") == "sogolpleasecomeback:((((("
        ):
            session["logged_in"] = True
            return jsonify({"success": True})
        return jsonify({"success": False, "error": "Invalid credentials"}), 401
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))


# ── Pages ─────────────────────────────────────────────────────────────────────


@app.route("/")
@login_required
def index():
    return redirect(url_for("symbol_page", symbol="BTC"))


@app.route("/symbol/<symbol>")
@login_required
def symbol_page(symbol):
    return render_template("symbol.html", symbol=symbol)


@app.route("/scanner")
@login_required
def scanner_page():
    return render_template("scanner.html")


# ── Scanner API ───────────────────────────────────────────────────────────────


@app.route("/api/scanner/watchlist", methods=["GET", "POST"])
@login_required
def scanner_watchlist():
    if request.method == "POST":
        data = request.get_json() or {}
        raw = data.get("symbols", "")
        if isinstance(raw, str):
            # Accept comma / space / newline separated input
            symbols = [s for s in raw.replace(",", " ").split() if s]
        else:
            symbols = [str(s).strip() for s in raw if str(s).strip()]
        added = db.add_to_watchlist(symbols)
        return jsonify(
            {"success": True, "added": added, "watchlist": db.get_watchlist()}
        )
    return jsonify({"success": True, "watchlist": db.get_watchlist()})


@app.route("/api/scanner/watchlist/<symbol>", methods=["DELETE"])
@login_required
def scanner_watchlist_remove(symbol):
    db.remove_from_watchlist(symbol)
    return jsonify({"success": True, "watchlist": db.get_watchlist()})


@app.route("/api/scanner/scan", methods=["POST"])
@login_required
def scanner_scan():
    """Scan now. Optional body {symbols:[...]} overrides the saved watchlist."""
    data = request.get_json() or {}
    symbols = data.get("symbols")
    if not symbols:
        symbols = db.get_watchlist()
    if not symbols:
        return jsonify(
            {"success": True, "results": [], "message": "Watchlist is empty"}
        )
    results = scanner.scan_symbols(symbols)
    return jsonify(
        {"success": True, "results": _sanitize(results), "count": len(results)}
    )


@app.route("/api/scanner/scan-and-alert", methods=["POST"])
@login_required
def scanner_scan_and_alert():
    """Scan the watchlist and actually fire emails for fresh alerts."""
    summary = scanner.scan_and_alert()
    return jsonify({"success": True, **_sanitize(summary)})


@app.route("/api/scanner/live")
@login_required
def scanner_live():
    """Latest scan results (from the background monitor or a manual scan)."""
    return jsonify({"success": True, **_sanitize(scanner.last_scan())})


@app.route("/api/scanner/alerts")
@login_required
def scanner_alerts():
    limit = int(request.args.get("limit", 50))
    return jsonify({"success": True, "alerts": db.get_recent_alerts(limit)})


@app.route("/api/scanner/monitor", methods=["GET", "POST"])
@login_required
def scanner_monitor():
    if request.method == "POST":
        action = (request.get_json() or {}).get("action")
        if action == "start":
            scanner.start_monitor()
        elif action == "stop":
            scanner.stop_monitor()
    return jsonify({"success": True, **scanner.monitor_status()})


@app.route("/api/scanner/test-email", methods=["POST"])
@login_required
def scanner_test_email():
    ok, message = emailer.send_test()
    return jsonify(
        {"success": ok, "message": message, "recipients": emailer.recipient_count()}
    )


@app.route("/api/scanner/test-telegram", methods=["POST"])
@login_required
def scanner_test_telegram():
    ok, message = telegram_notify.send_test()
    return jsonify({"success": ok, "message": message})


# ── Scalp metrics ─────────────────────────────────────────────────────────────


def compute_scalp_metrics(
    df, price, all_signals, chart_levels, timeframes, bias_direction
):
    """
    Compute all scalp-trading-specific metrics:
      - Volume ratio vs rolling average
      - ATR context (how many ATRs to reach 2% target)
      - Multi-timeframe alignment score
      - Room to run (nearest level above/below in %)
      - Key levels list sorted by proximity
      - Suggested stop levels
      - Setup quality score (0–10)
    """
    m = {
        "volume_ratio": None,
        "volume_trend": None,
        "atr_value": None,
        "atr_pct": None,
        "atrs_for_2pct": None,
        "target_realistic": None,
        "tf_alignment": None,
        "room_above": None,
        "room_below": None,
        "key_levels": [],
        "stop_above": None,
        "stop_below": None,
        "setup_score": None,
        "setup_max": 10,
        "setup_label": None,
        "setup_factors": [],
    }

    # ── Volume ratio & trend ──────────────────────────────────────────────────
    if df is not None and len(df) > 21:
        try:
            current_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].iloc[-21:-1].mean())
            if avg_vol > 0:
                m["volume_ratio"] = round(current_vol / avg_vol, 2)

            # Trend: last 3 candles vs prev 3
            last3 = float(df["volume"].iloc[-3:].mean())
            prev3 = float(df["volume"].iloc[-6:-3].mean())
            if prev3 > 0:
                ratio = last3 / prev3
                m["volume_trend"] = (
                    "rising" if ratio > 1.1 else ("falling" if ratio < 0.9 else "flat")
                )
        except Exception as e:
            logger.warning(f"Volume calc failed: {e}")

    # ── ATR (14-period) ───────────────────────────────────────────────────────
    if df is not None and len(df) > 15:
        try:
            hi, lo, cl = df["high"], df["low"], df["close"]
            tr = pd.concat(
                [(hi - lo), (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1
            ).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            atr_pct = (atr / price) * 100
            m["atr_value"] = round(atr, 4)
            m["atr_pct"] = round(atr_pct, 3)
            atrs = round(2.0 / atr_pct, 1) if atr_pct > 0 else None
            m["atrs_for_2pct"] = atrs
            m["target_realistic"] = bool(atrs is not None and atrs <= 6)
        except Exception as e:
            logger.warning(f"ATR calc failed: {e}")

    # ── Multi-timeframe alignment ─────────────────────────────────────────────
    tf_directions = {}
    for tf in timeframes:
        tf_signals = [s for s in all_signals if s["timeframe"] == tf]
        bull = sum(
            GRADE_WEIGHTS.get(s["grade"], 0.5)
            for s in tf_signals
            if s["direction"] == "bullish"
        )
        bear = sum(
            GRADE_WEIGHTS.get(s["grade"], 0.5)
            for s in tf_signals
            if s["direction"] == "bearish"
        )
        if bull > bear * 1.15:
            tf_directions[tf] = "bullish"
        elif bear > bull * 1.15:
            tf_directions[tf] = "bearish"
        else:
            tf_directions[tf] = "neutral"

    aligned_dir = bias_direction if bias_direction != "neutral" else "bullish"
    aligned_count = sum(1 for d in tf_directions.values() if d == aligned_dir)
    total = len(timeframes)
    strength = (
        "full"
        if aligned_count == total
        else ("partial" if aligned_count >= max(1, total * 0.6) else "split")
    )

    m["tf_alignment"] = {
        "timeframes": tf_directions,
        "aligned_direction": aligned_dir,
        "aligned_count": aligned_count,
        "total_count": total,
        "strength": strength,
    }

    # ── Room to run & key levels ──────────────────────────────────────────────
    # Deduplicate levels within 0.3% of each other
    raw_levels = [
        l
        for l in chart_levels
        if l.get("price") and 0 < abs(l["price"] - price) / price < 0.12
    ]
    raw_levels.sort(key=lambda x: x["price"])

    deduped, last_p = [], None
    for l in raw_levels:
        if last_p is None or abs(l["price"] - last_p) / price > 0.003:
            deduped.append(l)
            last_p = l["price"]

    above = [l for l in deduped if l["price"] > price * 1.001]
    below = [l for l in deduped if l["price"] < price * 0.999]
    above.sort(key=lambda x: x["price"])
    below.sort(key=lambda x: x["price"], reverse=True)

    # Key levels list — nearest 10, tagged with distance
    key_levels = []
    for l in above[:5]:
        pct = round(((l["price"] - price) / price) * 100, 2)
        key_levels.append(
            {
                "price": l["price"],
                "label": l["label"],
                "pct": pct,
                "side": "above",
                "grade": l["grade"],
                "direction": l["direction"],
            }
        )
    for l in below[:5]:
        pct = round(((price - l["price"]) / price) * 100, 2)
        key_levels.append(
            {
                "price": l["price"],
                "label": l["label"],
                "pct": -pct,
                "side": "below",
                "grade": l["grade"],
                "direction": l["direction"],
            }
        )
    key_levels.sort(key=lambda x: abs(x["pct"]))
    m["key_levels"] = key_levels[:10]

    if above:
        p = above[0]["price"]
        pct = round(((p - price) / price) * 100, 2)
        m["room_above"] = {
            "price": float(p),
            "label": above[0]["label"],
            "pct": float(pct),
            "enough": bool(pct >= 2.0),
        }
        m["stop_above"] = float(p)

    if below:
        p = below[0]["price"]
        pct = round(((price - p) / price) * 100, 2)
        m["room_below"] = {
            "price": float(p),
            "label": below[0]["label"],
            "pct": float(pct),
            "enough": bool(pct >= 2.0),
        }
        m["stop_below"] = float(p)

    # ── Setup quality score ───────────────────────────────────────────────────
    score = 0.0
    factors = []

    # TF alignment (max 3)
    if strength == "full":
        pts = 3
        detail = f"All {total}/{total} timeframes aligned {aligned_dir}"
    elif strength == "partial":
        pts = 1.5
        detail = f"{aligned_count}/{total} timeframes aligned {aligned_dir}"
    else:
        pts = 0
        detail = "Timeframes split — no clear consensus"
    score += pts
    factors.append({"name": "TF Alignment", "points": pts, "max": 3, "detail": detail})

    # Volume (max 2)
    vr = m["volume_ratio"]
    if vr and vr >= 2.5:
        pts = 2
        detail = f"{vr}× average — strong conviction"
    elif vr and vr >= 1.4:
        pts = 1
        detail = f"{vr}× average — moderate"
    else:
        pts = 0
        detail = f"{vr}× average — weak/no confirmation" if vr else "No volume data"
    score += pts
    factors.append({"name": "Volume", "points": pts, "max": 2, "detail": detail})

    # ADX trend strength (max 2)
    adx_val = next(
        (
            s["indicator_value"]
            for s in all_signals
            if s["signal_name"] == "adx_strong_trend" and s["indicator_value"]
        ),
        None,
    )
    if adx_val and adx_val >= 40:
        pts = 2
        detail = f"ADX {adx_val:.1f} — very strong trend"
    elif adx_val and adx_val >= 28:
        pts = 1
        detail = f"ADX {adx_val:.1f} — trend has strength"
    else:
        pts = 0
        detail = (
            f"ADX {adx_val:.1f} — weak trend, choppy" if adx_val else "ADX unavailable"
        )
    score += pts
    factors.append(
        {"name": "Trend Strength", "points": pts, "max": 2, "detail": detail}
    )

    # Room to run in trade direction (max 2)
    trade_room = m["room_below"] if aligned_dir == "bearish" else m["room_above"]
    if trade_room:
        r = trade_room["pct"]
        if r >= 2.5:
            pts = 2
            detail = f"{r:.1f}% clear in trade direction"
        elif r >= 1.5:
            pts = 1
            detail = f"{r:.1f}% — tight but workable"
        else:
            pts = 0
            detail = f"Only {r:.1f}% — level too close, blocked"
    else:
        pts = 1
        detail = "No nearby levels — open road"
    score += pts
    factors.append({"name": "Room to Run", "points": pts, "max": 2, "detail": detail})

    # ATR target realism (max 1)
    atrs = m["atrs_for_2pct"]
    if atrs is not None:
        if atrs <= 4:
            pts = 1
            detail = f"2% = {atrs}× ATR — highly realistic"
        elif atrs <= 6:
            pts = 1
            detail = f"2% = {atrs}× ATR — realistic"
        elif atrs <= 9:
            pts = 0.5
            detail = f"2% = {atrs}× ATR — stretched"
        else:
            pts = 0
            detail = f"2% = {atrs}× ATR — unlikely in one run"
    else:
        pts = 0
        detail = "ATR unavailable"
    score += pts
    factors.append({"name": "ATR Target", "points": pts, "max": 1, "detail": detail})

    m["setup_score"] = round(score, 1)
    m["setup_label"] = (
        "STRONG"
        if score >= 8
        else "GOOD"
        if score >= 6
        else "FAIR"
        if score >= 4
        else "WEAK"
    )
    m["setup_factors"] = factors
    return m


# ── Analysis API ──────────────────────────────────────────────────────────────


@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze():
    data = request.get_json() or {}
    symbol = data.get("symbol", "").strip().upper()
    horizon = data.get("horizon", "mid").lower()

    if not symbol:
        return jsonify({"success": False, "error": "symbol is required"}), 400
    if horizon not in HORIZONS:
        return jsonify(
            {
                "success": False,
                "error": f"horizon must be one of: {list(HORIZONS.keys())}",
            }
        ), 400

    timeframes = HORIZONS[horizon]["timeframes"]
    chart_tf = HORIZONS[horizon]["chart_tf"]

    try:
        raw = analyzer.analyze_symbol_all_timeframes(symbol, timeframes)
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    # Fetch OHLCV for scalp metrics (volume ratio, ATR)
    ohlcv_df = None
    try:
        ohlcv_df = analyzer.fetch_kucoin_data(symbol, chart_tf)
        if ohlcv_df is None:
            ohlcv_df = analyzer.fetch_binance_data(symbol, chart_tf)
        if ohlcv_df is None:
            ohlcv_df = analyzer.fetch_dexscreener_data(symbol, chart_tf)
    except Exception as e:
        logger.warning(f"OHLCV fetch for scalp metrics failed: {e}")

    result = _build_result(symbol, horizon, timeframes, raw, ohlcv_df)

    try:
        db.save_analysis(symbol, horizon, timeframes, result)
    except Exception as e:
        logger.warning(f"Failed to save analysis to DB: {e}")

    return jsonify({"success": True, **_sanitize(result)})


def _build_result(symbol, horizon, timeframes, raw, ohlcv_df=None):
    now = datetime.now(timezone.utc).isoformat()
    tf_seconds = HORIZONS[horizon]["tf_seconds"]

    # Price
    price = 0.0
    for tf_data in raw.get("timeframes", {}).values():
        if "error" not in tf_data and tf_data.get("price"):
            price = tf_data["price"]
            break

    # ── Flatten signals ───────────────────────────────────────────────────────
    all_signals = []
    for tf, tf_data in raw.get("timeframes", {}).items():
        if "error" in tf_data:
            continue
        detected_at = tf_data.get("timestamp", now)

        try:
            detected_ts = datetime.fromisoformat(
                str(detected_at).replace("Z", "+00:00")
            )
            elapsed_sec = (datetime.now(timezone.utc) - detected_ts).total_seconds()
            candles_ago = max(0, int(elapsed_sec / tf_seconds.get(tf, 3600)))
        except Exception:
            candles_ago = None

        for signal_name, signal_data in tf_data.get("signals", {}).items():
            meta = SIGNAL_LOOKUP.get(signal_name)
            if not meta:
                continue

            raw_value = signal_data.get("value")
            price_level = signal_data.get("level")
            if (
                meta["value_type"] == "price_level"
                and price_level is None
                and raw_value is not None
            ):
                price_level = raw_value
                raw_value = None

            all_signals.append(
                {
                    "signal_name": signal_name,
                    "family": meta["family"],
                    "family_name": meta["family_name"],
                    "category": meta["category"],
                    "role": meta["role"],
                    "grade": meta["grade"],
                    "direction": meta["direction"],
                    "value_type": meta["value_type"],
                    "value_label": meta["value_label"],
                    "timeframe": tf,
                    "indicator_value": raw_value,
                    "price_level": price_level,
                    "candles_ago": candles_ago,
                    "detected_at": detected_at,
                }
            )

    # ── Bias ──────────────────────────────────────────────────────────────────
    bull_score = bear_score = 0.0
    for s in all_signals:
        w = GRADE_WEIGHTS.get(s["grade"], 0.5)
        if s["direction"] == "bullish":
            bull_score += w
        elif s["direction"] == "bearish":
            bear_score += w

    total_weight = bull_score + bear_score
    if total_weight == 0:
        bias_score, bias_direction = 0.0, "neutral"
    else:
        bias_score = round((bull_score - bear_score) / total_weight, 3)
        bias_direction = (
            "bullish"
            if bias_score > 0.15
            else "bearish"
            if bias_score < -0.15
            else "neutral"
        )

    # ── Per-family summary ────────────────────────────────────────────────────
    families_out = {}
    for family_key, family_def in SIGNAL_FAMILIES.items():
        family_signals = [s for s in all_signals if s["family"] == family_key]
        if not family_signals:
            continue

        f_bull = sum(
            GRADE_WEIGHTS.get(s["grade"], 0.5)
            for s in family_signals
            if s["direction"] == "bullish"
        )
        f_bear = sum(
            GRADE_WEIGHTS.get(s["grade"], 0.5)
            for s in family_signals
            if s["direction"] == "bearish"
        )
        f_bias = (
            "bullish"
            if f_bull > f_bear
            else "bearish"
            if f_bear > f_bull
            else "neutral"
        )

        indicator_values, price_levels = {}, {}
        for s in family_signals:
            if s["indicator_value"] is not None and s["value_label"]:
                indicator_values[s["value_label"]] = s["indicator_value"]
            if s["price_level"] is not None and s["value_label"]:
                price_levels[f"{s['value_label']} ({s['timeframe']})"] = s[
                    "price_level"
                ]

        families_out[family_key] = {
            "name": family_def["name"],
            "description": family_def["description"],
            "category": family_def["category"],
            "bias": f_bias,
            "active_signals": family_signals,
            "indicator_values": indicator_values,
            "price_levels": price_levels,
        }

    # Chart price level lines
    chart_levels = [
        {
            "price": s["price_level"],
            "label": f"{s['value_label']} ({s['timeframe']})",
            "direction": s["direction"],
            "grade": s["grade"],
        }
        for s in all_signals
        if s["price_level"] is not None and s["value_label"]
    ]

    # ── Scalp metrics ─────────────────────────────────────────────────────────
    scalp_metrics = compute_scalp_metrics(
        ohlcv_df, price, all_signals, chart_levels, timeframes, bias_direction
    )

    return {
        "symbol": symbol,
        "horizon": horizon,
        "horizon_label": HORIZONS[horizon]["label"],
        "chart_tf": HORIZONS[horizon]["chart_tf"],
        "timeframes": timeframes,
        "price": price,
        "timestamp": now,
        "bias": {
            "direction": bias_direction,
            "score": bias_score,
            "bull_score": round(bull_score, 1),
            "bear_score": round(bear_score, 1),
        },
        "families": families_out,
        "primary_signals": [s for s in all_signals if s["grade"] == "A"],
        "all_signals": all_signals,
        "chart_levels": chart_levels,
        "scalp_metrics": scalp_metrics,
        "category_order": CATEGORY_ORDER,
        "source": raw.get("source"),
        "degraded": raw.get("degraded", False),
        "snapshot": raw.get("snapshot"),
    }


# ── OHLCV for chart ───────────────────────────────────────────────────────────


@app.route("/api/ohlcv/<symbol>/<timeframe>")
@login_required
def get_ohlcv(symbol, timeframe):
    try:
        df = analyzer.fetch_kucoin_data(symbol, timeframe)
        if df is None:
            df = analyzer.fetch_binance_data(symbol, timeframe)
        if df is None:
            df = analyzer.fetch_dexscreener_data(symbol, timeframe)
        if df is None:
            return jsonify({"success": False, "error": "No data available"}), 404

        candles = [
            {
                "time": int(r["timestamp"].timestamp()),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            }
            for _, r in df.tail(150).iterrows()
        ]
        return jsonify(
            {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Market data ───────────────────────────────────────────────────────────────


@app.route("/api/market-data/<symbol>")
@login_required
def get_market_data(symbol):
    result = {
        "funding_rate": None,
        "next_funding_in": None,
        "open_interest": None,
        "oi_change_pct": None,
        "bid_depth": None,
        "ask_depth": None,
        "imbalance_ratio": None,
        "imbalance_side": None,
    }
    binance_symbol = symbol.replace("-", "")

    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": binance_symbol},
            timeout=5,
        )
        if r.status_code == 200:
            d = r.json()
            result["funding_rate"] = float(d.get("lastFundingRate", 0))
            nft = d.get("nextFundingTime", 0)
            if nft:
                diff = int(
                    (
                        datetime.fromtimestamp(nft / 1000, tz=timezone.utc)
                        - datetime.now(timezone.utc)
                    ).total_seconds()
                    / 60
                )
                result["next_funding_in"] = max(0, diff)
    except Exception as e:
        logger.warning(f"Funding rate failed: {e}")

    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": binance_symbol, "period": "1h", "limit": 2},
            timeout=5,
        )
        if r.status_code == 200:
            d = r.json()
            if len(d) >= 2:
                oi_now = float(d[-1].get("sumOpenInterest", 0))
                oi_prev = float(d[-2].get("sumOpenInterest", 0))
                result["open_interest"] = oi_now
                if oi_prev > 0:
                    result["oi_change_pct"] = round(
                        ((oi_now - oi_prev) / oi_prev) * 100, 3
                    )
    except Exception as e:
        logger.warning(f"OI failed: {e}")

    try:
        r = requests.get(
            "https://api.kucoin.com/api/v1/market/orderbook/level2_20",
            params={"symbol": symbol},
            timeout=5,
        )
        if r.status_code == 200:
            d = r.json().get("data", {})
            bids = d.get("bids", [])
            asks = d.get("asks", [])
            bid_depth = sum(float(b[0]) * float(b[1]) for b in bids if len(b) >= 2)
            ask_depth = sum(float(a[0]) * float(a[1]) for a in asks if len(a) >= 2)
            result["bid_depth"] = round(bid_depth, 0)
            result["ask_depth"] = round(ask_depth, 0)
            if ask_depth > 0 and bid_depth > 0:
                ratio = bid_depth / ask_depth
                result["imbalance_ratio"] = round(ratio, 2)
                result["imbalance_side"] = (
                    "bullish"
                    if ratio > 1.1
                    else "bearish"
                    if ratio < 0.9
                    else "neutral"
                )
    except Exception as e:
        logger.warning(f"Order book failed: {e}")

    return jsonify({"success": True, "symbol": symbol, **result})


# ── Signal history ────────────────────────────────────────────────────────────


@app.route("/api/signals/<symbol>")
@login_required
def get_signals(symbol):
    horizon = request.args.get("horizon")
    limit = int(request.args.get("limit", 100))
    signals = db.get_recent_signals(symbol, horizon=horizon, limit=limit)
    return jsonify(
        {"success": True, "symbol": symbol, "signals": signals, "count": len(signals)}
    )


@app.route("/api/runs/<symbol>")
@login_required
def get_runs(symbol):
    horizon = request.args.get("horizon")
    runs = db.get_recent_runs(
        symbol, horizon=horizon, limit=int(request.args.get("limit", 20))
    )
    return jsonify({"success": True, "symbol": symbol, "runs": runs})


@app.route("/api/liquidations/<symbol>")
@login_required
def get_liquidations(symbol):
    """Estimate liquidation clusters from OI and leverage tier distribution."""
    binance_symbol = symbol.replace("-", "").replace("/", "")

    price = None
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/ticker/price",
            params={"symbol": binance_symbol},
            timeout=5,
        )
        if r.status_code == 200:
            price = float(r.json()["price"])
    except Exception:
        pass

    if not price:
        try:
            r = requests.get(
                "https://api.kucoin.com/api/v1/market/orderbook/level1",
                params={"symbol": symbol},
                timeout=5,
            )
            if r.status_code == 200:
                price = float(r.json()["data"]["price"])
        except Exception:
            pass

    if not price:
        return jsonify({"success": False, "error": "Could not fetch price"}), 404

    total_oi_usd = None
    funding_rate = 0.0
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": binance_symbol},
            timeout=5,
        )
        if r.status_code == 200:
            total_oi_usd = float(r.json()["openInterest"]) * price
    except Exception:
        pass

    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": binance_symbol},
            timeout=5,
        )
        if r.status_code == 200:
            funding_rate = float(r.json().get("lastFundingRate", 0))
    except Exception:
        pass

    if not total_oi_usd:
        return jsonify(
            {
                "success": True,
                "symbol": symbol,
                "price": price,
                "total_oi_usd": None,
                "liq_levels": [],
                "no_perp": True,
            }
        )

    # Infer long/short split from funding rate
    fr_pct = funding_rate * 100
    if fr_pct > 0.05:
        long_pct = 0.65
    elif fr_pct > 0.01:
        long_pct = 0.55
    elif fr_pct < -0.05:
        long_pct = 0.35
    elif fr_pct < -0.01:
        long_pct = 0.45
    else:
        long_pct = 0.50
    short_pct = 1.0 - long_pct

    long_oi = total_oi_usd * long_pct
    short_oi = total_oi_usd * short_pct

    # Industry-estimated leverage distribution
    leverage_dist = [
        (2, 0.04),
        (3, 0.06),
        (5, 0.14),
        (10, 0.28),
        (20, 0.22),
        (25, 0.10),
        (50, 0.10),
        (100, 0.06),
    ]

    liq_levels = []
    for leverage, fraction in leverage_dist:
        # Longs liquidated below current price
        liq_levels.append(
            {
                "price": round(price * (1 - 1 / leverage), 6),
                "pct_from_price": round(-100 / leverage, 2),
                "leverage": leverage,
                "side": "long",
                "size_usd": round(long_oi * fraction, 0),
            }
        )
        # Shorts liquidated above current price
        liq_levels.append(
            {
                "price": round(price * (1 + 1 / leverage), 6),
                "pct_from_price": round(100 / leverage, 2),
                "leverage": leverage,
                "side": "short",
                "size_usd": round(short_oi * fraction, 0),
            }
        )

    max_size = max(l["size_usd"] for l in liq_levels)
    for l in liq_levels:
        l["relative_size"] = round(l["size_usd"] / max_size, 3)

    liq_levels.sort(key=lambda x: x["price"])

    return jsonify(
        {
            "success": True,
            "symbol": symbol,
            "price": price,
            "total_oi_usd": round(total_oi_usd, 0),
            "long_pct": round(long_pct, 2),
            "short_pct": round(short_pct, 2),
            "funding_rate": funding_rate,
            "liq_levels": liq_levels,
        }
    )


@app.route("/api/config/horizons")
@login_required
def get_horizons():
    return jsonify({"success": True, "horizons": HORIZONS})


# Held for the lifetime of the process that owns the monitor, so the OS keeps
# the exclusive lock and no other worker can grab it.
_monitor_lock_fd = None


def _maybe_start_monitor():
    """Start the single background supervisor thread, once across all processes.

    The supervisor's on/off state lives in the DB (shared across workers); this
    just ensures exactly one process actually runs the scan loop. Guards:
      * Dev reloader (`app.run(debug=True)`) — only the reloaded child process,
        which has WERKZEUG_RUN_MAIN=true, should start it.
      * Gunicorn with multiple workers — the `__main__` block never runs, so we
        start it on import. Every worker imports this module, so an exclusive,
        non-blocking file lock lets exactly one worker own the supervisor;
        otherwise all of them would scan and fire duplicate alerts.
    """
    global _monitor_lock_fd

    # Dev reloader: skip the supervisor process, only start in the serving child.
    if app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    import fcntl
    import tempfile

    lock_path = os.path.join(tempfile.gettempdir(), "crypto1k_monitor.lock")
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Another worker already owns the supervisor — stand down.
        fd.close()
        return

    _monitor_lock_fd = fd  # keep the fd open to hold the lock
    # Seed the shared on/off flag from config, then start the always-on
    # supervisor (it scans only while the flag is on).
    db.set_monitor_enabled(bool(MONITOR.get("enabled_on_start")))
    scanner.start_supervisor()
    logger.info("Scanner monitor supervisor started (pid %s)", os.getpid())


# Under gunicorn the `__main__` block below never executes, so trigger the
# monitor at import time. The file lock inside ensures only one worker starts it.
if sys.argv and "gunicorn" in os.path.basename(sys.argv[0]):
    _maybe_start_monitor()


if __name__ == "__main__":
    _maybe_start_monitor()
    app.run(debug=True, host="0.0.0.0", port=5000)
