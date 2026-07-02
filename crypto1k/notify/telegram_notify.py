"""
Telegram alerting via the Bot API.

Free, no SMTP needed — messages go out over HTTPS (api.telegram.org:443), which
works even where outbound SMTP is blocked. Configuration lives in
scanner_config.py (TELEGRAM: bot_token + chat_id).

Setup (one-time):
  1. Talk to @BotFather in Telegram → /newbot → copy the bot TOKEN.
  2. Create a channel (or group), add the bot as an ADMIN.
  3. Get the chat id:
       - Public channel: use "@your_channel_username" as chat_id.
       - Private channel/group: forward a message to @userinfobot, or call
         https://api.telegram.org/bot<TOKEN>/getUpdates and read chat.id
         (private channels look like -1001234567890).

Public surface:
  is_configured()  -> bool
  send_message(text) -> bool
  send_alert(r)    -> bool
  send_test()      -> (ok, message)
"""

import logging

import requests

from crypto1k.config.scanner_config import TELEGRAM

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org"
_TIMEOUT = 10


def is_configured() -> bool:
    if not TELEGRAM.get("enabled"):
        return False
    token = TELEGRAM.get("bot_token") or ""
    chat = TELEGRAM.get("chat_id") or ""
    if not token or "YOUR_" in token:
        return False
    if not chat or "YOUR_" in str(chat):
        return False
    return True


def send_message(text: str) -> bool:
    """Send a plain-text message to the configured chat/channel."""
    if not is_configured():
        logger.info("Telegram not configured — skipping send")
        return False

    url = f"{_API}/bot{TELEGRAM['bot_token']}/sendMessage"
    payload = {
        "chat_id": TELEGRAM["chat_id"],
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.warning("Telegram send failed (HTTP %s): %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:
        logger.warning("Telegram send error: %s", e)
        return False


def send_alert(r: dict) -> bool:
    """Send a scanner alert. Reuses the plain-English summary + chart link."""
    summary = r.get("summary") or f"{r.get('symbol', '?')} alert"
    lines = [summary]
    if r.get("url"):
        lines.append(f"\n📊 Chart: {r['url']}")
    return send_message("\n".join(lines))


def send_test() -> tuple:
    """Send a test message. Returns (ok, human-readable message)."""
    if not is_configured():
        return False, "Telegram is not configured (set TELEGRAM in scanner_config.py)."
    ok = send_message("✅ Crypto1k test alert — your Telegram channel is wired up.")
    if ok:
        return True, f"Test message sent to {TELEGRAM['chat_id']}."
    return False, "Telegram send failed — check the bot token, chat_id, and that the bot is a channel admin."
