"""
Email alerting via SMTP.

Credentials and the recipient list live in scanner_config.py (EMAIL + RECIPIENTS)
so you can fill them in one place. Nothing here needs editing.

Public surface:
  is_configured()        -> bool   (enabled + credentials + recipients present)
  recipient_count()      -> int
  send_alert(result)     -> bool   (builds + sends the alert email)
  send_test()            -> (bool, message)
"""

import ssl
import socket
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from scanner_config import EMAIL, RECIPIENTS

logger = logging.getLogger(__name__)


# Many cloud hosts hand out an IPv6 address but have no working IPv6 route, so
# Python tries the SMTP server's IPv6 address first and fails immediately with
# "[Errno 101] Network is unreachable". These subclasses force the connection
# over IPv4 while still verifying TLS against the real hostname (so the cert
# check stays valid — we connect by IP but validate by name).

def _ipv4_socket(host, port, timeout, source_address):
    ip = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4]
    return socket.create_connection(ip, timeout, source_address)


class _IPv4SMTP(smtplib.SMTP):
    def _get_socket(self, host, port, timeout):
        return _ipv4_socket(host, port, timeout, self.source_address)


class _IPv4SMTP_SSL(smtplib.SMTP_SSL):
    def _get_socket(self, host, port, timeout):
        sock = _ipv4_socket(host, port, timeout, self.source_address)
        return self.context.wrap_socket(sock, server_hostname=self._host)


def is_configured() -> bool:
    if not EMAIL.get('enabled'):
        return False
    if not RECIPIENTS:
        return False
    if 'YOUR_' in (EMAIL.get('username') or '') or 'YOUR_' in (EMAIL.get('password') or ''):
        return False
    return True


def recipient_count() -> int:
    return len(RECIPIENTS)


# ── SMTP send ───────────────────────────────────────────────────────────────

def _send(subject: str, html_body: str, text_body: str) -> bool:
    if not is_configured():
        logger.info('Email not configured — skipping send')
        return False

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = f"{EMAIL['from_name']} <{EMAIL['from_address']}>"
    msg['To'] = ', '.join(RECIPIENTS)
    msg.attach(MIMEText(text_body, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))

    host, port = EMAIL['smtp_host'], EMAIL['smtp_port']
    try:
        if EMAIL.get('use_ssl'):
            ctx = ssl.create_default_context()
            with _IPv4SMTP_SSL(host, port, context=ctx, timeout=15) as s:
                s.login(EMAIL['username'], EMAIL['password'])
                s.sendmail(EMAIL['from_address'], RECIPIENTS, msg.as_string())
        else:
            with _IPv4SMTP(host, port, timeout=15) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(EMAIL['username'], EMAIL['password'])
                s.sendmail(EMAIL['from_address'], RECIPIENTS, msg.as_string())
        logger.info(f'Sent email to {len(RECIPIENTS)} recipient(s): {subject}')
        return True
    except Exception as e:
        logger.error(f'SMTP send failed: {e}')
        raise


# ── Alert email ─────────────────────────────────────────────────────────────

def _fmt(n, prefix='$'):
    if n is None:
        return '—'
    return f'{prefix}{n:,.0f}'


def send_alert(r: dict) -> bool:
    subject = f"🚨 Volume alert: {r['symbol']} ({r['label']} {r['score']}/10, {r['direction']})"

    sig_rows = ''.join(
        f"<li><b>{s['name'].replace('_', ' ').title()}</b> "
        f"<span style='color:{'#16a34a' if s['direction']=='bullish' else '#dc2626' if s['direction']=='bearish' else '#666'}'>"
        f"({s['direction']})</span> — {s['detail']}</li>"
        for s in r.get('signals', [])
    ) or '<li>No individual signals fired.</li>'

    m = r['metrics']
    html = f"""\
<div style="font-family:Inter,Arial,sans-serif;max-width:620px;color:#1a1a1a">
  <h2 style="margin:0 0 4px">🚨 {r['symbol']} — unusual volume detected</h2>
  <p style="margin:0 0 16px;color:#555">{r.get('name') or ''} · {r['dex']} / {r['chain']}</p>

  <div style="background:#f4f4f5;border-radius:10px;padding:16px;margin-bottom:16px">
    <p style="margin:0;font-size:15px;line-height:1.5">{r['summary']}</p>
  </div>

  <table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:16px">
    <tr><td style="padding:4px 0;color:#666">Price</td><td style="text-align:right"><b>${r['price_usd']}</b></td></tr>
    <tr><td style="padding:4px 0;color:#666">Setup score</td><td style="text-align:right"><b>{r['score']}/10 · {r['label']}</b></td></tr>
    <tr><td style="padding:4px 0;color:#666">1h volume vs avg pace</td><td style="text-align:right"><b>{m['vol_pace_1h']}×</b></td></tr>
    <tr><td style="padding:4px 0;color:#666">5m volume vs avg pace</td><td style="text-align:right"><b>{m['vol_pace_5m']}×</b></td></tr>
    <tr><td style="padding:4px 0;color:#666">Price change (1h)</td><td style="text-align:right"><b>{r['price_change']['h1']:+.1f}%</b></td></tr>
    <tr><td style="padding:4px 0;color:#666">Buy/sell ratio (1h)</td><td style="text-align:right"><b>{m['buy_ratio_1h']}:1</b></td></tr>
    <tr><td style="padding:4px 0;color:#666">Liquidity</td><td style="text-align:right"><b>{_fmt(r['liquidity_usd'])}</b></td></tr>
    <tr><td style="padding:4px 0;color:#666">24h volume</td><td style="text-align:right"><b>{_fmt(r['volume']['h24'])}</b></td></tr>
    <tr><td style="padding:4px 0;color:#666">Market cap</td><td style="text-align:right"><b>{_fmt(r['market_cap'])}</b></td></tr>
  </table>

  <h3 style="margin:0 0 8px">Signals</h3>
  <ul style="margin:0 0 16px;padding-left:18px;line-height:1.6">{sig_rows}</ul>

  <a href="{r['url']}" style="display:inline-block;background:#6366f1;color:#fff;
     padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600">
     View chart on DexScreener →</a>

  <p style="margin:18px 0 0;color:#999;font-size:12px">
    Sent by Crypto1k Scanner. This is not financial advice — do your own research.
  </p>
</div>"""

    # Plain-text fallback
    text = (
        f"{r['symbol']} — unusual volume detected ({r['label']} {r['score']}/10, {r['direction']})\n\n"
        f"{r['summary']}\n\n"
        f"Price: ${r['price_usd']}\n"
        f"1h vol pace: {m['vol_pace_1h']}x | 5m vol pace: {m['vol_pace_5m']}x\n"
        f"1h change: {r['price_change']['h1']:+.1f}% | buy/sell: {m['buy_ratio_1h']}:1\n"
        f"Liquidity: {_fmt(r['liquidity_usd'])} | 24h vol: {_fmt(r['volume']['h24'])}\n\n"
        f"Signals:\n" + '\n'.join(f"  - {s['name']}: {s['detail']}" for s in r.get('signals', [])) +
        f"\n\nChart: {r['url']}\n"
    )

    return _send(subject, html, text)


def send_test() -> tuple:
    """Send a simple test email to verify SMTP works. Returns (ok, message)."""
    if not EMAIL.get('enabled'):
        return False, 'EMAIL.enabled is False in scanner_config.py'
    if not RECIPIENTS:
        return False, 'RECIPIENTS list is empty in scanner_config.py'
    if not is_configured():
        return False, 'Email credentials still contain placeholder values'
    try:
        ok = _send(
            '✅ Crypto1k Scanner — test email',
            "<div style='font-family:Arial'>If you can read this, your scanner email "
            "alerts are configured correctly. 🎉</div>",
            'Your scanner email alerts are configured correctly.',
        )
        return ok, ('Sent' if ok else 'Not sent')
    except Exception as e:
        return False, str(e)
