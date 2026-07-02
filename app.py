"""
Entry-point shim.

The real Flask app lives in crypto1k/web/app.py; this keeps the existing
deploy commands working unchanged:
    gunicorn app:app
    python app.py
"""

from crypto1k.web.app import app, _maybe_start_monitor

if __name__ == "__main__":
    _maybe_start_monitor()
    app.run(debug=True, host="0.0.0.0", port=5000)
