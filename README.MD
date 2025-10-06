# Crypto Pattern Dashboard - Installation Guide

## Complete Flask Application with Integrated Frontend

This is a complete, production-ready Flask application that combines your crypto pattern monitoring system with a modern web dashboard.

## Directory Structure

```
crypto_dashboard/
├── app.py                      # Main Flask application
├── monitor.py                  # Pattern monitoring logic (separated)
├── requirements.txt            # Python dependencies
├── templates/
│   ├── login.html             # Login page
│   └── index.html             # Main dashboard
├── patterns.json              # Your indicator patterns
├── priority_coins.json        # Priority coins list (auto-created)
├── crypto_signals.db          # SQLite database (auto-created)
└── README.md                  # This file
```

## Installation Steps

### 1. System Requirements
- Python 3.8+
- Linux server (Ubuntu/Debian recommended)
- 2GB RAM minimum
- Internet connection for API calls

### 2. Install Python Dependencies

```bash
# Create project directory
mkdir crypto_dashboard
cd crypto_dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Create Required Files

Save the Flask app as `app.py` and create the templates directory:

```bash
mkdir templates
# Save login.html and index.html in templates/
```

### 4. Create patterns.json

Create a `patterns.json` file with your indicator patterns:

```json
[
  {
    "indicator": "[12h] Momentum_5 + [2h] MACD_position + [30m] WILLR_oversold",
    "accuracy": 0.8461538461538461,
    "success": 11,
    "total": 13
  },
  {
    "indicator": "[12h] Momentum_5 + [8h] MOM_5_10_bullish + [8h] Momentum_10",
    "accuracy": 0.7647058823529411,
    "success": 13,
    "total": 17
  }
]
```

### 5. Configuration

Edit `app.py` and change the secret key:

```python
app.secret_key = 'your-unique-secret-key-here'  # IMPORTANT: Change this!
```

### 6. Run the Application

```bash
# Development mode
python app.py

# Production mode (recommended)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 7. Access the Dashboard

Open your browser and navigate to:
- `http://your-server-ip:5000`

## Production Deployment

### Using Systemd Service

Create `/etc/systemd/system/crypto-dashboard.service`:

```ini
[Unit]
Description=Crypto Pattern Dashboard
After=network.target

[Service]
User=your-username
WorkingDirectory=/path/to/crypto_dashboard
Environment="PATH=/path/to/crypto_dashboard/venv/bin"
ExecStart=/path/to/crypto_dashboard/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable crypto-dashboard
sudo systemctl start crypto-dashboard
sudo systemctl status crypto-dashboard
```

### Using Nginx Reverse Proxy

Install nginx:

```bash
sudo apt install nginx
```

Create `/etc/nginx/sites-available/crypto-dashboard`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/crypto-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### SSL Certificate (Optional but Recommended)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Features

### Dashboard Pages

1. **Signals List**
   - Paginated view of all signals
   - Filters: Symbol, Min Accuracy, Min Patterns
   - Export to Excel with filters
   - Click symbol to view history

2. **Symbol Detail**
   - Complete signal history per coin
   - Statistics and trends
   - Validity period calculation

3. **Control Panel**
   - Set priority coins
   - View system configuration
   - System status

### Signal Validity Calculation

Signals are valid for **12-60 hours** based on:
- **Confidence Level** (60% weight): Higher confidence = longer validity
- **Pattern Count** (40% weight): More patterns = longer validity

Formula:
```
validity = 12 + (48 * ((confidence_factor * 0.6) + (pattern_factor * 0.4)))
```

### Security Features

- Session-based authentication
- Login required for all pages
- Configurable credentials
- CSRF protection via Flask sessions

## Monitoring Integration

The monitoring logic from your original script is integrated. To run continuous monitoring:

1. The monitor runs in a separate thread
2. Priority coins are checked first
3. Then fetches from Nobitex API
4. Saves signals directly to database
5. Dashboard updates automatically

## Database Schema

SQLite database (`crypto_signals.db`) with table:

```sql
CREATE TABLE pattern_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    signal TEXT NOT NULL,
    pattern_confidence REAL NOT NULL,
    pattern_count INTEGER NOT NULL,
    best_pattern TEXT NOT NULL,
    best_pattern_accuracy REAL NOT NULL,
    all_patterns TEXT,
    price REAL NOT NULL,
    stop_loss REAL,
    datetime_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Troubleshooting

### Database Issues
```bash
# Reset database
rm crypto_signals.db
python app.py  # Will recreate
```

### Port Already in Use
```bash
# Find process using port 5000
sudo lsof -i :5000
# Kill process
sudo kill -9 <PID>
```

### Permission Denied
```bash
# Fix file permissions
chmod +x app.py
chmod 755 crypto_dashboard/
```

## Customization

### Change Login Credentials

Edit the login route in `app.py`:

```python
if data.get('username') == 'YOUR_USER' and data.get('password') == 'YOUR_PASS':
```

### Change Thresholds

Edit in `app.py` or `monitor.py`:

```python
self.min_pattern_accuracy = 0.70  # 70% minimum
```

### Change Port

```python
app.run(host='0.0.0.0', port=8080)  # Change to 8080
```

## Maintenance

### Backup Database

```bash
# Daily backup
cp crypto_signals.db crypto_signals.db.$(date +%Y%m%d)
```

### Clean Old Signals

```sql
DELETE FROM pattern_signals 
WHERE datetime_created < datetime('now', '-30 days');
```

### View Logs

```bash
# If using systemd
sudo journalctl -u crypto-dashboard -f
```

## Support

For issues:
1. Check logs: `journalctl -u crypto-dashboard -f`
2. Verify database: `sqlite3 crypto_signals.db .schema`
3. Test API: `curl http://localhost:5000/api/signals`

## License

Your existing license applies.