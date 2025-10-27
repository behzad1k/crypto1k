# Crypto Pattern Dashboard - Complete Integration Guide

## Overview

This guide shows you how to integrate your crypto monitoring script into the Flask dashboard for a unified application with:

- ✅ Real-time monitoring with start/stop controls
- ✅ Live status updates every 3 seconds
- ✅ Background thread execution
- ✅ Shared SQLite database
- ✅ Priority coin management
- ✅ Historical signal tracking
- ✅ Excel export functionality

## File Structure

```
crypto_dashboard/
├── app.py                      # Main Flask application (UPDATED)
├── monitor.py                  # Monitoring module (NEW)
├── requirements.txt            # Python dependencies
├── templates/
│   ├── login.html             # Login page
│   └── index.html             # Dashboard with monitoring controls (UPDATED)
├── patterns.json              # Your indicator patterns
├── priority_coins.json        # Priority coins (auto-created)
├── crypto_signals.db          # SQLite database (auto-created)
└── README.md
```

## Installation Steps

### 1. Backup Your Current Setup

```bash
# Backup your database
cp crypto_signals.db crypto_signals.db.backup

# Backup your patterns file
cp patterns.json patterns.json.backup
```

### 2. Create Virtual Environment (if not exists)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Update Files

Replace/create the following files:

1. **monitor.py** - New file with monitoring logic
2. **app.py** - Updated Flask app with monitoring integration
3. **templates/index.html** - Updated dashboard with controls
4. **requirements.txt** - Same dependencies

### 5. Configure Credentials

Edit `app.py` and change the login credentials:

```python
# Line 31-32
if data.get('username') == 'YOUR_USERNAME' and data.get('password') == 'YOUR_PASSWORD':
```

Also change the secret key:

```python
# Line 16
app.secret_key = 'your-unique-secret-key-here-make-it-random'
```

### 6. Verify Patterns File

Make sure your `patterns.json` exists and has the correct format:

```json
[
  {
    "indicator": "[12h] Momentum_5 + [2h] MACD_position + [30m] WILLR_oversold",
    "accuracy": 0.8461538461538461,
    "success": 11,
    "total": 13
  }
]
```

### 7. Test the Application

```bash
# Development mode
python app.py
```

Open browser: `http://localhost:5000`

## Using the Dashboard

### Login Page

1. Navigate to `http://your-server:5000`
2. Enter credentials (default: admin/admin123 - CHANGE THIS!)
3. Click "Sign In"

### Signals Page

**Features:**
- View all detected signals
- Filter by symbol, accuracy, pattern count
- Click symbol to view history
- Export to Excel with filters
- Real-time status indicator in header

### Control Panel

**Monitor Control:**
- **Start Button**: Starts background monitoring
- **Stop Button**: Stops monitoring gracefully
- **Real-time Stats**:
  - Status (Running/Stopped)
  - Symbols Processed (count)
  - Alerts Triggered (count)
  - Current Symbol (what's being analyzed now)
  - Last Update (timestamp)

**Priority Coins:**
- Add coins to monitor first (comma-separated)
- Example: `BTC, ETH, SOL, NEAR, APT`
- Click "Save Settings"
- Priority coins are checked before tabdeal fetch

**System Info:**
- View current thresholds
- Pattern count
- Configuration settings

### Status Indicator

In the header, you'll see a status badge:
- **Green (pulsing)**: Monitoring is active
- **Gray (static)**: Monitoring is offline

Status updates automatically every 3 seconds.

## How It Works

### Architecture

```
┌─────────────────────────────────────────────┐
│           Flask Application                  │
│  ┌────────────────────────────────────────┐ │
│  │  Main Thread (Flask Server)            │ │
│  │  - Handles HTTP requests               │ │
│  │  - Serves dashboard                    │ │
│  │  - API endpoints                       │ │
│  └────────────────────────────────────────┘ │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │  Background Thread (Monitor)           │ │
│  │  - Fetches symbols from tabdeal        │ │
│  │  - Analyzes patterns                   │ │
│  │  - Saves signals to database           │ │
│  │  - Updates real-time stats             │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
                      ↓
           ┌──────────────────┐
           │  SQLite Database │
           │  (shared access) │
           └──────────────────┘
```

### Monitoring Flow

1. **Start Monitoring** → Creates `CryptoPatternMonitor` instance
2. **Background Thread** → Runs `monitor.run()` in daemon thread
3. **Continuous Loop**:
   - Load priority coins from file
   - Fetch top 100 coins from tabdeal
   - Combine lists (priority first)
   - Analyze each symbol:
     - Fetch 30m OHLCV data
     - Calculate indicators across timeframes
     - Check pattern matches
     - If thresholds met → Save to database
   - Update statistics
   - Repeat after 10 seconds

4. **Dashboard Polling** → Frontend polls `/api/monitor/status` every 3 seconds
5. **Real-time Updates** → Stats displayed in control panel

### Thread Safety

- Database writes are serialized (SQLite handles this)
- Monitor can be safely started/stopped
- Stats are updated atomically
- No race conditions

## Production Deployment

### Using Systemd (Recommended)

Create `/etc/systemd/system/crypto-dashboard.service`:

```ini
[Unit]
Description=Crypto Pattern Dashboard
After=network.target

[Service]
User=your-username
WorkingDirectory=/path/to/crypto_dashboard
Environment="PATH=/path/to/crypto_dashboard/venv/bin"
ExecStart=/path/to/crypto_dashboard/venv/bin/gunicorn -w 1 -b 0.0.0.0:5000 --timeout 300 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Important**: Use `-w 1` (single worker) because:
- Background monitoring thread should run once
- Multiple workers would create multiple monitoring threads
- Avoid duplicate alerts

Enable and start:

```bash
sudo systemctl enable crypto-dashboard
sudo systemctl start crypto-dashboard
sudo systemctl status crypto-dashboard
```

View logs:

```bash
sudo journalctl -u crypto-dashboard -f
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
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

Enable:

```bash
sudo ln -s /etc/nginx/sites-available/crypto-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### SSL Certificate (Optional)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Configuration

### Monitoring Thresholds

Edit `monitor.py` to change thresholds:

```python
# Line 31-33
self.min_pattern_accuracy = 0.70   # 70% minimum accuracy
self.min_pattern_count = 100       # 100+ patterns required
self.min_confidence = 0.80         # 80%+ confidence required
```

### Symbol Fetching

Edit `monitor.py` to change how many coins to fetch:

```python
# In app.py, line 142
monitor_thread = threading.Thread(
    target=monitor.run,
    args=(100,),  # Change 100 to desired number
    daemon=True
)
```

### Monitoring Interval

Edit `monitor.py` to change loop timing:

```python
# Line 420
time.sleep(1)  # Delay between symbols (seconds)

# Line 423  
time.sleep(10)  # Delay between full loops (seconds)
```

## Troubleshooting

### Monitor Won't Start

**Check logs:**

```bash
# If using systemd
sudo journalctl -u crypto-dashboard -f

# If running manually
# Check console output
```

**Common issues:**
- Patterns file not found → Check `patterns.json` exists
- Database locked → Stop all running instances
- Port in use → Change port in `app.py` line 160

### Database Issues

**Reset database:**

```bash
# Backup first!
cp crypto_signals.db crypto_signals.db.old

# Delete and restart app (will recreate)
rm crypto_signals.db
python app.py
```

**Check database:**

```bash
sqlite3 crypto_signals.db
.tables
.schema pattern_signals
SELECT COUNT(*) FROM pattern_signals;
.quit
```

### Monitor Stuck

If monitoring seems stuck:

1. Stop via dashboard "Stop" button
2. Check logs for errors
3. Restart monitoring
4. If needed, restart entire application

### High CPU Usage

If CPU usage is high:

1. Increase delays in `monitor.py`:
   - `time.sleep(1)` → `time.sleep(2)` between symbols
   - `time.sleep(10)` → `time.sleep(30)` between loops

2. Reduce number of coins:
   - Change `args=(100,)` → `args=(50,)` in app.py

## Monitoring Best Practices

### 1. Start Small

- Begin with 20-30 priority coins
- Monitor for a few hours
- Check database size and performance
- Gradually increase if needed

### 2. Priority Coins

Add high-volume, stable coins first:
```
BTC, ETH, SOL, BNB, ADA, DOT, MATIC, AVAX, LINK, UNI
```

### 3. Database Maintenance

**Weekly cleanup:**

```sql
-- Delete signals older than 30 days
DELETE FROM pattern_signals 
WHERE datetime_created < datetime('now', '-30 days');

-- Vacuum database
VACUUM;
```

### 4. Backup Strategy

**Daily backups:**

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d)
cp crypto_signals.db "backups/crypto_signals_$DATE.db"

# Keep only last 7 days
find backups/ -name "crypto_signals_*.db" -mtime +7 -delete
```

Add to crontab:

```bash
crontab -e
# Add line:
0 2 * * * /path/to/backup.sh
```

## API Endpoints

### Monitoring Control

- `POST /api/monitor/start` - Start monitoring
- `POST /api/monitor/stop` - Stop monitoring  
- `GET /api/monitor/status` - Get real-time status

### Signals

- `GET /api/signals?page=1&minAccuracy=0.8&minPatterns=100&symbol=BTC` - Get signals
- `GET /api/symbols/<symbol>` - Get symbol history
- `GET /api/export` - Export to Excel

### Configuration

- `GET /api/priority-coins` - Get priority coins
- `POST /api/priority-coins` - Set priority coins

## Security Recommendations

1. **Change default credentials** in `app.py`
2. **Change secret key** to random string
3. **Use HTTPS** in production (via nginx + certbot)
4. **Firewall rules**:
   ```bash
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

5. **Regular updates**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## Performance Optimization

### For Large Scale (200+ coins)

1. **Database Indexing** (already done):
   ```sql
   CREATE INDEX idx_symbol_datetime ON pattern_signals(symbol, datetime_created);
   ```

2. **Pagination** (already implemented in API)

3. **Caching** - Add Redis for frequently accessed data

4. **Multiple Workers** - Use Celery for distributed monitoring

## Support

For issues:

1. Check logs: `journalctl -u crypto-dashboard -f`
2. Verify database: `sqlite3 crypto_signals.db .schema`
3. Test API: `curl http://localhost:5000/api/monitor/status`
4. Check patterns: Number and format in `patterns.json`

## License

Your existing license applies.