import os
import time
import requests
import logging
import json
import app
class PatternMonitor:
  def __init__(self, db_path, pattern_file='patterns.json'):
    self.db_path = db_path
    self.pattern_file = pattern_file
    self.running = False
    self.min_pattern_accuracy = 0.70
    self.indicator_patterns = self.load_patterns()

    self.timeframes = {
      '30m': 30, '1h': 60, '2h': 120, '4h': 240,
      '6h': 360, '8h': 480, '12h': 720
    }

  def load_patterns(self):
    """Load patterns from file"""
    try:
      if os.path.exists(self.pattern_file):
        with open(self.pattern_file, 'r') as f:
          patterns = json.load(f)
        filtered = [p for p in patterns if p['accuracy'] >= self.min_pattern_accuracy]
        for pattern in filtered:
          pattern['parsed'] = self.parse_pattern(pattern['indicator'])
        logging.info(f"Loaded {len(filtered)} patterns")
        return sorted(filtered, key=lambda x: x['accuracy'], reverse=True)
    except Exception as e:
      logging.error(f"Failed to load patterns: {e}")
    return []

  def parse_pattern(self, pattern_string):
    """Parse pattern string"""
    components = pattern_string.split(' + ')
    parsed = {'timeframe_indicators': {}, 'required_count': len(components)}

    for component in components:
      if '[' in component and ']' in component:
        start = component.find('[') + 1
        end = component.find(']')
        timeframe = component[start:end]
        indicator = component[end + 2:].strip()

        if timeframe not in parsed['timeframe_indicators']:
          parsed['timeframe_indicators'][timeframe] = []
        parsed['timeframe_indicators'][timeframe].append(indicator)

    return parsed

  def get_priority_coins(self):
    """Get priority coins from file"""
    try:
      if os.path.exists(app.config['PRIORITY_COINS_FILE']):
        with open(app.config['PRIORITY_COINS_FILE'], 'r') as f:
          return json.load(f)
    except:
      pass
    return []

  def fetch_tabdeal_symbols(self, limit=100):
    """Fetch symbols from tabdeal"""
    try:
      response = requests.get('https://apiv2.tabdeal.ir/market/stats', timeout=10)
      response.raise_for_status()
      data = response.json()

      coins = []
      for key, val in data['stats'].items():
        if 'dayChange' in val:
          coins.append({'symbol': key, 'dayChange': val['dayChange']})

      sorted_coins = sorted(
        [x for x in coins if 'usdt' in x['symbol'].lower() and float(x['dayChange']) > 0],
        key=lambda c: float(c['dayChange']),
        reverse=True
      )

      return [c['symbol'].split('-')[0].upper() for c in sorted_coins[:limit]]
    except Exception as e:
      logging.error(f"Failed to fetch tabdeal symbols: {e}")
      return ['BTC', 'ETH', 'SOL']

  def monitor_and_save(self, symbol):
    """Monitor symbol and save to database (simplified version)"""
    # This is a placeholder - implement full monitoring logic
    # For now, just log that monitoring would happen
    logging.info(f"Monitoring {symbol}...")

  def start_monitoring(self, top_coins=100):
    """Start monitoring loop"""
    self.running = True
    logging.info("Starting pattern monitoring...")

    while self.running:
      try:
        # Get priority coins first
        priority = self.get_priority_coins()
        tabdeal = self.fetch_tabdeal_symbols(top_coins)

        # Combine with priority first
        symbols = priority + [s for s in tabdeal if s not in priority]

        logging.info(f"Monitoring {len(symbols)} symbols...")

        for symbol in symbols:
          if not self.running:
            break
          self.monitor_and_save(symbol)
          time.sleep(1)

      except Exception as e:
        logging.error(f"Monitoring error: {e}")
        time.sleep(60)

  def stop_monitoring(self):
    """Stop monitoring"""
    self.running = False
