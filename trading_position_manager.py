"""
Live Trading Position Manager
Manages trading positions with real-time monitoring and signal tracking
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)


class TradingPositionManager:
    """Manage trading positions with database persistence"""

    def __init__(self, db_path: str = 'crypto_signals.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create tables for trading positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                amount REAL NOT NULL,
                buy_datetime TIMESTAMP NOT NULL,
                sold_price REAL,
                sell_datetime TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'waiting_for_right_time_to_enter',
                
                -- Leveraged trading fields
                liquidation_at REAL,
                break_even REAL,
                leverage_multiplier REAL,
                
                -- Stop loss/profit
                stop_loss REAL,
                stop_profit REAL,
                
                -- Entry/Exit reasons
                entry_reason TEXT,
                exit_reason TEXT,
                
                -- Calculated fields
                pnl REAL,
                pnl_percentage REAL,
                
                -- Metadata
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                CHECK(status IN ('waiting_for_right_time_to_enter', 'open', 'sold'))
            )
        ''')

        # Position signals table (track signals for each position)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER NOT NULL,
                signal_name TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                detected_at TIMESTAMP NOT NULL,
                price_at_detection REAL NOT NULL,
                FOREIGN KEY (position_id) REFERENCES trading_positions(id) ON DELETE CASCADE
            )
        ''')

        # Signal fact-check results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_fact_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                detected_at TIMESTAMP NOT NULL,
                price_at_detection REAL NOT NULL,
                
                -- Fact-check results
                actual_move TEXT,
                predicted_correctly BOOLEAN,
                price_change_pct REAL,
                
                -- Check timing
                checked_at TIMESTAMP NOT NULL,
                candles_elapsed INTEGER NOT NULL,
                exit_reason TEXT,
                validation_window INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Signal confidence adjustments
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_confidence_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                original_confidence INTEGER NOT NULL,
                adjusted_confidence INTEGER NOT NULL,
                accuracy_rate REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(signal_name, timeframe)
            )
        ''')

        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_positions_status 
            ON trading_positions(status, symbol)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_position_signals_position 
            ON position_signals(position_id, detected_at DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_fact_checks_signal 
            ON signal_fact_checks(signal_name, timeframe, detected_at DESC)
        ''')

        conn.commit()
        conn.close()
        logging.info("✅ Trading position database initialized")

    def create_position(self, symbol: str, entry_price: float, amount: float,
                       buy_datetime: datetime = None, status: str = 'waiting_for_right_time_to_enter',
                       leverage_multiplier: float = None, liquidation_at: float = None,
                       break_even: float = None, stop_loss: float = None,
                       stop_profit: float = None, entry_reason: str = None,
                       notes: str = None) -> int:
        """Create a new trading position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if buy_datetime is None:
            buy_datetime = datetime.now()

        cursor.execute('''
            INSERT INTO trading_positions (
                symbol, entry_price, amount, buy_datetime, status,
                leverage_multiplier, liquidation_at, break_even,
                stop_loss, stop_profit, entry_reason, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, entry_price, amount, buy_datetime, status,
            leverage_multiplier, liquidation_at, break_even,
            stop_loss, stop_profit, entry_reason, notes
        ))

        position_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logging.info(f"✅ Created position #{position_id}: {symbol} @ ${entry_price}")
        return position_id

    def update_position(self, position_id: int, **kwargs) -> bool:
        """Update position fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        allowed_fields = {
            'entry_price', 'amount', 'sold_price', 'sell_datetime', 'status',
            'liquidation_at', 'break_even', 'leverage_multiplier',
            'stop_loss', 'stop_profit', 'entry_reason', 'exit_reason',
            'pnl', 'pnl_percentage', 'notes'
        }

        updates = []
        values = []

        for key, value in kwargs.items():
            if key in allowed_fields:
                updates.append(f"{key} = ?")
                values.append(value)

        if not updates:
            conn.close()
            return False

        updates.append("updated_at = ?")
        values.append(datetime.now())
        values.append(position_id)

        query = f"UPDATE trading_positions SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, values)

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if success:
            logging.info(f"✅ Updated position #{position_id}")

        return success

    def close_position(self, position_id: int, sold_price: float,
                      exit_reason: str = None) -> bool:
        """Close a position and calculate PnL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get position details
        cursor.execute('SELECT entry_price, amount FROM trading_positions WHERE id = ?',
                      (position_id,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return False

        entry_price, amount = result

        # Calculate PnL
        pnl = (sold_price - entry_price) * amount
        pnl_percentage = ((sold_price - entry_price) / entry_price) * 100

        # Update position
        cursor.execute('''
            UPDATE trading_positions 
            SET sold_price = ?, sell_datetime = ?, status = 'sold',
                exit_reason = ?, pnl = ?, pnl_percentage = ?, updated_at = ?
            WHERE id = ?
        ''', (sold_price, datetime.now(), exit_reason, pnl, pnl_percentage,
              datetime.now(), position_id))

        conn.commit()
        conn.close()

        logging.info(f"✅ Closed position #{position_id}: PnL ${pnl:.2f} ({pnl_percentage:.2f}%)")
        return True

    def delete_position(self, position_id: int) -> bool:
        """Delete a position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM trading_positions WHERE id = ?', (position_id,))
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()

        if success:
            logging.info(f"🗑️ Deleted position #{position_id}")

        return success

    def get_position(self, position_id: int) -> Optional[Dict]:
        """Get single position by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM trading_positions WHERE id = ?', (position_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_all_positions(self, status: str = None, symbol: str = None) -> List[Dict]:
        """Get all positions with optional filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = 'SELECT * FROM trading_positions WHERE 1=1'
        params = []

        if status:
            query += ' AND status = ?'
            params.append(status)

        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)

        query += ' ORDER BY created_at DESC'

        cursor.execute(query, params)
        positions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return positions

    def add_signal_to_position(self, position_id: int, signal_name: str,
                              signal_type: str, timeframe: str, confidence: int,
                              price_at_detection: float) -> int:
        """Track a signal detected for a position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO position_signals (
                position_id, signal_name, signal_type, timeframe,
                confidence, detected_at, price_at_detection
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            position_id, signal_name, signal_type, timeframe,
            confidence, datetime.now(), price_at_detection
        ))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return signal_id

    def get_position_signals(self, position_id: int, limit: int = 100) -> List[Dict]:
        """Get all signals for a position"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM position_signals 
            WHERE position_id = ?
            ORDER BY detected_at DESC
            LIMIT ?
        ''', (position_id, limit))

        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return signals

    def calculate_stop_loss_profit(self, signal_type: str, entry_price: float,
                                   confidence: int, leverage: float = 1.0) -> Dict:
        """Calculate suggested stop loss and stop profit based on signal confidence"""
        # Base percentages (adjust based on your risk tolerance)
        if confidence >= 85:
            stop_loss_pct = 2.0
            stop_profit_pct = 8.0
        elif confidence >= 75:
            stop_loss_pct = 3.0
            stop_profit_pct = 6.0
        elif confidence >= 65:
            stop_loss_pct = 4.0
            stop_profit_pct = 5.0
        else:
            stop_loss_pct = 5.0
            stop_profit_pct = 4.0

        # Adjust for leverage
        if leverage > 1.0:
            stop_loss_pct = stop_loss_pct / leverage
            stop_profit_pct = stop_profit_pct * (1 + (leverage - 1) * 0.5)

        if signal_type == 'BUY':
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            stop_profit = entry_price * (1 + stop_profit_pct / 100)
        else:  # SELL
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            stop_profit = entry_price * (1 - stop_profit_pct / 100)

        return {
            'stop_loss': stop_loss,
            'stop_profit': stop_profit,
            'stop_loss_pct': stop_loss_pct,
            'stop_profit_pct': stop_profit_pct,
            'risk_reward_ratio': stop_profit_pct / stop_loss_pct
        }

    def get_position_summary(self) -> Dict:
        """Get summary statistics of all positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Overall stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_positions,
                SUM(CASE WHEN status = 'sold' THEN 1 ELSE 0 END) as closed_positions,
                SUM(CASE WHEN status = 'waiting_for_right_time_to_enter' THEN 1 ELSE 0 END) as waiting,
                AVG(CASE WHEN pnl IS NOT NULL THEN pnl END) as avg_pnl,
                SUM(CASE WHEN pnl IS NOT NULL THEN pnl END) as total_pnl,
                AVG(CASE WHEN pnl_percentage IS NOT NULL THEN pnl_percentage END) as avg_pnl_pct,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades
            FROM trading_positions
        ''')

        stats = cursor.fetchone()
        conn.close()

        return {
            'total_positions': stats[0] or 0,
            'open_positions': stats[1] or 0,
            'closed_positions': stats[2] or 0,
            'waiting_positions': stats[3] or 0,
            'avg_pnl': stats[4] or 0,
            'total_pnl': stats[5] or 0,
            'avg_pnl_percentage': stats[6] or 0,
            'winning_trades': stats[7] or 0,
            'losing_trades': stats[8] or 0,
            'win_rate': (stats[7] / max(stats[2], 1)) * 100 if stats[2] else 0
        }