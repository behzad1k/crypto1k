"""
Signal configuration: families, roles, grades, and value metadata for all signals.

Families group signals that come from the same underlying indicator.
This prevents counting multiple signals from the same source as independent confluence.

Grades:
  A - Primary signals. Complex, multi-condition. Meaningful on their own.
  B - Confirmation signals. Single condition, adds weight when supporting a primary signal.
  C - Context signals. Describes market state. Useful as background, not as triggers.

Roles:
  primary      - Can drive a trading decision alone (A-grade only)
  confirmation - Supports a primary signal
  context      - Background state, not a trigger

value_type:
  indicator    - The raw indicator reading (RSI: 22.8, ADX: 36.5, CCI: -110)
  price_level  - An actual price level (support at $68,400, POC at $69,100)
  None         - No meaningful numerical value returned

Categories:
  momentum    - Rate of price change, overbought/oversold
  trend       - Direction and strength of trend
  volume      - Buying/selling pressure via volume
  volatility  - Price range expansion/contraction
  structure   - Market structure: support, resistance, breaks
  smc         - Smart money concepts: order blocks, FVG, liquidity
  pattern     - Candlestick formations
"""

SIGNAL_FAMILIES = {

    # ── MOMENTUM ─────────────────────────────────────────────────────────────

    'rsi': {
        'name': 'RSI',
        'description': 'Relative Strength Index — measures momentum and identifies overbought/oversold conditions',
        'category': 'momentum',
        'signals': {
            'rsi_divergence_bullish':    {'grade': 'A', 'role': 'primary',       'direction': 'bullish', 'value_type': 'indicator',  'value_label': 'RSI'},
            'rsi_divergence_bearish':    {'grade': 'A', 'role': 'primary',       'direction': 'bearish', 'value_type': 'indicator',  'value_label': 'RSI'},
            'rsi_centerline_cross_up':   {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': 'indicator',  'value_label': 'RSI'},
            'rsi_centerline_cross_down': {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': 'indicator',  'value_label': 'RSI'},
            'rsi_oversold':              {'grade': 'C', 'role': 'context',       'direction': 'bullish', 'value_type': 'indicator',  'value_label': 'RSI'},
            'rsi_overbought':            {'grade': 'C', 'role': 'context',       'direction': 'bearish', 'value_type': 'indicator',  'value_label': 'RSI'},
        }
    },

    'macd': {
        'name': 'MACD',
        'description': 'Moving Average Convergence Divergence — tracks momentum shifts and trend changes',
        'category': 'momentum',
        'signals': {
            'macd_divergence_bullish':         {'grade': 'A', 'role': 'primary',       'direction': 'bullish', 'value_type': None,         'value_label': None},
            'macd_divergence_bearish':         {'grade': 'A', 'role': 'primary',       'direction': 'bearish', 'value_type': None,         'value_label': None},
            'macd_cross_bullish':              {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None,         'value_label': None},
            'macd_cross_bearish':              {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None,         'value_label': None},
            'macd_histogram_reversal_bullish': {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': 'indicator',  'value_label': 'Hist'},
            'macd_histogram_reversal_bearish': {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': 'indicator',  'value_label': 'Hist'},
        }
    },

    'stochastic': {
        'name': 'Stochastic',
        'description': 'Stochastic Oscillator — compares closing price to price range, signals reversals',
        'category': 'momentum',
        'signals': {
            'stoch_divergence_bullish': {'grade': 'A', 'role': 'primary',       'direction': 'bullish', 'value_type': None,        'value_label': None},
            'stoch_divergence_bearish': {'grade': 'A', 'role': 'primary',       'direction': 'bearish', 'value_type': None,        'value_label': None},
            'stoch_cross_bullish':      {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': 'indicator', 'value_label': 'Stoch'},
            'stoch_cross_bearish':      {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': 'indicator', 'value_label': 'Stoch'},
            'stoch_oversold':           {'grade': 'C', 'role': 'context',       'direction': 'bullish', 'value_type': 'indicator', 'value_label': 'Stoch'},
            'stoch_overbought':         {'grade': 'C', 'role': 'context',       'direction': 'bearish', 'value_type': 'indicator', 'value_label': 'Stoch'},
        }
    },

    'mfi': {
        'name': 'MFI',
        'description': 'Money Flow Index — volume-weighted RSI, measures buying/selling pressure',
        'category': 'momentum',
        'signals': {
            'mfi_divergence_bullish': {'grade': 'A', 'role': 'primary',       'direction': 'bullish', 'value_type': None,        'value_label': None},
            'mfi_divergence_bearish': {'grade': 'A', 'role': 'primary',       'direction': 'bearish', 'value_type': None,        'value_label': None},
            'mfi_oversold':           {'grade': 'C', 'role': 'context',       'direction': 'bullish', 'value_type': 'indicator', 'value_label': 'MFI'},
            'mfi_overbought':         {'grade': 'C', 'role': 'context',       'direction': 'bearish', 'value_type': 'indicator', 'value_label': 'MFI'},
        }
    },

    'cci': {
        'name': 'CCI',
        'description': 'Commodity Channel Index — measures price deviation from its average',
        'category': 'momentum',
        'signals': {
            'cci_oversold':   {'grade': 'C', 'role': 'context', 'direction': 'bullish', 'value_type': 'indicator', 'value_label': 'CCI'},
            'cci_overbought': {'grade': 'C', 'role': 'context', 'direction': 'bearish', 'value_type': 'indicator', 'value_label': 'CCI'},
        }
    },

    'williams_r': {
        'name': 'Williams %R',
        'description': 'Williams Percent Range — momentum indicator showing overbought/oversold levels',
        'category': 'momentum',
        'signals': {
            'williams_r_oversold':   {'grade': 'C', 'role': 'context', 'direction': 'bullish', 'value_type': 'indicator', 'value_label': '%R'},
            'williams_r_overbought': {'grade': 'C', 'role': 'context', 'direction': 'bearish', 'value_type': 'indicator', 'value_label': '%R'},
        }
    },

    'tsi': {
        'name': 'TSI',
        'description': 'True Strength Index — double-smoothed momentum indicator',
        'category': 'momentum',
        'signals': {
            'tsi_cross_bullish': {'grade': 'B', 'role': 'confirmation', 'direction': 'bullish', 'value_type': 'indicator', 'value_label': 'TSI'},
            'tsi_cross_bearish': {'grade': 'B', 'role': 'confirmation', 'direction': 'bearish', 'value_type': 'indicator', 'value_label': 'TSI'},
            'tsi_oversold':      {'grade': 'C', 'role': 'context',      'direction': 'bullish', 'value_type': 'indicator', 'value_label': 'TSI'},
            'tsi_overbought':    {'grade': 'C', 'role': 'context',      'direction': 'bearish', 'value_type': 'indicator', 'value_label': 'TSI'},
        }
    },

    'momentum_roc': {
        'name': 'Momentum / ROC',
        'description': 'Rate of Change and raw momentum — speed of price movement',
        'category': 'momentum',
        'signals': {
            'momentum_5':   {'grade': 'C', 'role': 'context', 'direction': 'bullish', 'value_type': None, 'value_label': None},
            'momentum_10':  {'grade': 'C', 'role': 'context', 'direction': 'bullish', 'value_type': None, 'value_label': None},
            'roc_bullish':  {'grade': 'C', 'role': 'context', 'direction': 'bullish', 'value_type': 'indicator', 'value_label': 'ROC'},
            'roc_bearish':  {'grade': 'C', 'role': 'context', 'direction': 'bearish', 'value_type': 'indicator', 'value_label': 'ROC'},
        }
    },

    # ── TREND ─────────────────────────────────────────────────────────────────

    'moving_averages': {
        'name': 'Moving Averages',
        'description': 'MA/EMA crossovers and ribbon — identify trend direction and momentum shifts',
        'category': 'trend',
        'signals': {
            'ma_cross_golden':    {'grade': 'B', 'role': 'primary',       'direction': 'bullish', 'value_type': None,        'value_label': None},
            'ma_cross_death':     {'grade': 'B', 'role': 'primary',       'direction': 'bearish', 'value_type': None,        'value_label': None},
            'ma_ribbon_bullish':  {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None,        'value_label': None},
            'ma_ribbon_bearish':  {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None,        'value_label': None},
            'triple_ema_bullish': {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None,        'value_label': None},
            'triple_ema_bearish': {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None,        'value_label': None},
            'ema_cross_fast':     {'grade': 'C', 'role': 'confirmation',  'direction': 'neutral', 'value_type': None,        'value_label': None},
            'price_above_ma20':   {'grade': 'C', 'role': 'context',       'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'MA20'},
            'price_below_ma20':   {'grade': 'C', 'role': 'context',       'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'MA20'},
        }
    },

    'supertrend': {
        'name': 'Supertrend',
        'description': 'Supertrend indicator — dynamic support/resistance that flips on trend change',
        'category': 'trend',
        'signals': {
            'supertrend_bullish': {'grade': 'B', 'role': 'primary', 'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'ST'},
            'supertrend_bearish': {'grade': 'B', 'role': 'primary', 'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'ST'},
        }
    },

    'ichimoku': {
        'name': 'Ichimoku',
        'description': 'Ichimoku Cloud — multi-component system showing trend, momentum and support/resistance',
        'category': 'trend',
        'signals': {
            'ichimoku_bullish': {'grade': 'B', 'role': 'primary', 'direction': 'bullish', 'value_type': None, 'value_label': None},
            'ichimoku_bearish': {'grade': 'B', 'role': 'primary', 'direction': 'bearish', 'value_type': None, 'value_label': None},
        }
    },

    'adx': {
        'name': 'ADX',
        'description': 'Average Directional Index — measures trend strength, not direction',
        'category': 'trend',
        'signals': {
            'adx_strong_trend': {'grade': 'B', 'role': 'context',       'direction': 'neutral', 'value_type': 'indicator', 'value_label': 'ADX'},
            'adx_weak_trend':   {'grade': 'C', 'role': 'context',       'direction': 'neutral', 'value_type': 'indicator', 'value_label': 'ADX'},
            'adx_reversal':     {'grade': 'B', 'role': 'confirmation',  'direction': 'neutral', 'value_type': 'indicator', 'value_label': 'ADX'},
        }
    },

    'parabolic_sar': {
        'name': 'Parabolic SAR',
        'description': 'Stop and Reverse — trailing stop that flips above/below price on trend change',
        'category': 'trend',
        'signals': {
            'parabolic_sar_flip_bullish': {'grade': 'B', 'role': 'confirmation', 'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'SAR'},
            'parabolic_sar_flip_bearish': {'grade': 'B', 'role': 'confirmation', 'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'SAR'},
        }
    },

    'aroon': {
        'name': 'Aroon',
        'description': 'Aroon indicator — measures time since last high/low to identify trend beginnings',
        'category': 'trend',
        'signals': {
            'aroon_bullish': {'grade': 'B', 'role': 'confirmation', 'direction': 'bullish', 'value_type': None, 'value_label': None},
            'aroon_bearish': {'grade': 'B', 'role': 'confirmation', 'direction': 'bearish', 'value_type': None, 'value_label': None},
        }
    },

    'elder_ray': {
        'name': 'Elder Ray',
        'description': 'Elder Ray Index — bull/bear power relative to EMA, measures buyer vs seller strength',
        'category': 'trend',
        'signals': {
            'elder_ray_bullish': {'grade': 'C', 'role': 'confirmation', 'direction': 'bullish', 'value_type': None, 'value_label': None},
            'elder_ray_bearish': {'grade': 'C', 'role': 'confirmation', 'direction': 'bearish', 'value_type': None, 'value_label': None},
        }
    },

    # ── VOLUME ────────────────────────────────────────────────────────────────

    'volume_flow': {
        'name': 'Volume Flow',
        'description': 'OBV, CMF, Accumulation/Distribution — tracks whether volume confirms or diverges from price',
        'category': 'volume',
        'signals': {
            'volume_divergence_bullish':         {'grade': 'A', 'role': 'primary',       'direction': 'bullish', 'value_type': None,        'value_label': None},
            'volume_divergence_bearish':         {'grade': 'A', 'role': 'primary',       'direction': 'bearish', 'value_type': None,        'value_label': None},
            'obv_bullish':                       {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None,        'value_label': None},
            'obv_bearish':                       {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None,        'value_label': None},
            'cmf_bullish':                       {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': 'indicator', 'value_label': 'CMF'},
            'cmf_bearish':                       {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': 'indicator', 'value_label': 'CMF'},
            'accumulation_distribution_bullish': {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None,        'value_label': None},
            'accumulation_distribution_bearish': {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None,        'value_label': None},
        }
    },

    'vwap': {
        'name': 'VWAP',
        'description': 'Volume Weighted Average Price — institutional benchmark; above = bullish, below = bearish',
        'category': 'volume',
        'signals': {
            'vwap_cross_above': {'grade': 'B', 'role': 'confirmation', 'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'VWAP'},
            'vwap_cross_below': {'grade': 'B', 'role': 'confirmation', 'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'VWAP'},
        }
    },

    'volume_activity': {
        'name': 'Volume Activity',
        'description': 'Volume spikes and climax events — unusual volume that signals conviction or exhaustion',
        'category': 'volume',
        'signals': {
            'volume_spike_bullish':  {'grade': 'B', 'role': 'confirmation', 'direction': 'bullish', 'value_type': None, 'value_label': None},
            'volume_spike_bearish':  {'grade': 'B', 'role': 'confirmation', 'direction': 'bearish', 'value_type': None, 'value_label': None},
            'volume_climax_bullish': {'grade': 'B', 'role': 'primary',      'direction': 'bullish', 'value_type': None, 'value_label': None},
            'volume_climax_bearish': {'grade': 'B', 'role': 'primary',      'direction': 'bearish', 'value_type': None, 'value_label': None},
        }
    },

    'volume_profile': {
        'name': 'Volume Profile',
        'description': 'Point of Control and Value Area — price levels where most volume was traded',
        'category': 'volume',
        'signals': {
            'poc_support':    {'grade': 'B', 'role': 'context', 'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'POC'},
            'poc_resistance': {'grade': 'B', 'role': 'context', 'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'POC'},
            'value_area_high':{'grade': 'B', 'role': 'context', 'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'VAH'},
            'value_area_low': {'grade': 'B', 'role': 'context', 'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'VAL'},
            'high_volume_node':{'grade': 'C', 'role': 'context', 'direction': 'neutral', 'value_type': 'price_level', 'value_label': 'HVN'},
            'low_volume_node': {'grade': 'C', 'role': 'context', 'direction': 'neutral', 'value_type': 'price_level', 'value_label': 'LVN'},
        }
    },

    # ── VOLATILITY ────────────────────────────────────────────────────────────

    'bollinger': {
        'name': 'Bollinger Bands',
        'description': 'Bollinger Bands — price envelope based on standard deviation; squeeze precedes breakout',
        'category': 'volatility',
        'signals': {
            'bollinger_squeeze':       {'grade': 'B', 'role': 'context',      'direction': 'neutral', 'value_type': None,        'value_label': None},
            'bollinger_breakout_up':   {'grade': 'B', 'role': 'primary',      'direction': 'bullish', 'value_type': 'price_level','value_label': 'BB Upper'},
            'bollinger_breakout_down': {'grade': 'B', 'role': 'primary',      'direction': 'bearish', 'value_type': 'price_level','value_label': 'BB Lower'},
            'bollinger_bounce_up':     {'grade': 'C', 'role': 'confirmation', 'direction': 'bullish', 'value_type': 'price_level','value_label': 'BB Lower'},
            'bollinger_bounce_down':   {'grade': 'C', 'role': 'confirmation', 'direction': 'bearish', 'value_type': 'price_level','value_label': 'BB Upper'},
        }
    },

    'keltner_donchian': {
        'name': 'Keltner / Donchian',
        'description': 'Keltner Channel and Donchian Channel breakouts — confirm momentum and trend continuation',
        'category': 'volatility',
        'signals': {
            'keltner_breakout_up':   {'grade': 'B', 'role': 'confirmation', 'direction': 'bullish', 'value_type': None,        'value_label': None},
            'keltner_breakout_down': {'grade': 'B', 'role': 'confirmation', 'direction': 'bearish', 'value_type': None,        'value_label': None},
            'donchian_breakout_up':  {'grade': 'B', 'role': 'confirmation', 'direction': 'bullish', 'value_type': 'price_level','value_label': 'Donchian'},
            'donchian_breakout_down':{'grade': 'B', 'role': 'confirmation', 'direction': 'bearish', 'value_type': 'price_level','value_label': 'Donchian'},
            'atr_expansion':         {'grade': 'C', 'role': 'context',      'direction': 'neutral', 'value_type': 'indicator',  'value_label': 'ATR'},
        }
    },

    # ── STRUCTURE ─────────────────────────────────────────────────────────────

    'market_structure': {
        'name': 'Market Structure',
        'description': 'Break of Structure and Change of Character — key shifts in directional bias',
        'category': 'structure',
        'signals': {
            'break_of_structure_bullish': {'grade': 'A', 'role': 'primary',       'direction': 'bullish', 'value_type': None, 'value_label': None},
            'break_of_structure_bearish': {'grade': 'A', 'role': 'primary',       'direction': 'bearish', 'value_type': None, 'value_label': None},
            'choch_bullish':              {'grade': 'A', 'role': 'primary',       'direction': 'bullish', 'value_type': None, 'value_label': None},
            'choch_bearish':              {'grade': 'A', 'role': 'primary',       'direction': 'bearish', 'value_type': None, 'value_label': None},
            'higher_high':                {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None, 'value_label': None},
            'lower_low':                  {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None, 'value_label': None},
        }
    },

    'support_resistance': {
        'name': 'Support / Resistance',
        'description': 'Key price levels — bounces and breaks at support/resistance zones',
        'category': 'structure',
        'signals': {
            'support_bounce':           {'grade': 'B', 'role': 'primary',       'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'Support'},
            'resistance_rejection':     {'grade': 'B', 'role': 'primary',       'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'Resistance'},
            'support_break':            {'grade': 'B', 'role': 'primary',       'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'Support'},
            'resistance_break':         {'grade': 'B', 'role': 'primary',       'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'Resistance'},
            'fibonacci_bounce_382':     {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'Fib 38.2%'},
            'fibonacci_bounce_618':     {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'Fib 61.8%'},
            'pivot_point_bullish':      {'grade': 'C', 'role': 'context',       'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'Pivot'},
            'pivot_point_bearish':      {'grade': 'C', 'role': 'context',       'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'Pivot'},
            'round_number_support':     {'grade': 'C', 'role': 'context',       'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'Round'},
            'round_number_resistance':  {'grade': 'C', 'role': 'context',       'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'Round'},
        }
    },

    # ── SMART MONEY (SMC) ─────────────────────────────────────────────────────

    'smart_money': {
        'name': 'Smart Money',
        'description': 'Order blocks, FVG, and liquidity sweeps — institutional price action concepts',
        'category': 'smc',
        'signals': {
            'liquidity_sweep_bullish': {'grade': 'A', 'role': 'primary',       'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'Sweep'},
            'liquidity_sweep_bearish': {'grade': 'A', 'role': 'primary',       'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'Sweep'},
            'order_block_bullish':     {'grade': 'A', 'role': 'primary',       'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'OB'},
            'order_block_bearish':     {'grade': 'A', 'role': 'primary',       'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'OB'},
            'fvg_bullish':             {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'FVG'},
            'fvg_bearish':             {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'FVG'},
            'discount_zone':           {'grade': 'B', 'role': 'context',       'direction': 'bullish', 'value_type': None,          'value_label': None},
            'premium_zone':            {'grade': 'B', 'role': 'context',       'direction': 'bearish', 'value_type': None,          'value_label': None},
        }
    },

    # ── PRICE PATTERNS ────────────────────────────────────────────────────────

    'candlestick_reversal': {
        'name': 'Candlestick Patterns',
        'description': 'Single and multi-candle formations — short-term reversal and continuation signals',
        'category': 'pattern',
        'signals': {
            'engulfing_bullish':    {'grade': 'B', 'role': 'primary',       'direction': 'bullish', 'value_type': None, 'value_label': None},
            'engulfing_bearish':    {'grade': 'B', 'role': 'primary',       'direction': 'bearish', 'value_type': None, 'value_label': None},
            'morning_star':         {'grade': 'B', 'role': 'primary',       'direction': 'bullish', 'value_type': None, 'value_label': None},
            'evening_star':         {'grade': 'B', 'role': 'primary',       'direction': 'bearish', 'value_type': None, 'value_label': None},
            'three_white_soldiers': {'grade': 'B', 'role': 'primary',       'direction': 'bullish', 'value_type': None, 'value_label': None},
            'three_black_crows':    {'grade': 'B', 'role': 'primary',       'direction': 'bearish', 'value_type': None, 'value_label': None},
            'hammer':               {'grade': 'C', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None, 'value_label': None},
            'inverted_hammer':      {'grade': 'C', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None, 'value_label': None},
            'hanging_man':          {'grade': 'C', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None, 'value_label': None},
            'shooting_star':        {'grade': 'C', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None, 'value_label': None},
            'piercing_pattern':     {'grade': 'C', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None, 'value_label': None},
            'dark_cloud_cover':     {'grade': 'C', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None, 'value_label': None},
            'harami_bullish':       {'grade': 'C', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None, 'value_label': None},
            'harami_bearish':       {'grade': 'C', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None, 'value_label': None},
            'tweezer_bottom':       {'grade': 'C', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None, 'value_label': None},
            'tweezer_top':          {'grade': 'C', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None, 'value_label': None},
            'marubozu_bullish':     {'grade': 'C', 'role': 'confirmation',  'direction': 'bullish', 'value_type': None, 'value_label': None},
            'marubozu_bearish':     {'grade': 'C', 'role': 'confirmation',  'direction': 'bearish', 'value_type': None, 'value_label': None},
            'doji_reversal':        {'grade': 'C', 'role': 'context',       'direction': 'neutral', 'value_type': None, 'value_label': None},
            'gap_up':               {'grade': 'B', 'role': 'confirmation',  'direction': 'bullish', 'value_type': 'price_level', 'value_label': 'Gap'},
            'gap_down':             {'grade': 'B', 'role': 'confirmation',  'direction': 'bearish', 'value_type': 'price_level', 'value_label': 'Gap'},
        }
    },
}

# ── Flat lookup: signal_name → {family, ...meta} ──────────────────────────────

SIGNAL_LOOKUP: dict = {}
for _family_key, _family in SIGNAL_FAMILIES.items():
    for _signal_name, _meta in _family['signals'].items():
        SIGNAL_LOOKUP[_signal_name] = {
            'family':       _family_key,
            'family_name':  _family['name'],
            'category':     _family['category'],
            'description':  _family['description'],
            'grade':        _meta['grade'],
            'role':         _meta['role'],
            'direction':    _meta['direction'],
            'value_type':   _meta.get('value_type'),
            'value_label':  _meta.get('value_label'),
        }

# ── Timeframe horizon groups ──────────────────────────────────────────────────

HORIZONS = {
    'short': {
        'label':       'Short-term',
        'timeframes':  ['5m', '15m', '30m'],
        'chart_tf':    '15m',
        'description': 'Scalping / intraday',
        'tf_seconds':  {'5m': 300, '15m': 900, '30m': 1800},
    },
    'mid': {
        'label':       'Mid-term',
        'timeframes':  ['1h', '2h', '4h'],
        'chart_tf':    '1h',
        'description': 'Intraday swing / day trading',
        'tf_seconds':  {'1h': 3600, '2h': 7200, '4h': 14400},
    },
    'long': {
        'label':       'Long-term',
        'timeframes':  ['6h', '12h', '1d', '1w'],
        'chart_tf':    '4h',
        'description': 'Swing / position trading',
        'tf_seconds':  {'6h': 21600, '12h': 43200, '1d': 86400, '1w': 604800},
    },
}

# ── Grade weights for bias scoring ───────────────────────────────────────────

GRADE_WEIGHTS = {
    'A': 3,
    'B': 1.5,
    'C': 0.5,
}

CATEGORY_ORDER = ['momentum', 'trend', 'volume', 'volatility', 'structure', 'smc', 'pattern']
