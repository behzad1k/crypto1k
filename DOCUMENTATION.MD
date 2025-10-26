I applied these indicators on the top 8 stable coins historical data of the past 5 year to check their
accuracy:
[ 'MACD_crossover', 'MACD_position', 'Momentum_5', 'Momentum_10', 'RSI_oversold', 'RSI_overbought', 'RSI_midline', 'STOCH_D_oversold', 'STOCH_D_overbought', 'STOCH_D_midline', 'WILLR_overbought', 'WILLR_oversold', 'WILLR_midline' ]
the results were disappointing as the maximum accuracy was 58 percent which is only slightly better than random. then I
tested their
combinations:
[ 'MACD_RSI_bullish', 'MACD_RSI_bearish', 'MACD_MOM_bullish', 'MACD_MOM_bearish', 'RSI_STOCH_oversold', 'RSI_STOCH_overbought', 'RSI_WILLR_oversold', 'RSI_WILLR_overbought', 'MOM_5_10_bullish', 'MOM_5_10_bearish', 'STOCH_WILLR_oversold', 'STOCH_WILLR_overbought' ]
and the results went up to 65% accuracy. then I added timeframes into the mix, I applied the combinations on 8 different
timeframes per each coin to extract which combinations in the same timeframe lead to better results. then I extracted
which combinations had the most accuracy when appeared in the same timeframe and the results were significantly
improved up to 70% accuracy. at last I tested the combination of those combinations cross different time frames and the results got even
much much better up ta 90% accuracy. finally I wrote a script to apply all those cross timeframe combinations per each coin and after 60
hours of the last row of data (500 rows) I had the top 2700 patterns (more than 70% accuracy) then I made a script to
alert me if more than 800 of those patterns was detected in a coin with more than 80% overall confidence. so far it has
shown a promising accuracy.
I didnt only test on historical data. I had a script applying all the indicators and their combinations cross timeframes
in real time by fetching their data from binance api and storing the signals in the database until I had at least 500
rows of Buy signals. after that I had another script that fetched the data of the past 60 hours and checked which one of
the signals(combinations of indicators cross-timeframes) had better accuracy and out of how many signals that accuracy
was driven. my sell signals were awful so I removed them completely for now and am only working on buy signals. the
signal counts of those 2700 patterns vary from 20 - 140 and all have more than at least 70% accuracy. I logged on the
top 100 coins not just ETH and BTC.
I have been running a script that applies those top 2700 patterns on top 100 coins in real time in an infinite loop and
saving the result in my database pattern signals table: TABLE pattern_signals (
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
, scalp_validation TEXT)
some sample rows:
1 BNB BUY 0.752688172043011 6    [1h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10 0.764705882352941    [
{
"pattern": "[1h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[1h] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[2h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[2h] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[30m] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[30m] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[4h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.75,
"success": 12,
"total": 16,
"matched_components": [
"[4h] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[12h] Momentum_10 + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.75,
"success": 9,
"total": 12,
"matched_components": [
"[12h] Momentum_10",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[6h] Momentum_5 + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7142857142857143,
"success": 10,
"total": 14,
"matched_components": [
"[6h] Momentum_5",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
}
]    1029.581 998.69357 2025-10-02 00:07:15
2 BNB BUY 0.752688172043011 6    [1h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10 0.764705882352941    [
{
"pattern": "[1h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[1h] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[2h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[2h] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[30m] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[30m] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[4h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.75,
"success": 12,
"total": 16,
"matched_components": [
"[4h] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[12h] Momentum_10 + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.75,
"success": 9,
"total": 12,
"matched_components": [
"[12h] Momentum_10",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[6h] Momentum_5 + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7142857142857143,
"success": 10,
"total": 14,
"matched_components": [
"[6h] Momentum_5",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
}
]    1029.687 998.79639 2025-10-02 00:07:32
3 BNB BUY 0.752688172043011 6    [1h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10 0.764705882352941    [
{
"pattern": "[1h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[1h] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[2h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[2h] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[30m] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[30m] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[4h] MACD_position + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.75,
"success": 12,
"total": 16,
"matched_components": [
"[4h] MACD_position",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[12h] Momentum_10 + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.75,
"success": 9,
"total": 12,
"matched_components": [
"[12h] Momentum_10",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
},
{
"pattern": "[6h] Momentum_5 + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7142857142857143,
"success": 10,
"total": 14,
"matched_components": [
"[6h] Momentum_5",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
]
}
]    1029.687 998.79639 2025-10-02 00:07:54
4 LAYER BUY 0.8 2    [12h] Momentum_5 + [2h] MACD_position + [30m] WILLR_oversold 0.846153846153846    [
{
"pattern": "[12h] Momentum_5 + [2h] MACD_position + [30m] WILLR_oversold",
"accuracy": 0.8461538461538461,
"success": 11,
"total": 13,
"matched_components": [
"[12h] Momentum_5",
"[2h] MACD_position",
"[30m] WILLR_oversold"
],
"parsed": {
"timeframe_indicators": {
"12h": [
"Momentum_5"
],
"2h": [
"MACD_position"
],
"30m": [
"WILLR_oversold"
]
},
"required_count": 3
}
},
{
"pattern": "[12h] Momentum_5 + [8h] MOM_5_10_bullish + [8h] Momentum_10",
"accuracy": 0.7647058823529411,
"success": 13,
"total": 17,
"matched_components": [
"[12h] Momentum_5",
"[8h] MOM_5_10_bullish",
"[8h] Momentum_10"
],
"parsed": {
"timeframe_indicators": {
"12h": [
"Momentum_5"
],
"8h": [
"MOM_5_10_bullish",
"Momentum_10"
]
},
"required_count": 3
}
}
]    0.01201 0.0116497 2025-10-02 00:57:41

which i have now around 14,000 rows of. the real-time results are fascinating. this is phase one of my project. process
this and I will tell you my phase two then we will discuss my phase 3 in my phase two I implemented 82 indicators and
signs and ran on some coins occasionally (especially on the ones my phase 1 signaled on) on live data: my live_signals:
CREATE TABLE live_signals (
id INTEGER PRIMARY KEY AUTOINCREMENT,
symbol TEXT NOT NULL,
timeframe TEXT NOT NULL,
signal_name TEXT NOT NULL,
signal_type TEXT NOT NULL,
confidence INTEGER NOT NULL,
strength TEXT,
signal_value REAL,
price REAL NOT NULL,
timestamp TIMESTAMP NOT NULL,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
UNIQUE(symbol, timeframe, signal_name, timestamp)
)
some sample rows:
INSERT INTO "main"."live_signals" ("id", "symbol", "timeframe", "signal_name", "signal_type", "confidence", "
strength", "signal_value", "price", "timestamp", "created_at") VALUES ('1', 'RDNT', '1m', 'rsi_oversold', 'BUY', '68', '
MODERATE', '0.0', '0.0214', '2025-10-07 07:39:00', '2025-10-07 07:43:37');
INSERT INTO "main"."live_signals" ("id", "symbol", "timeframe", "signal_name", "signal_type", "confidence", "
strength", "signal_value", "price", "timestamp", "created_at") VALUES ('2', 'RDNT', '1m', 'macd_cross_bearish', '
SELL', '79', 'STRONG', NULL, '0.0214', '2025-10-07 07:39:00', '2025-10-07 07:43:37');
INSERT INTO "main"."live_signals" ("id", "symbol", "timeframe", "signal_name", "signal_type", "confidence", "
strength", "signal_value", "price", "timestamp", "created_at") VALUES ('3', 'RDNT', '1m', '
macd_histogram_reversal_bearish', 'SELL', '75', 'MODERATE', NULL, '0.0214', '2025-10-07 07:39:00', '2025-10-07 07:43:
37');
INSERT INTO "main"."live_signals" ("id", "symbol", "timeframe", "signal_name", "signal_type", "confidence", "
strength", "signal_value", "price", "timestamp", "created_at") VALUES ('4', 'RDNT', '1m', 'cci_oversold', 'BUY', '67', '
MODERATE', '-283.235277149917', '0.0214', '2025-10-07 07:39:00', '2025-10-07 07:43:37');

I have 33394 rows;
each signal had an initial confidence rate based on my theoretical researches.

I also have a position management system that ran the indicators on my actual positions containing 448,437 rows.
I implemented a fact-checker system that tested the signals after enough time had passed to test their actual accuracy
and adjusted their confidence based on those fact-checks.
CREATE TABLE signal_fact_checks (
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
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )

some sample rows:
INSERT INTO "main"."signal_fact_checks" ("id", "signal_name", "timeframe", "detected_at", "price_at_detection", "
actual_move", "predicted_correctly", "price_change_pct", "checked_at", "candles_elapsed", "created_at") VALUES ('1', '
adx_strong_trend', '1h', '2025-10-07 12:51:26.214284', '0.1379', 'DOWN', '0', '-2.10297316896301', '2025-10-09 18:17:
26.835104', '5', '2025-10-09 18:17:26');
INSERT INTO "main"."signal_fact_checks" ("id", "signal_name", "timeframe", "detected_at", "price_at_detection", "
actual_move", "predicted_correctly", "price_change_pct", "checked_at", "candles_elapsed", "created_at") VALUES ('2', '
bollinger_breakout_up', '1h', '2025-10-07 12:51:26.211345', '0.1379', 'DOWN', '0', '-2.10297316896301', '2025-10-09 18:
17:27.189896', '5', '2025-10-09 18:17:27');
INSERT INTO "main"."signal_fact_checks" ("id", "signal_name", "timeframe", "detected_at", "price_at_detection", "
actual_move", "predicted_correctly", "price_change_pct", "checked_at", "candles_elapsed", "created_at") VALUES ('3', '
obv_bullish', '1h', '2025-10-07 12:51:26.208726', '0.1379', 'DOWN', '0', '-2.10297316896301', '2025-10-09 18:17:
27.525913', '5', '2025-10-09 18:17:27');
INSERT INTO "main"."signal_fact_checks" ("id", "signal_name", "timeframe", "detected_at", "price_at_detection", "
actual_move", "predicted_correctly", "price_change_pct", "checked_at", "candles_elapsed", "created_at") VALUES ('4', '
volume_spike_bullish', '1h', '2025-10-07 12:51:26.206199', '0.1379', 'DOWN', '0', '-2.10297316896301', '2025-10-09 18:
17:27.926476', '5', '2025-10-09 18:17:27');
INSERT INTO "main"."signal_fact_checks" ("id", "signal_name", "timeframe", "detected_at", "price_at_detection", "
actual_move", "predicted_correctly", "price_change_pct", "checked_at", "candles_elapsed", "created_at") VALUES ('5', '
cci_overbought', '1h', '2025-10-07 12:51:26.203206', '0.1379', 'DOWN', '1', '-2.10297316896301', '2025-10-09 18:17:
28.403910', '5', '2025-10-09 18:17:28');
INSERT INTO "main"."signal_fact_checks" ("id", "signal_name", "timeframe", "detected_at", "price_at_detection", "
actual_move", "predicted_correctly", "price_change_pct", "checked_at", "candles_elapsed", "created_at") VALUES ('6', '
stoch_cross_bearish', '1h', '2025-10-07 12:51:26.199429', '0.1379', 'DOWN', '1', '-2.10297316896301', '2025-10-09 18:17:
28.708310', '5', '2025-10-09 18:17:28');
this table has about 173,398 rows

and finally:
CREATE TABLE signal_fact_checks (
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
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )

INSERT INTO "main"."signal_confidence_adjustments" ("id", "signal_name", "timeframe", "original_confidence", "
adjusted_confidence", "accuracy_rate", "sample_size", "last_updated") VALUES ('1', 'adx_strong_trend', '15m', '79', '
60', '17.9856115107914', '1668', '2025-10-09 19:41:14.055072');
INSERT INTO "main"."signal_confidence_adjustments" ("id", "signal_name", "timeframe", "original_confidence", "
adjusted_confidence", "accuracy_rate", "sample_size", "last_updated") VALUES ('2', 'adx_strong_trend', '1h', '79', '
62', '23.1191652937946', '1821', '2025-10-09 19:41:14.097211');
INSERT INTO "main"."signal_confidence_adjustments" ("id", "signal_name", "timeframe", "original_confidence", "
adjusted_confidence", "accuracy_rate", "sample_size", "last_updated") VALUES ('3', 'adx_strong_trend', '1m', '79', '
66', '37.5514403292181', '972', '2025-10-09 19:41:14.168681');
INSERT INTO "main"."signal_confidence_adjustments" ("id", "signal_name", "timeframe", "original_confidence", "
adjusted_confidence", "accuracy_rate", "sample_size", "last_updated") VALUES ('4', 'adx_strong_trend', '5m', '79', '
67', '41.4074074074074', '1350', '2025-10-09 19:41:14.219127');
INSERT INTO "main"."signal_confidence_adjustments" ("id", "signal_name", "timeframe", "original_confidence", "
adjusted_confidence", "accuracy_rate", "sample_size", "last_updated") VALUES ('5', 'adx_weak_trend', '15m', '70', '
49', '0.0', '155', '2025-10-09 19:41:14.250517');
INSERT INTO "main"."signal_confidence_adjustments" ("id", "signal_name", "timeframe", "original_confidence", "
adjusted_confidence", "accuracy_rate", "sample_size", "last_updated") VALUES ('6', 'adx_weak_trend', '1m', '70', '49', '
0.0', '352', '2025-10-09 19:41:14.274732');

this table has 367 rows.
not all the system's generated signals have been fact-checked yet.
for my phase three I want to build a paper trading system
that applies the live scalping signals on the signals my phase 1 generates, then based on them and some other pro scalping
techniques, for example I myself tend to check if the current price of the coin is above/under the average price of that
coin in the past year so I dont buy it at peaks. the system then imagines we have bought that coin on the signaled price
and sells it at either a constant stop-profit percentage or a dynamic stop-profit percentage based on the signal's
confidence and timeframe. this way I can test both my systems, find the flaws and set up stop_profit/stop_loss systems.