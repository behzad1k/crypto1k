# Signal Effectiveness Analysis — 2026-07-26

> **Update after implementation.** Two conclusions below were revised when tested properly before being coded. Both corrections are in §3 and §5; the config now reflects the corrected versions, not the originals.
>
> 1. **The BTC entry gate is not implementable.** §5 recommends suppressing alerts when BTC is falling, based on a 21% vs 49% win-rate split. That split uses BTC's move *during the outcome window* — information from the alert's future. What is visible at fire time (regime score, BTC 1h/24h change) does not survive a train/test split: the worst regime bucket flips between the two halves of the month. The gate stays off.
> 2. **Smart-wallet auto-qualification is anti-predictive, not merely loose.** §3 recommends re-qualifying wallets on forward returns. That was tested — qualify on trades before a cutoff, measure the buys after it — at six cutoffs, and auto-qualified wallets underperformed the *all-wallet* baseline at 1h every single time. Tightening the bar made it worse. Auto-qualification is now off entirely rather than retuned.
>
> Replaying the implemented stack over this same month: 167 alerts survive of 598, 1h return goes +0.34% → +0.44% (win 48.0% → 56.3%), 4h turns positive (-1.27% → +0.14%), 24h -5.54% → -0.45%, and the stop-loss hit rate falls 52.3% → 21.0%.

**Data:** `signals_v2.db` snapshot from sogol (163 MB, consistent `.backup` including WAL).
**Sample:** 810 alerts with complete 24h outcomes, 2026-06-24 → 2026-07-25 (598 bullish, 148 bearish, 64 neutral). Smart-money tracker: 96,663 completed wallet-buy outcomes.

> ⚠️ Sample concentration: CASHCAT (131) + DIH (60) are ~32% of bullish alerts. Every conclusion below was re-checked excluding them; where that changes the picture it is called out.

---

## 1. The system's edge is short-lived — it finds pumps, not 24h winners

Bullish alerts, forward returns from alert price:

| Horizon | Mean | Median | Win rate |
|---|---|---|---|
| 15m | +0.55% | +0.03% | 51.8% |
| 1h | +0.34% | −0.08% | 48.0% |
| 4h | −1.27% | −1.14% | 39.7% |
| 24h | −5.54% | −3.20% | 37.6% |
| **Max upside within 24h** | **+48.6%** | **+7.5%** | 98.7% hit some green |
| Max drawdown within 24h | −16.1% | −10.7% | — |

- **59% of alerts touch +5% within 24h; 39% touch +10%; 20% touch +20%** — but holding to 24h loses money on average.
- Median time-to-peak is **5.6h**; 43% of alerts peak within 4h.
- 54% of alerts draw down −10% at some point; 30% draw down −20%.

**Implication:** the alerts are usable only with an exit plan. A take-profit near +10% with a stop near −10% fits the observed distribution far better than holding 24h. Buy-and-hold-24h on these alerts is a losing strategy in this period.

## 2. Most / least effective signals

Per-signal 24h performance on bullish alerts (signal present in alert payload), full sample and after excluding CASHCAT+DIH:

| Signal | 24h mean (all) | 24h mean (excl. C+D) | 1h win (all) | Verdict |
|---|---|---|---|---|
| exit_liquidity_risk* | +1.31% | +1.31% (n=20) | 50% | Small n, but consistently least-bad |
| whale_net_flow | −7.53% | −1.48% | 45% | Neutral-ish once C+D removed |
| whale_print | −7.74% | −1.82% | 47% | Neutral-ish |
| buy_pressure | −5.59% | −2.28% | 50% | Neutral |
| volume_surge | −3.90% | −3.42% | **56%** | **Best 1h signal** (+3.05% mean at 1h) |
| volume_acceleration | −3.61% | −2.60% | 49% | Fires on 70% of alerts — little discrimination |
| momentum_alignment | −5.59% | −3.17% | 48% | Slightly negative |
| price_momentum | −7.62% | −4.12% | 46% | **Consistently negative — chasing** |
| price_momentum_5m | −7.63% | −4.28% | 47% | **Consistently negative — chasing** |
| smart_wallet_buy | **−10.42%** | −2.20% (n=24) | 43% | **Worst in sample; see §3** |

\* `exit_liquidity_risk` is a *warning* signal — its positive association likely reflects that it fires on higher-liquidity, less-manipulated pairs.

**Takeaways**
- **`volume_surge` is the only signal with a real positive short-horizon edge** (56% 1h win, +3.05% 1h mean). It decays by 24h like everything else.
- **`price_momentum` / `price_momentum_5m` are the most reliably harmful** — they hold up as negative in every cut (full sample, excl. C+D, and symbol-level mean-of-means −7.8% / −8.6%). Alerts driven mainly by "price already moved" are late entries.
- Worst signal *pairs* are all `momentum × smart_wallet_buy` combos (−11% to −13% mean 24h).

## 3. The smart-money path is the biggest problem

- `alert_path = smart` alerts: **−10.4% mean 24h, 32% win** vs standard −5.4%, fast −0.1%, none −0.5%.
- The smart path admits low scores (FAIR ≥5.5) that other paths would reject — and *both* halves underperform: smart alerts with score ≥6.5 → −8.7%, score <6.5 → −11.4%. So it's not just the lower threshold; the trigger itself is weak.
- Aggregate wallet-tracker stats agree: across **91,804 completed tracked-wallet buys**, price is +1.45% (51% win) 1h later but **−2.36% (47% win) 24h later**. Even filtering to buys ≥$2k: −0.61% at 24h. The tracked "smart" wallets are scalpers at best — their buys are not a 24h hold signal.
- Concentration caveat: most smart-path pain came from DIH/CASHCAT repeat alerts, but even excluding them the path is negative (−2.0%, 36% win) and symbol-level mean-of-means is −11% across 8 symbols. No cut shows it positive.

**Recommendation:** either demote `smart_wallet_buy` to a confirmation-only input (no score bonus, no dedicated alert path), or re-qualify tracked wallets on realized 24h forward returns rather than past-trade PnL. Note `smart_wallets` table is empty (auto-tracked only) and 0 wallet *sell* outcomes are recorded — you can't currently distinguish wallets that flip in minutes, which the 1h-vs-24h gap suggests they do.

## 4. Scoring / labels are miscalibrated

- Full sample: STRONG −5.98% (34% win) vs GOOD −4.34% (42% win). Excluding C+D: STRONG −4.42% (33%) vs GOOD **−1.12% (46%)**. **STRONG underperforms GOOD everywhere.**
- Score buckets are non-monotonic: 7–8 is the best bucket (−2.8%, 44% win); 8+ is *worse* (−6.0%, 34% win). Very high scores mostly mean "everything momentum-related fired at once" — i.e., the pump is already mature.

## 5. Context filters that actually work

These conditioning variables separated outcomes better than most signals:

| Filter | 24h mean | Win |
|---|---|---|
| Liquidity >$500k (n=382) | −2.74% | 43.5% |
| Liquidity $150–500k (n=99) | −8.73% | 31.3% |
| Liquidity <$150k (n=91) | ~−14% | ~20% |
| BTC +1%+ during window (n=154) | −3.93% | 49.4% |
| BTC flat (n=256) | −4.24% | 39.5% |
| BTC −1%+ during window (n=132) | −8.87% | **21.2%** |
| Chain = ethereum (n=32) | **+4.59%** | **56.2%** |
| Chain = base (n=112) | −1.00% | 39.3% |
| Chain = solana (n=227) | −3.25% | 37.4% |
| Chain = robinhood (n=197) | −12.55% | 33.0% |

- **Liquidity is the strongest single conditioner in the dataset.** Sub-$150k-liquidity alerts were near-uniformly bad.
- **BTC tailwind more than doubles the win rate** (49% vs 21%). The btc_regime_score itself, as stored, did *not* separate outcomes (bull-regime n=19 did worst) — realized BTC movement during the window is what matters; the regime score at alert time isn't capturing it.
- Robinhood-chain alerts (mostly DIH) were the single worst pocket.

## 6. Bearish alerts don't work

148 bearish alerts: price *rose* on average afterward (inverted 24h mean −1.20%, win 42.5%). No horizon shows a profitable short edge. These are currently noise.

## 7. Symbol concentration / repeat alerting

- CASHCAT alerted 131×, DIH 60×, CARDS 39×, KINS 30×… Repeat alerts on the same symbol trend worse (DIH: −19.9% mean, 22% win across 58 alerts).
- UPEG (+5.6% mean, 59% win, n=27) and AVICI (61% win, n=18) show the scanner *can* find good coins — but the alert volume goes to the manipulated ones.
- Consider a per-symbol cooldown or decaying score for repeat alerts: the first alert on a symbol is systematically better than the 20th.

## 8. Trend over the month

Weekly 24h mean: wk26 −0.2% → wk27 −2.1% → wk28 −8.0% → wk29 −7.0% → wk30 −7.9%. Deterioration coincides with the smart path ramping up and DIH/CASHCAT alert spam — supports the cooldown + smart-path fixes.

---

## Action list — all implemented 2026-07-26

| # | Action | Status |
|---|---|---|
| 1 | Treat alerts as short trades: TP +10%, stop −10%, time-stop 8h, shipped in the alert text | `EXIT_PLAN` in scanner_config |
| 2 | Smart-money path disabled; score weight cut 1.5 → 0.25; auto-qualification off; wallet sells now scored | scanner_config + smart_money.py + db.py |
| 3 | Liquidity floor 15k → 150k | `FILTERS['min_liquidity_usd']` |
| 4 | BTC gate — **not implemented**, no cutoff survives a train/test split | documented in `BTC_IMPACT['gate']` |
| 5 | Momentum weight 3.0 → 1.5 and inverted at the top end; volume 4.0 → 5.0 with a new 10×+ tier; liquidity depth 1.0 → 2.5 | `SCORING` + scanner.py |
| 6 | Per-symbol cooldown 2h → 6h, plus a hard 3-alerts-per-24h cap | `ALERTING` + db.alert_count_within |
| 7 | Labels recalibrated (STRONG 8.0 → 7.0, GOOD 6.0 → 5.5) for the new weight distribution | `SCORING` |
| 8 | Bearish alerts fixed at the source — they were leaking through a re-count of direction after smart-money enrichment, despite `require_bullish` | scanner._revalidate_after_enrichment |

Additional changes made while implementing:

- **Over-extension gate** (new): blocks alerts above +15%/1h, +6%/5m or +50%/6h, with a softer score penalty from +8%/1h. This was the most monotone relationship in the data and had no representation in the old model.
- **Wash-trade threshold 75 → 8**: no alert in a month ever exceeded turnover 20, so the old value was dead code.
- **In-app per-signal effectiveness reporting**: `evaluation.generate_report()` now returns `by_signal`, `by_liquidity` and `exit_calibration`, and flags any signal whose presence correlates with worse returns. This analysis was done by hand against a database copy; it is now repeatable from the app.
- **`tp_hit_at` / `sl_hit_at` columns**: the stored outcomes only kept the 24h high and low, so it was impossible to tell whether the target filled *before* the stop. These record the first crossing of each level so the next review can set TP/SL on a real backtest instead of unordered touch rates.

---

*Generated from `signals_v2_latest.db` (snapshot of sogol:/opt/crypto1k/signals_v2.db taken 2026-07-26). Note: `crypto_signals.db` on the server is an empty schema-only file untouched since Jun 24 — the live data lives in `signals_v2.db`.*
