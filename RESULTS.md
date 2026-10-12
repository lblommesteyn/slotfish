# Results: baseline study, 2026-10-12

Every number here comes from `python scripts/run_baseline.py` on the capture
described below. Full tables are in [RESULTS_TABLES.md](RESULTS_TABLES.md) and
the raw JSON is in `results/`.

## TL;DR

**No strategy is profitable after realistic costs.** Microstructure features
predict executable returns with real, out-of-sample rank skill (2-minute
test IC 0.21, 15-minute-block t = 4.2). That skill is almost entirely about
**which tokens will lose most**: the top-scored decile still loses 2-5% net
per trade at every horizon. Wallet-skill and coordination features **add
nothing** beyond momentum and order flow. Survival (graduation, collapse,
crash) is very predictable from early activity, but there is no way to
short the curve, so a crash forecast is not a trade.

The answer to the core question, on this data: microstructure signals carry
information about *executable* memecoin returns, but not enough to overcome
fees, impact and adverse selection on the long side. Wallet-network signals
did not help.

## Data

| | |
|---|---|
| Window | slots 455,530,000 to 455,660,000 (130k slots, **7.87 h** contiguous, 2026-10-11 05:08 to 13:00 UTC) |
| Source | free publicnode RPC `getBlock`, decoded in-house ([docs/DATA.md](docs/DATA.md)) |
| Completeness | 25 skipped slots (leader skips), **10 missing** (pruned by the node mid-fetch; listed in `manifest.jsonl`) |
| Events | 6.36M: 947k curve trades, 3.6M PumpSwap swaps, 1.81M failed attempts, 10,621 launches, 329 graduations |
| Launches | 10,621 total: 1,721 non-SOL-quoted, 3,013 Mayhem mode, **6,027 in the tradable cohort** (all included, incl. 817 that never traded again) |
| Cohort graduation rate | 201 / 6,027 = 3.3% |
| Slot time | 0.218 s |
| Store | 391 MB zstd parquet |

The capture was planned as 24 h. It was cut to 7.9 h because the free node
prunes history at about the rate we can download it (~5 slots/s each) and
the margin was shrinking ~1k slots/hour. A day-scale study needs either a
paid archive or several forward captures.

## Splits and labels

Chronological 60/20/20 split by slot with a **630 s purge** (longest horizon
plus slack) between splits: train 4.5 h, validation 1.4 h, test 1.6 h.
Decision points are a cohort token's last trade in a slot, so 293k rows in
total. Labels are **net returns of a full simulated round trip** through the
execution model: 0.1 SOL, 2-slot latency, end-of-landing-slot fills, double
price impact, swap fees from the events, 105k lamports per transaction,
slippage limit 20%. They are not mid-price moves.

Base rates (train): the mean net return of buying at a random decision point
is **-4.8% (30 s), -8.5% (2 m), -18.2% (10 m)**, and P(net > 0) is 31%, 30%
and 19%.

## Phase 3/4: predictive models (test period, 7 blocks of 15 min)

| horizon | model | test IC | block t | AUC(sign) | top-10% net ret [95% CI] | bottom decile |
|---|---|---|---|---|---|---|
| 30 s | ridge | 0.089 | 1.7 | 0.50 | -2.4% [-4.6, -0.2] | |
| 30 s | LightGBM | 0.135 | 3.7 | 0.48 | -3.4% [-6.1, -0.9] | -13.2% |
| 2 m | ridge | 0.101 | 2.5 | 0.50 | -5.2% [-13.3, 2.1] | |
| 2 m | LightGBM | 0.211 | 4.2 | 0.51 | -4.7% [-14.7, 1.4] | -22.1% |
| 10 m | ridge | 0.374 | 8.4 | 0.56 | +4.0% [-5.8, 12.7] | |
| 10 m | LightGBM | 0.431 | 14.3 | 0.52 | -2.0% [-8.3, 2.2] | -34.2% |

CIs resample whole tokens. How to read this table:

* **IC is high but AUC of the sign is ~0.5.** The models separate -100%
  from -10% very well and cannot find the minority of winners. The deciles
  are monotone at the bottom and flat at the top (see the calibration
  tables).
* **The only positive top-decile point estimate (ridge, 10 m, +4.0%) has a
  CI spanning zero.** At the threshold chosen on validation it is -1.4% on
  test.
* The 10 m IC is high partly because the curve has a price floor: tokens
  whose early buyers are exiting drain toward the floor almost
  deterministically. That is real, but it is not long-side alpha.

### Ablations and the wallet question

Change in test IC when feature groups are removed (LightGBM):

| feature set | 30 s | 2 m | 10 m |
|---|---|---|---|
| all | 0.135 | 0.211 | 0.431 |
| no wallet (skill + coordination) | 0.130 | 0.216 | 0.423 |
| momentum only | 0.124 | 0.152 | 0.282 |
| momentum + wallet | 0.101 | 0.134 | 0.294 |
| no order flow | 0.114 | 0.223 | 0.404 |
| no token state | 0.112 | 0.123 | 0.347 |

Paired over 15-minute blocks, **momentum+wallet vs momentum-only has an IC
difference of -0.030 (t = -1.9), -0.022 (t = -2.8) and +0.003 (t = 0.1)**,
and all vs no-wallet has -0.005, -0.002 and +0.015 (all |t| < 1.1). Wallet
features get high LightGBM *gain* (`skill_wavg_75` is the top split feature
at 30 s and 2 m) but that importance does not survive out of sample. **We
find no evidence that Bayesian wallet skill or co-buying coordination
predicts returns beyond momentum.** Caveat: skill is estimated from at most
~8 h of history, and wallets with real long-run skill may need far longer to
identify.

Token state (age, market cap, curve progress, creator holdings, sniper
count) carries the most information at 2 m and 10 m.

## Phase 4: event-driven backtests (test window, portfolio limit 20)

Net of everything. 0.1 SOL per trade, 2-slot latency, p_fail 5%.

| hold | strategy | trades | net SOL | mean ret/trade | t | max DD (SOL) | turnover (SOL) |
|---|---|---|---|---|---|---|---|
| 2 m | no trade | 0 | 0 | 0 | | 0 | 0 |
| 2 m | random (rate-matched) | 473 | -2.85 | -6.0% | -4.2 | 3.10 | 92 |
| 2 m | momentum (ret_75 >= 0.2, >= 5 buys) | 277 | -2.08 | -7.5% | -3.8 | 2.12 | 53 |
| 2 m | ridge @ val threshold | 920 | -3.77 | -4.1% | -6.2 | 3.81 | 180 |
| 2 m | LightGBM @ val threshold | 918 | -3.05 | -3.3% | -6.4 | 3.05 | 181 |
| 10 m | random | 153 | -2.49 | -16.3% | -5.2 | 2.49 | 28 |
| 10 m | momentum | 138 | -2.86 | -20.7% | -5.0 | 2.97 | 25 |
| 10 m | ridge | 173 | -0.51 | -2.9% | -4.8 | 0.51 | 34 |
| 10 m | LightGBM | 180 | -0.78 | -4.3% | -4.5 | 0.78 | 35 |

At 2 m and 10 m the models lose 1.8x to 5x less per trade than random or
momentum entries. At 30 s they are no better than random (ridge -3.3% vs
random -3.9%; LightGBM's validation threshold admitted only 14 trades).
**No-trade dominates every strategy at every horizon.** The 15-minute Sharpe ratios
are all deeply negative.

**Latency** (2 m LightGBM, 1 to 32 slots, i.e. 0.2 to 7 s): mean return
stays between -3.0% and -3.9% throughout. Nothing time-sensitive is being
captured, so there is nothing for latency to destroy. **Failure rate** (0,
5%, 20%) moves results by under 1 point.

**Capacity** (2 m LightGBM, latency 2): the loss per trade grows with size,
from -3.3% at 0.05 to 0.1 SOL, to -5.2% at 0.5, -7.7% at 1 and -12.2% at 2
SOL. Curve depth is a binding constraint even before any edge exists.

## Phase 3: token survival (landmark 60 s after launch, 6,027 tokens)

Kaplan-Meier cumulative incidence from the landmark, at 10 min / 6 h:
graduation 0.6% / 0.8%, liquidity collapse (real SOL < 20% of peak, peak >=
1 SOL) 21% / 26%, -50% drawdown 5.4% / 6.7%, -80% drawdown 0.8% / 1.7%.

Classifier for "endpoint within 10 minutes", trained on earlier launches and
tested on later ones:

| endpoint | AUC (all) | AUC (active: >= 10 trades by landmark) | Brier vs base rate |
|---|---|---|---|
| graduation | 0.996 | 0.986 | 0.0028 vs 0.0029 |
| collapse | 0.962 | 0.755 | 0.057 vs 0.157 |
| drawdown 50 | 0.970 | 0.928 | 0.024 vs 0.049 |
| drawdown 80 | 0.987 | 0.964 | 0.0034 vs 0.0034 |

Most of the headline AUC comes from separating dead tokens from live ones,
which is why the active-token column is reported. Collapse risk among
active tokens is moderately predictable (0.75). Graduation and the -80%
crash are rare enough that the Brier gain over the base rate is negligible
despite the high AUC. In the penalized Cox model, early trade count
dominates every hazard (HR 3.3 to 11 per SD): activity is what makes
anything happen. Curve progress raises graduation hazard (HR 1.57) and
lowers collapse hazard (HR 0.65).

## Phase 5: live paper trading (2026-10-12, 09:21 to 10:21 EDT)

`slotfish paper` tailed confirmed blocks live for 60 minutes (16.4k slots,
975k events), warm-started from the study window so wallet skill was
populated. It scored every decision point with the exported 2 m LightGBM
model, using the threshold chosen on validation; the Rust scorer matches
Python to 6e-17. Nothing was sent to the chain. Summary: `results/paper.json`.

| | |
|---|---|
| Paper entries | 1,546 signals: 1,459 settled round trips, 78 simulated failures, 8 slippage rejects, 1 graduated |
| Data lag (block time to block in hand) | **p50 0.14 s, p90 7.0 s, p99 21 s** |
| Effective latency used (lag + 1 submit slot) | median 8 slots (1.7 s) |
| Net return per trade at **measured** latency | **-3.4%** (SE 0.4%), hit rate 7.1% |
| Net return at the backtest's assumed 2-slot latency | -1.6% (SE 1.5%; median -3.3%; the mean is lifted by one outlier) |
| Backtest, same model, test window | -3.3% per trade |

* **Live replicates the backtest.** -3.4% live vs -3.3% in the backtest,
  on data that is a day newer than the study window. The 7% hit rate shows
  what the model does: it picks tokens that then barely move, so the round
  trip mostly pays the ~2.5% swap fees plus transaction costs.
* **The fill model is conservative.** For 53 entries, real curve buys of the
  same token landed in the same slot as our simulated order. In 96% of
  those, our simulated price was at or above the cheapest real buy, and the
  median was 0.41% *above* the most expensive real buy. End-of-slot filling
  prices us behind everyone in the slot, as intended.
* **Infrastructure.** The free RPC delivers confirmed blocks with
  sub-second median lag, but only keeps pace with the chain at 6 workers and
  24-slot batches. With 4 workers it fell behind by ~7 s/minute. The p90/p99
  tail (7 s, 21 s) comes from rate-limit retries, and that tail alone rules
  out anything latency-sensitive on free infrastructure.

**Verdict for real execution: do not.** No strategy has positive expected
value net of costs in either the backtest or live paper trading.

## Engine benchmark (6.36M events, 16-thread desktop, Windows)

| | |
|---|---|
| Load 391 MB parquet into the columnar tape | 4.0 s |
| Build per-token state index | 0.19 s |
| Replay, no strategy | **46M events/s**; p50 / p99 per-slot latency 0.8 / 2.6 us |
| Replay with full point-in-time features (incl. wallet skill and coordination) + momentum strategy | **5.1M events/s**; p50 / p99 / p99.9 per-slot latency **8.3 / 33 / 52 us** |
| Resident memory (process) | 0.78 GB |

A slot lasts 218 ms, so the feature engine uses about 0.015% of real time
at p99. A full 7.9 h replay takes 1.2 s, so parameter sweeps (90 runs in
parallel) finish in seconds. GPU acceleration is not justified: LightGBM on
300k rows trains in seconds on the CPU.

## What would change these conclusions

* **More history.** Eight hours is enough to reject large edges, not small
  ones. Wallet skill in particular needs days.
* **Shorting or exits on others' flow.** The predictable thing is collapse;
  the curve offers no short.
* **Better entry timing than the end-of-slot fill.** A sniper landing first
  in the launch slot is a different (and adversarially contested) game this
  engine deliberately does not assume.
