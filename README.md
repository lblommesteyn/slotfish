# Slotfish

A deterministic, event-driven research engine for Solana memecoin
microstructure (pump.fun bonding curves and PumpSwap). It asks one question:

> Can wallet-network intelligence and microstructure signals predict
> **executable** memecoin returns after fees, liquidity, latency and adverse
> selection?

Slotfish answers it with real on-chain data, an execution model that charges
every cost we can measure, and evaluation that only ever touches the test
period once.

**Status:** all five phases have a working baseline. A 7.9 h contiguous
capture (6.36M events, 6,027-token survivorship-free cohort) was studied
end to end, and a 60-minute live paper-trading session ran on the result.
**Finding: no profitable strategy.** Order-flow and token-state features
rank executable returns out of sample (2 m IC 0.21), but only by
identifying losers, and wallet intelligence adds nothing. Live paper
trading reproduced the backtest's -3.3% per trade. See [RESULTS.md](RESULTS.md).

## Quick start

```bash
cargo build --release
# 1. capture a contiguous slot range (resumable; ~5 slots/s on the free endpoint)
scripts/fetch_supervised.sh --start <slot> --end <slot>   # restarts on crash
# 2. inspect
./target/release/slotfish info
# 3. reproduce everything: dataset, models, backtests, sweeps, benchmark, survival
python scripts/run_baseline.py
cargo test --release
# live paper trading with an exported model (no transactions are ever sent)
./target/release/slotfish paper --model results/research/model_lgbm_2m.json \
    --threshold=<val threshold> --warm data/events --workers 6
python scripts/paper_report.py
```

Requirements: Rust 1.85+, Python 3.12 with `polars pyarrow lightgbm
scikit-learn scipy statsmodels`. No API keys. No GPU.

## What is in the box

| | |
|---|---|
| **Data** ([docs/DATA.md](docs/DATA.md)) | Contiguous blocks from a free RPC, decoded from pump.fun/PumpSwap Anchor events: launches, every swap with post-trade reserves, graduations, migrations, failed transactions. Every launch in the window is included, so the cohort cannot be survivorship biased. ~650 MB/day as zstd parquet. |
| **Replay engine** ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) | Rust. Columnar tape, per-mint state index, strategy plugins, an order queue with landing slots, parameter sweeps in parallel (rayon). |
| **Execution model** ([docs/EXECUTION.md](docs/EXECUTION.md)) | Latency in slots, end-of-landing-slot fills, double price impact (no feedback), protocol + creator + LP fees, base + priority fees on every landed tx, slippage rejects, random failures with retries, graduation handling, censoring. |
| **Point-in-time features** | Streaming order flow, token state, and Bayesian wallet skill whose outcomes are revealed only after their horizon passes. |
| **Research** (`python/slotfish/`) | Purged chronological splits, ridge and LightGBM on executable labels, ablations, a block-paired test of wallet increment, token-cluster bootstrap CIs, calibration, landmark survival analysis (KM, penalized Cox, classifiers). |
| **Live paper trading** | `slotfish paper`: tails confirmed blocks, scores with the exported model (Rust scorer equals Python), evaluates trades at measured latency, checks fills against real same-slot buys. |
| **Tests** | Decoder snapshot on real blocks; constant-product math; curve reserve continuity on real data; engine equals reference round trip; determinism; a lookahead test that recomputes features on a truncated tape (verified by mutation). |

## Results

[RESULTS.md](RESULTS.md) has the full write-up. Headlines:

| | 30 s | 2 m | 10 m |
|---|---|---|---|
| LightGBM test IC (15-min block t) | 0.135 (3.7) | 0.211 (4.2) | 0.431 (14.3) |
| Top-decile net return | -3.4% | -4.7% | -2.0% |
| Backtest net/trade: LightGBM / momentum / random | -7.4%* / -4.3% / -3.9% | -3.3% / -7.5% / -6.0% | -4.3% / -20.7% / -16.3% |
| Wallet features' IC increment over momentum (t) | -0.030 (-1.9) | -0.022 (-2.8) | +0.003 (0.1) |

\* only 14 trades passed the validation threshold.

Engine: 5.1M events/s with full point-in-time features; per-block latency
p50 8 us, p99 33 us.

## Layout

```
src/            Rust engine (fetch, decode, store, market, amm, exec, features, engine, strategy, dataset, report)
tests/          decoder snapshot (3 real blocks) and engine invariants (real 1000-slot fixture)
python/slotfish research.py, survival.py
src/paper.rs    live paper trader; src/model.rs: LightGBM/ridge scorer
scripts/        run_baseline.py (end-to-end), make_report.py, data_quality.py, paper_report.py, fetch_supervised.sh, dev/
docs/           DATA.md, EXECUTION.md, ARCHITECTURE.md
results/        generated JSON (committed for the reported run)
```

## License

MIT
