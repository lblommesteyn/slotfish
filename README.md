# Slotfish

A deterministic, event-driven research engine for Solana memecoin
microstructure (pump.fun bonding curves and PumpSwap). It asks one question:

> Can wallet-network intelligence and microstructure signals predict
> **executable** memecoin returns after fees, liquidity, latency and adverse
> selection?

Slotfish answers it with real on-chain data, an execution model that charges
every cost we can measure, and evaluation that only ever touches the test
period once.

**Status:** Phases 1 to 4 baseline complete (data, replay engine, baseline
alpha research, validation). Phase 5 (live paper trading) is not started: the
fetcher already runs forward as a live tail, but no model has earned it yet.
See [Results](#results).

## Quick start

```bash
cargo build --release
# 1. capture a contiguous slot range (resumable; ~5 slots/s on the free endpoint)
./target/release/slotfish fetch --start <slot> --end <slot>
# 2. inspect
./target/release/slotfish info
# 3. reproduce everything: dataset, models, backtests, sweeps, benchmark, survival
python scripts/run_baseline.py
cargo test --release
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
| **Research** (`python/slotfish/`) | Purged chronological splits, ridge and LightGBM on executable labels, ablations, an hour-paired test of wallet increment, block-bootstrap CIs, calibration, landmark survival analysis. |
| **Tests** | Constant-product math; curve reserve continuity on real data; engine equals reference round trip; determinism; a lookahead test that recomputes features on a truncated tape (verified by mutation). |

## Results

See [RESULTS.md](RESULTS.md) for the numbers from the current capture and how
to read them.

## Layout

```
src/            Rust engine (fetch, decode, store, market, amm, exec, features, engine, strategy, dataset, report)
tests/          integration tests on a real 1000-slot fixture
python/slotfish research.py, survival.py
scripts/        run_baseline.py (end-to-end reproduction), dev/ (decoder forensics)
docs/           DATA.md, EXECUTION.md, ARCHITECTURE.md
results/        generated JSON (committed for the reported run)
```

## License

MIT
