# Architecture

```
            publicnode RPC (getBlock, contiguous slots)
                         |
          slotfish fetch (Rust, 4 workers, resumable)
                         |  decode.rs: Anchor event-CPI -> typed events
                         v
   data/events/events_<chunk>.parquet  (+ manifest.jsonl)       <- the only stored data
                         |
          store::read_tape -> Tape (columnar, interned u32 ids)
                         |
          market::Timeline (per-mint CSR state history, binary search)
             /                     |                         \
  engine::Replay            dataset::build                 bench
  (event-driven backtest:   (FeatureEngine streaming PIT     (throughput, p50/p99
   strategies, order queue,  features + exec::round_trip      per-event latency,
   end-of-slot fills)        executable labels -> parquet)    memory)
        |                          |
  report::summarize          python/slotfish/research.py  (ridge, LightGBM, ablations,
  (net P&L, Sharpe, DD,        chronological purged splits, bootstrap CIs)
   turnover, skips)                 |                \
        ^                     signal_<model>_<h>.parquet   python/slotfish/survival.py
        |__________________________|                    (KM, Cox, landmark classifier)
          strategy::Signal (scores keyed by (slot, mint))
```

## Modules (`src/`)

| module | role |
|---|---|
| `fetch` | RPC client and chunked, atomic, resumable backfill. Doubles as the live tail (Phase 5). |
| `decode` | Raw block JSON to `Event`s. Anchor event discriminators, borsh reader, parent-instruction tracking for PumpSwap base mints, failed-attempt decoding. |
| `events` | The canonical event row (post-event state semantics). |
| `store` | Parquet write/read. `Tape` is the columnar in-memory form. |
| `market` | `Timeline`: per-mint state history and universe metadata (`MintInfo::tradable_at`, the point-in-time universe rule). |
| `amm` | Constant-product math with fees and program-faithful rounding. |
| `exec` | Execution model and the reference `round_trip`. |
| `features` | Streaming point-in-time feature engine with Bayesian wallet skill. |
| `engine` | Event-driven replay with strategy plugins (`Strategy` trait). |
| `strategy` | `NoTrade`, `RandomEntry`, `Momentum`, `Signal` (offline model scores). |
| `dataset` | Feature + label table for research. |
| `report` | Metrics. |

## Design rules

1. **One clock.** Everything is ordered by `(slot, tx_idx, ev_idx)`.
   Decisions happen at a mint's last event in a slot. Orders fill when the
   clock passes their landing slot.
2. **Point-in-time by construction.** Features are computed in a single
   forward pass. Wallet-skill outcomes enter the posterior only after their
   horizon has elapsed on the replay clock. The universe rule uses the
   launch-time Mayhem flag and the first-seen zero-fee slot, never future
   facts. Tested by `features_use_no_future_information`, which recomputes
   features on a tape physically truncated at the decision event. A
   mutation check (injecting a 3-slot peek) makes the test fail.
3. **One execution model.** Labels and backtests share `exec` semantics, and
   `engine_matches_reference_round_trip` checks the streaming engine against
   the reference trade for trade.
4. **Determinism.** No wall-clock or thread-order dependence. Failures come
   from a counter-based hash. LightGBM runs with `deterministic=True` and a
   fixed seed.
5. **Python never touches raw data.** It reads Rust-built tables, so leakage
   control lives in one place.

## Why Rust, and no GPU (yet)

Replay is branchy, sequential and memory-bound, so it belongs on the CPU.
See `results/bench.json` for measured throughput. GPU acceleration was not
used: with ~10^5 to 10^6 labelled rows, LightGBM on the CPU trains in
seconds. A GPU would only be justified for the deferred temporal-GNN work,
and only after a benchmark shows CPU training as the bottleneck.
