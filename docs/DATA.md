# Data

## Source audit (2026-10-12)

The goal was the cheapest source that gives *complete* pump.fun order flow
(every launch, every trade, every wallet, failed transactions included) with
exact pool state, without trusting a vendor's survivorship choices.

| Source | Cost | Coverage | Verdict |
|---|---|---|---|
| `api.mainnet-beta.solana.com` (official public RPC) | free | full archive (Bigtable) | `getBlock` throttled to ~0.13 blocks/s (429, retry-after 10s). `getTransaction` ~1/s. Unusable for contiguous history; fine for spot checks. |
| `solana-rpc.publicnode.com` | free, keyless | rolling ~418k slots (~25 h at the current 0.217 s slot time) | ~5.5 blocks/s with 4 workers (8 workers get disconnected). **Chosen.** |
| drpc, onfinality, ankr, blastapi, omniatech public endpoints | free | n/a | Errors, keys required, or no `getBlock` (tested 2026-10-12). |
| leorpc free key | free | recent only | ~0.4 blocks/s, no history. |
| Google BigQuery `crypto_solana_mainnet_us` | 1 TB/month free | historical | Reported stale since 2025-03-31; needs a GCP account. Not used. |
| Bitquery / Dune / Helius / Blockworks | paid or keyed | good | Not needed for a first, trustworthy baseline; vendor-decoded data also hides decoding choices. |
| Hugging Face datasets | free | none found | No public pump.fun trade-level dataset located. |

**Decision:** pull contiguous blocks from publicnode and decode them
ourselves. Contiguous slots (rather than per-token history) mean the cohort is
every token launched in the window, so the dataset cannot be survivorship
biased: tokens that never traded again are present, as are rugs.

The trade-off is a rolling ~25 h horizon. The fetcher must outrun ledger
pruning (~5.1 slots/s observed vs ~5.5 slots/s fetch rate), and history older
than ~1 day is not reachable for free. Longer studies need either repeated
captures (the same `fetch` command run forward, which is also the Phase 5 live
path) or a paid archive.

## What is decoded

pump.fun (`6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`) and PumpSwap
(`pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA`) are Anchor programs that emit
events by self-CPI (`emit_cpi!`). Each event is an inner instruction whose data
is `e445a52e51cb9a1d ++ sha256("event:<Name>")[..8] ++ borsh(fields)`. The
decoder (`src/decode.rs`) reads:

| kind | source event | key fields |
|---|---|---|
| `Create` (0) | `CreateEvent` | mint, creator, initial reserves, `is_mayhem_mode` (stored in `is_buy`) |
| `Trade` (1) | `TradeEvent` | mint, user, direction, SOL and token amounts, **post-trade virtual and real reserves**, fee bps and lamports (protocol + creator) |
| `Complete` (2) | `CompleteEvent` | curve hit its graduation threshold |
| `Migrate` (3) | `CompletePumpAmmMigrationEvent` | mint -> canonical PumpSwap pool |
| `AmmBuy`/`AmmSell` (4/5) | PumpSwap `BuyEvent`/`SellEvent` | pool, user, amounts, pool reserves (converted to post-trade), total fee |
| `Failed` (6) | failed tx with a pump/PumpSwap buy/sell instruction | mint, user, venue, error string |

Every row means *state after this event*. Rows are ordered by
`(slot, tx_idx, ev_idx)`, the on-chain execution order. Each row's
`(slot, tx_idx)` pair is an exact pointer back to the chain, so signatures are
not stored (they are 64 incompressible bytes per row and would roughly double
the store).

## Validation

* **Curve state continuity.** For every bonding-curve trade on a non-Mayhem
  token, `post - own delta` equals the previous event's post state, exactly,
  for SOL and tokens (100.0% of 4,385 transitions in the first sample chunk;
  enforced by `tests/engine.rs::curve_reserves_are_continuous_for_cohort`).
  Standard curves show the expected 30 SOL virtual offset.
* **PumpSwap layout** was pinned empirically (`scripts/dev/amm_layout.py`):
  the pool quote reserve moves by `quote_amount_in_with_lp_fee` (buys) /
  `quote_amount_out_without_lp_fee` (sells). Token continuity is 99.5% and SOL
  continuity ~90% per pool. The breaks are concentrated in `BuyEvent` layout
  variants (event lengths 500/508/515 bytes, ~10% of buys, probably the
  exact-quote-in instruction family): base-reserve accounting is exact for
  them, but the quote-reserve change follows neither the standard field
  semantics nor constant product. Because every PumpSwap event reports its
  own pre-trade reserves, the error never accumulates; it lasts only until
  the next event on that pool. PumpSwap state is used only to price exits of
  positions that graduate mid-hold, a small share of trades (counted as
  `exits_on_amm` in every report).
* **Mayhem mode.** Tokens launched with `is_mayhem_mode` are traded by
  pump.fun's Mayhem agent (program `MAyhSm...`) with zero-fee trades whose
  reserve updates are *not* constant-product consistent. They are flagged at
  launch and excluded from the tradable universe. A few unflagged tokens also
  receive zero-fee trades; such a token stops being tradable from the slot of
  its first zero-fee trade (point-in-time rule, `MintInfo::tradable_at`).

* **Non-SOL quote curves.** About 6% of curve trades belong to tokens whose
  curve is quoted in another asset: their TradeEvents report the quote
  asset's amounts and reserves in the SOL fields (often zero). They are identified at launch (CreateEvent virtual SOL differs
  from the standard 30 SOL) and excluded, since SOL-denominated execution
  cannot be simulated on them.

## Failed transactions

About half of all landed transactions that invoke these programs fail. The
dominant error on curve buys (`Custom 7`, about 74% of curve-buy attempts in
the sample) is raised by *bots' own guard programs*, not by pump.fun: bots
submit speculative transactions that abort cheaply when unprofitable. Those
failures say nothing about an ordinary order's failure rate. The execution
model's `p_fail` is therefore a swept assumption (default 5%) rather than this
raw rate; the share of pump-native errors (60xx) is reported separately.

## Storage

`data/events/events_<chunk>.parquet`: one zstd parquet file per 1000 slots
(~3 MB, ~60k rows), written atomically, with `manifest.jsonl` recording each
chunk's skipped and missing slots. A day is ~650 MB. Raw block JSON
(~2.5 MB per block, ~50 GB/day) is never stored.

## Timing facts measured from the data

* Slot time is **0.217 s** (2000 slots in 433 s; no skipped slots in the first
  chunks), not the historical 0.4 s. All horizons are specified in seconds and
  converted with the measured slot time.
* The priority fee on successful curve trades has a median of 46k lamports
  and a p75 of 109k. The default execution cost is 5k base + 100k priority per
  transaction.
