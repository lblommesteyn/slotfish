# Execution model

Every trade in a label or a backtest is a full round trip through
`src/exec.rs`. The streaming engine (`src/engine.rs`) implements the same
semantics event by event, and `tests/engine.rs::engine_matches_reference_round_trip`
asserts the two agree trade for trade.

## Timeline of one trade

```
slot s          decision: strategy sees state after the mint's last trade in s
slot s+L        entry lands; fills against the state at the END of slot s+L
slot s+L+H      exit decided (fixed horizon H)
slot s+L+H+L    exit lands; fills at the END of that slot; on failure retry
                one slot later, paying fees again
```

* **Decision granularity.** At most one decision per (mint, slot), taken
  after the mint's last trade in that slot. That is everything a block-stream
  subscriber knows when the slot completes. Acting needs at least `L >= 1`
  slots.
* **End-of-slot fills.** Every observed trade in the landing slot is assumed
  to execute before ours. This is pessimistic for chasing momentum: the
  competing buyers in our slot move the price against us first.
* **No feedback.** The replayed history is fixed: our buy does not move the
  pool other traders see, and our exit sells into a pool that never got our
  buy. A round trip therefore pays price impact twice, which is conservative
  (`amm::tests::sell_into_unmoved_pool_is_conservative`). It also means
  capacity results are optimistic about *other* traders not reacting to us;
  sizes well beyond typical trade sizes are not credible.

## Costs

| cost | model | default |
|---|---|---|
| swap fee | event-reported bps for that token at that time (protocol + creator; PumpSwap adds LP) | from data (125 bps typical on the curve) |
| base fee | per landed transaction | 5,000 lamports |
| priority fee | per landed transaction, success or not | 100,000 lamports (~p75 observed) |
| failed entry | probability `p_fail`, pays tx cost, no position | 5% |
| slippage reject | fill worse than the decision-time quote by more than `max_slippage_bps` -> tx fails, pays tx cost | 2,000 bps |
| failed exit | probability `p_fail` per attempt; retry next slot; written off at zero after 20 attempts | |

Random failures come from a counter-based hash (`exec::uniform`) of
`(seed, mint, slot)`, so results do not depend on replay order or threading.

## Graduation

* If a token has graduated by the entry slot, the entry is skipped
  (`graduated`).
* A position that graduates mid-hold exits on the canonical PumpSwap pool
  when the pool has traded by the exit slot. Before the pool's first trade,
  the last curve state is used as the price: the pool opens with the curve's
  final reserves, so the price is continuous, but the fee differs.

## Censoring

Trades whose exit would land after the last slot in the data are marked
`censored` and excluded from every metric (and counted). Labels at the end of
the window are null for the same reason.

## Known optimism / pessimism

| assumption | direction |
|---|---|
| End-of-landing-slot fills | pessimistic for momentum, optimistic for contrarian entries |
| Double price impact (no feedback) | pessimistic |
| Other traders don't react to our orders | optimistic at large size |
| `p_fail` 5% (pump-native errors are rarer; network drops are not modelled) | uncertain; swept |
| No MEV sandwiching of our orders beyond the slippage limit | optimistic; the slippage limit caps the damage |
| Jito tips not modelled separately (folded into the priority fee) | uncertain; swept via `--priority-fee` |

## Validation against live data

`slotfish paper` evaluates each paper trade at its *measured* latency and
checks the simulated entry against real buys that landed in the same slot.
On 2026-10-12 (60 min live), in 96% of comparable entries the simulated
price was at or above the cheapest real buy, with a median 0.41% above the
most expensive one. That confirms end-of-slot fills are conservative.
Measured data lag on the free RPC: p50 0.14 s, p90 7 s, p99 21 s.
