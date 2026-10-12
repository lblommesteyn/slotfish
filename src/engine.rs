//! Deterministic event-driven replay.
//!
//! Events are consumed strictly in chain order. Orders are queued with a
//! landing slot and filled the moment the replay clock moves past that slot,
//! against the live state at that point: identical semantics to
//! `exec::round_trip` (end-of-landing-slot fills), which a test asserts.
//! Strategies see only the event just processed plus point-in-time features.

use crate::amm::Pool;
use crate::events::Kind;
use crate::exec::{uniform, ExecConfig, RoundTrip};
use crate::features::FeatureEngine;
use crate::market::Timeline;
use crate::store::Tape;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

/// What a strategy sees at each decision point.
pub struct Ctx<'a> {
    pub tape: &'a Tape,
    pub tl: &'a Timeline,
    pub i: usize,
    pub pool: Pool,
    /// Point-in-time features (empty if the strategy does not request them).
    pub features: &'a [f32],
}

pub trait Strategy {
    fn name(&self) -> String;
    fn needs_features(&self) -> bool {
        false
    }
    /// Return true to enter a position in `ctx.tape.mint[ctx.i]` now.
    fn on_decision(&mut self, ctx: &Ctx) -> bool;
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub exec: ExecConfig,
    pub hold_slots: u64,
    pub max_positions: usize,
    /// Only consider decisions in [decide_from, decide_to) (e.g. the test split).
    pub decide_from: u64,
    pub decide_to: u64,
    /// Wallet-skill outcome horizon (slots) for strategies that use features.
    pub skill_horizon: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Order {
    /// (position index)
    Entry(usize),
    Exit(usize),
}

pub struct RunStats {
    pub events: usize,
    pub decisions: usize,
    pub elapsed_s: f64,
    /// Per-event processing latency samples (ns), every event.
    pub latency_ns: Vec<u32>,
}

pub struct Replay<'a> {
    pub tape: &'a Tape,
    pub tl: &'a Timeline,
    pub cfg: EngineConfig,
}

impl<'a> Replay<'a> {
    pub fn run(
        &self,
        strat: &mut dyn Strategy,
        measure_latency: bool,
    ) -> (Vec<RoundTrip>, RunStats) {
        let (t, tl, cfg) = (self.tape, self.tl, &self.cfg);
        let n_mints = t.mints.names.len();
        let data_end = *t.slot.last().unwrap_or(&0);
        let mut live: Vec<Option<(Pool, bool)>> = vec![None; n_mints];
        let mut busy = vec![false; n_mints];
        let mut open = 0usize;
        let mut trades: Vec<RoundTrip> = Vec::new();
        let mut quotes: Vec<Pool> = Vec::new();
        let mut queue: BinaryHeap<Reverse<(u64, Order)>> = BinaryHeap::new();
        let mut fe = if strat.needs_features() {
            Some(FeatureEngine::new(t, tl, cfg.skill_horizon))
        } else {
            None
        };
        let mut feats: Vec<f32> = Vec::new();
        let mut decisions = 0usize;
        let mut lat = Vec::with_capacity(if measure_latency { t.len() } else { 0 });
        let ex = &cfg.exec;
        let t0 = Instant::now();

        // Fill every queued order whose landing slot is final (< `now`).
        let settle = |now: u64,
                      queue: &mut BinaryHeap<Reverse<(u64, Order)>>,
                      trades: &mut Vec<RoundTrip>,
                      live: &Vec<Option<(Pool, bool)>>,
                      busy: &mut Vec<bool>,
                      open: &mut usize,
                      quotes: &Vec<Pool>| {
            while let Some(Reverse((land, ord))) = queue.peek().copied() {
                if land >= now {
                    break;
                }
                queue.pop();
                match ord {
                    Order::Entry(k) => {
                        let rt = &mut trades[k];
                        let m = rt.mint as usize;
                        let state = live[m];
                        let mut ok = false;
                        match state {
                            None => rt.skip_reason = "nostate",
                            Some((_, true)) => rt.skip_reason = "graduated",
                            Some(_) if tl.info[m].complete_slot.is_some_and(|c| c <= land) => {
                                rt.skip_reason = "graduated"
                            }
                            Some((pool, false)) => {
                                rt.tx_costs = ex.tx_cost();
                                if uniform(ex.seed, rt.mint as u64, rt.decision_slot) < ex.p_fail {
                                    rt.skip_reason = "fail";
                                } else {
                                    let (quoted, _) = quotes[k].buy_with_budget(ex.size_lamports);
                                    let (tokens, fee) = pool.buy_with_budget(ex.size_lamports);
                                    if (tokens as f64)
                                        < quoted as f64 * (1.0 - ex.max_slippage_bps as f64 / 1e4)
                                    {
                                        rt.skip_reason = "slippage";
                                    } else {
                                        rt.entered = true;
                                        rt.tokens = tokens;
                                        rt.sol_in = ex.size_lamports;
                                        rt.swap_fees = fee;
                                        ok = true;
                                    }
                                }
                            }
                        }
                        if ok {
                            let land_exit = land + cfg.hold_slots + ex.latency_slots;
                            queue.push(Reverse((land_exit, Order::Exit(k))));
                        } else {
                            busy[m] = false;
                            *open -= 1;
                        }
                    }
                    Order::Exit(k) => {
                        let rt = &mut trades[k];
                        let m = rt.mint as usize;
                        rt.exit_attempts += 1;
                        rt.tx_costs += ex.tx_cost();
                        if uniform(ex.seed ^ 0x5E11, rt.mint as u64, land) >= ex.p_fail {
                            let (pool, on_amm) = live[m].expect("entered => state exists");
                            let (out, fee) = pool.sell(rt.tokens);
                            rt.sol_out = out;
                            rt.swap_fees += fee;
                            rt.exit_slot = land;
                            rt.exit_on_amm = on_amm;
                            busy[m] = false;
                            *open -= 1;
                        } else if rt.exit_attempts > ex.max_exit_retries {
                            rt.exit_slot = land + 1;
                            busy[m] = false;
                            *open -= 1;
                        } else {
                            queue.push(Reverse((land + 1, Order::Exit(k))));
                        }
                    }
                }
            }
        };

        for i in 0..t.len() {
            let ts = if measure_latency {
                Some(Instant::now())
            } else {
                None
            };
            let slot = t.slot[i];
            settle(
                slot,
                &mut queue,
                &mut trades,
                &live,
                &mut busy,
                &mut open,
                &quotes,
            );
            let m = t.mint[i] as usize;
            let kind = t.kind[i];
            let is_state = match kind {
                Kind::Trade => true,
                Kind::AmmBuy | Kind::AmmSell => {
                    tl.info[m].pool.as_deref() == Some(t.pools.names[t.pool[i] as usize].as_str())
                }
                _ => false,
            };
            if is_state {
                live[m] = Some((
                    Pool {
                        sol: t.vsol[i],
                        tok: t.vtok[i],
                        fee_bps: t.fee_bps[i],
                    },
                    kind != Kind::Trade,
                ));
            }
            if let Some(f) = fe.as_mut() {
                f.update(i);
            }
            let decide = kind == Kind::Trade
                && slot >= cfg.decide_from
                && slot < cfg.decide_to
                && tl.info[m].tradable_at(slot)
                // one decision per (mint, slot): only at the mint's last event in this slot
                && !(i + 1 < t.len() && t.slot[i + 1] == slot && next_same_mint_in_slot(t, i));
            if decide {
                decisions += 1;
                if let Some(f) = fe.as_ref() {
                    f.features(i, &mut feats);
                }
                let pool = live[m].unwrap().0;
                let ctx = Ctx {
                    tape: t,
                    tl,
                    i,
                    pool,
                    features: &feats,
                };
                if strat.on_decision(&ctx) && !busy[m] && open < cfg.max_positions {
                    busy[m] = true;
                    open += 1;
                    let k = trades.len();
                    trades.push(RoundTrip {
                        mint: m as u32,
                        decision_slot: slot,
                        entry_slot: slot + ex.latency_slots,
                        ..Default::default()
                    });
                    quotes.push(pool);
                    queue.push(Reverse((slot + ex.latency_slots, Order::Entry(k))));
                }
            }
            if let Some(ts) = ts {
                lat.push(ts.elapsed().as_nanos().min(u32::MAX as u128) as u32);
            }
        }
        // Drain orders that land within the data; the rest are censored.
        settle(
            data_end + 1,
            &mut queue,
            &mut trades,
            &live,
            &mut busy,
            &mut open,
            &quotes,
        );
        for Reverse((_, ord)) in queue.into_iter() {
            if let Order::Exit(k) = ord {
                trades[k].censored = true;
            }
        }
        let stats = RunStats {
            events: t.len(),
            decisions,
            elapsed_s: t0.elapsed().as_secs_f64(),
            latency_ns: lat,
        };
        (trades, stats)
    }
}

/// True if a later event in the same slot touches the same mint (scan is short: same slot only).
pub fn next_same_mint_in_slot(t: &Tape, i: usize) -> bool {
    let (slot, m) = (t.slot[i], t.mint[i]);
    let mut j = i + 1;
    while j < t.len() && t.slot[j] == slot {
        if t.mint[j] == m && t.kind[j] == Kind::Trade {
            return true;
        }
        j += 1;
    }
    false
}
