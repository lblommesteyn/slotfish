//! Phase 5: live paper trading.
//!
//! Tails finalized blocks from the RPC, appends decoded events to the same
//! Tape/Timeline/FeatureEngine used in research, scores every decision point
//! with the exported research model, and records hypothetical orders. Nothing
//! is ever sent to the chain.
//!
//! Two things are measured that a historical backtest cannot see:
//!   * **Real decision latency.** When a slot's block reaches us, wall-clock
//!     time minus that block's time is our data lag. An order decided then
//!     cannot land before `decision_slot + lag_slots + submit_slots`. Each paper
//!     trade is evaluated at that *measured* latency and also at the
//!     backtest's assumed latency, so the gap is explicit.
//!   * **Fill realism.** For each simulated entry we look at real curve buys
//!     of the same token in the landing slot and compare their realized price
//!     (SOL paid per token, fees included) with our simulated fill.
//!
//! Every live event is also written to parquet chunks, so a session can be
//! replayed later by the ordinary backtester.

use crate::amm::Pool;
use crate::engine::next_same_mint_in_slot;
use crate::events::{Event, Kind};
use crate::exec::{round_trip, ExecConfig};
use crate::features::FeatureEngine;
use crate::fetch::{fetch_range, Rpc};
use crate::market::Timeline;
use crate::model::Scorer;
use crate::store::{read_tape, write_events, Tape};
use anyhow::{Context, Result};
use serde_json::json;
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub struct PaperConfig {
    pub url: String,
    pub model: PathBuf,
    pub threshold: f64,
    pub hold_secs: f64,
    pub exec: ExecConfig,
    /// Slots between our decision and our transaction landing, on top of data lag.
    pub submit_slots: u64,
    pub workers: usize,
    pub out_dir: PathBuf,
    /// Historical event store replayed first so wallet skill starts warm.
    pub warm: Option<PathBuf>,
    pub duration_secs: f64,
    pub slot_secs: f64,
    pub commitment: String,
}

struct Pending {
    mint: u32,
    decision_slot: u64,
    score: f64,
    quote: Pool,
    lag_s: f64,
    lag_slots: u64,
}

fn now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

pub fn run(cfg: &PaperConfig) -> Result<()> {
    std::fs::create_dir_all(cfg.out_dir.join("events"))?;
    let scorer = Scorer::load(&cfg.model)?;
    let mut rpc = Rpc::new(&cfg.url)?;
    rpc.commitment = cfg.commitment.clone();
    let hold = (cfg.hold_secs / cfg.slot_secs).round() as u64;
    let skill_h = (120.0 / cfg.slot_secs).round() as u64;

    let mut tape = match &cfg.warm {
        Some(p) => read_tape(p)?,
        None => Tape::default(),
    };
    let mut tl = Timeline::build(&tape);
    let mut fe = FeatureEngine::new(skill_h);
    for i in 0..tape.len() {
        fe.update(&tape, &tl, i);
    }
    let warm_events = tape.len();
    eprintln!("warm start: {warm_events} events");

    let tip = rpc.call("getSlot", json!([{"commitment": rpc.commitment}]))?["result"]
        .as_u64()
        .context("getSlot")?;
    let live_start = tip;
    let mut next = tip;
    let mut pending: Vec<Pending> = Vec::new();
    let mut holding: HashSet<u32> = HashSet::new();
    let mut chunk_buf: Vec<Event> = Vec::new();
    let mut chunk_start = next;
    let mut trades_log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cfg.out_dir.join("paper_trades.jsonl"))?;
    let mut lag_log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cfg.out_dir.join("lag.jsonl"))?;
    let t_start = now_s();
    let mut feats = Vec::new();
    let mut assumed = cfg.exec.clone();
    assumed.p_fail = 0.0;

    while now_s() - t_start < cfg.duration_secs {
        let tip = rpc.call("getSlot", json!([{"commitment": rpc.commitment}]))?["result"]
            .as_u64()
            .context("getSlot")?;
        if tip < next {
            std::thread::sleep(Duration::from_millis(400));
            continue;
        }
        let hi = (tip + 1).min(next + 24);
        let (events, _) = fetch_range(&rpc, next, hi, cfg.workers);
        let arrived = now_s();
        let first_new = tape.len();
        for e in &events {
            tape.push_event(e);
            tl.push(&tape, tape.len() - 1);
        }
        chunk_buf.extend(events);
        // Data lag of the newest block we hold, measured on arrival.
        let newest_bt = *tape.block_time.last().unwrap_or(&0) as f64;
        let lag_s = arrived - newest_bt;
        writeln!(
            lag_log,
            "{}",
            json!({"t": arrived, "slots": [next, hi], "lag_s": lag_s})
        )?;
        let lag_slots = (lag_s / cfg.slot_secs).ceil().max(0.0) as u64;

        for i in first_new..tape.len() {
            fe.update(&tape, &tl, i);
            let m = tape.mint[i];
            let slot = tape.slot[i];
            if tape.kind[i] != Kind::Trade
                || !tl.info[m as usize].tradable_at(slot)
                || holding.contains(&m)
            {
                continue;
            }
            // Whole slots arrive together, so "last event of this mint in the slot" is known.
            if i + 1 < tape.len() && tape.slot[i + 1] == slot && next_same_mint_in_slot(&tape, i) {
                continue;
            }
            fe.features(&tape, &tl, i, &mut feats);
            let score = scorer.score(&feats);
            if score >= cfg.threshold {
                holding.insert(m);
                pending.push(Pending {
                    mint: m,
                    decision_slot: slot,
                    score,
                    quote: Pool {
                        sol: tape.vsol[i],
                        tok: tape.vtok[i],
                        fee_bps: tape.fee_bps[i],
                    },
                    lag_s,
                    lag_slots,
                });
            }
        }

        // Settle paper trades whose exit is now inside the data.
        let last = *tape.slot.last().unwrap_or(&0);
        let mut keep = Vec::new();
        for p in pending.drain(..) {
            let mut real = cfg.exec.clone();
            real.latency_slots = p.lag_slots + cfg.submit_slots;
            let horizon_end = p.decision_slot + 2 * real.latency_slots + hold + 2;
            if horizon_end > last {
                keep.push(p);
                continue;
            }
            let rt_real = round_trip(&tl, &real, p.mint, p.decision_slot, p.quote, hold, last);
            let rt_assumed =
                round_trip(&tl, &assumed, p.mint, p.decision_slot, p.quote, hold, last);
            let observed = observed_buys(&tape, p.mint, rt_real.entry_slot);
            let sim_px = if rt_real.entered {
                rt_real.sol_in as f64 / rt_real.tokens.max(1) as f64
            } else {
                f64::NAN
            };
            writeln!(
                trades_log,
                "{}",
                json!({
                    "mint": tape.mints.names[p.mint as usize], "decision_slot": p.decision_slot, "score": p.score,
                    "lag_s": p.lag_s, "latency_slots_real": real.latency_slots,
                    "real": {"entered": rt_real.entered, "skip": rt_real.skip_reason, "pnl": rt_real.pnl(),
                             "ret": rt_real.ret(real.size_lamports), "entry_slot": rt_real.entry_slot,
                             "exit_slot": rt_real.exit_slot, "censored": rt_real.censored},
                    "assumed": {"latency_slots": assumed.latency_slots, "entered": rt_assumed.entered,
                                "pnl": rt_assumed.pnl(), "ret": rt_assumed.ret(assumed.size_lamports)},
                    "fill_check": {"sim_lamports_per_token": sim_px, "real_buys_in_landing_slot": observed.0,
                                   "real_min_px": observed.1, "real_max_px": observed.2},
                })
            )?;
            holding.remove(&p.mint);
        }
        pending = keep;

        if hi - chunk_start >= 1000 {
            let path = cfg
                .out_dir
                .join("events")
                .join(format!("events_{chunk_start:012}.parquet"));
            write_events(&path, &chunk_buf)?;
            chunk_buf.clear();
            chunk_start = hi;
        }
        next = hi;
        eprintln!(
            "slot {next} (live {} slots), lag {lag_s:.1}s, events {}, open paper positions {}",
            next - live_start,
            tape.len() - warm_events,
            pending.len()
        );
    }
    if !chunk_buf.is_empty() {
        let path = cfg
            .out_dir
            .join("events")
            .join(format!("events_{chunk_start:012}.parquet"));
        write_events(&path, &chunk_buf)?;
    }
    Ok(())
}

/// Real curve buys of `mint` in `slot`: (count, min, max) lamports-per-token incl. fee.
fn observed_buys(t: &Tape, mint: u32, slot: u64) -> (usize, f64, f64) {
    let lo = t.slot.partition_point(|&s| s < slot);
    let hi = t.slot.partition_point(|&s| s <= slot);
    let mut n = 0;
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in lo..hi {
        if t.mint[i] == mint && t.kind[i] == Kind::Trade && t.is_buy[i] && t.token_amount[i] > 0 {
            let paid = t.sol_amount[i] as f64 * (1.0 + t.fee_bps[i] as f64 / 1e4);
            let px = paid / t.token_amount[i] as f64;
            n += 1;
            mn = mn.min(px);
            mx = mx.max(px);
        }
    }
    if n == 0 {
        (0, f64::NAN, f64::NAN)
    } else {
        (n, mn, mx)
    }
}
