use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use rayon::prelude::*;
use slotfish::engine::{EngineConfig, Replay, Strategy};
use slotfish::exec::ExecConfig;
use slotfish::market::Timeline;
use slotfish::report::summarize;
use slotfish::store::{read_tape, Tape};
use slotfish::strategy::{Momentum, NoTrade, RandomEntry, Signal};
use slotfish::{dataset, decode, fetch, store};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "slotfish",
    about = "Solana pump.fun microstructure research engine"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct ExecArgs {
    /// Slots between decision and landing.
    #[arg(long, default_value_t = 2)]
    latency: u64,
    /// Position size in SOL.
    #[arg(long, default_value_t = 0.1)]
    size: f64,
    /// Priority fee per transaction, lamports.
    #[arg(long, default_value_t = 100_000)]
    priority_fee: u64,
    #[arg(long, default_value_t = 0.05)]
    p_fail: f64,
    #[arg(long, default_value_t = 2000)]
    max_slippage_bps: u32,
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

impl ExecArgs {
    fn cfg(&self) -> ExecConfig {
        ExecConfig {
            latency_slots: self.latency,
            size_lamports: (self.size * 1e9) as u64,
            priority_fee: self.priority_fee,
            p_fail: self.p_fail,
            max_slippage_bps: self.max_slippage_bps,
            seed: self.seed,
            ..Default::default()
        }
    }
}

#[derive(Args, Clone)]
struct StratArgs {
    /// none | random | momentum | signal
    #[arg(long, default_value = "momentum")]
    strategy: String,
    #[arg(long, default_value_t = 0.01)]
    random_p: f64,
    #[arg(long, default_value_t = 0.2)]
    mom_ret: f32,
    #[arg(long, default_value_t = 5.0)]
    mom_buys: f32,
    /// Parquet with (slot u64, mint utf8, score f64) for --strategy signal.
    #[arg(long)]
    signal: Option<PathBuf>,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    threshold: f64,
    #[arg(long, default_value = "model")]
    label: String,
}

impl StratArgs {
    fn build(&self, tape: &Tape) -> Result<Box<dyn Strategy + Send>> {
        Ok(match self.strategy.as_str() {
            "none" => Box::new(NoTrade),
            "random" => Box::new(RandomEntry {
                p: self.random_p,
                seed: 11,
            }),
            "momentum" => Box::new(Momentum {
                min_ret: self.mom_ret,
                min_buys: self.mom_buys,
            }),
            "signal" => {
                let Some(p) = &self.signal else {
                    bail!("--signal required")
                };
                Box::new(Signal::load(
                    p,
                    &tape.mints,
                    self.threshold,
                    self.label.clone(),
                )?)
            }
            s => bail!("unknown strategy {s}"),
        })
    }
}

#[derive(Args, Clone)]
struct Window {
    /// Decide only in slots >= this fraction of the data's slot range.
    #[arg(long, default_value_t = 0.0)]
    from_frac: f64,
    #[arg(long, default_value_t = 1.0)]
    to_frac: f64,
    /// Explicit decision window [from_slot, to_slot) (overrides the fractions).
    #[arg(long)]
    from_slot: Option<u64>,
    #[arg(long)]
    to_slot: Option<u64>,
    /// Hold period in slots (overridden by --hold-secs).
    #[arg(long, default_value_t = 554)]
    hold: u64,
    /// Hold period in seconds, converted with the measured slot time.
    #[arg(long)]
    hold_secs: Option<f64>,
    #[arg(long, default_value_t = 20)]
    max_positions: usize,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fetch contiguous slots over RPC, decode, write chunked event parquet.
    Fetch {
        #[arg(long, default_value = "https://solana-rpc.publicnode.com")]
        url: String,
        #[arg(long, default_value = "data/events")]
        out: PathBuf,
        #[arg(long)]
        start: u64,
        #[arg(long)]
        end: u64,
        #[arg(long, default_value_t = 4)]
        workers: usize,
    },
    /// Decode a gzip JSONL file of slot records (fixtures / raw archives) to parquet.
    DecodeRaw { input: PathBuf, output: PathBuf },
    /// Summarize an event store.
    Info {
        #[arg(long, default_value = "data/events")]
        events: PathBuf,
    },
    /// Run one strategy through the event-driven replay.
    Backtest {
        #[arg(long, default_value = "data/events")]
        events: PathBuf,
        #[command(flatten)]
        exec: ExecArgs,
        #[command(flatten)]
        strat: StratArgs,
        #[command(flatten)]
        window: Window,
        /// Write per-trade CSV here.
        #[arg(long)]
        trades: Option<PathBuf>,
    },
    /// Grid over latency x size x hold x p_fail for one strategy (parallel runs).
    Sweep {
        #[arg(long, default_value = "data/events")]
        events: PathBuf,
        #[command(flatten)]
        exec: ExecArgs,
        #[command(flatten)]
        strat: StratArgs,
        #[command(flatten)]
        window: Window,
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
        latencies: Vec<u64>,
        #[arg(long, value_delimiter = ',', default_value = "0.1")]
        sizes: Vec<f64>,
        /// Hold periods in seconds.
        #[arg(long, value_delimiter = ',', default_value = "120")]
        holds: Vec<f64>,
        #[arg(long, value_delimiter = ',', default_value = "0.05")]
        p_fails: Vec<f64>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Build the supervised dataset (point-in-time features + executable labels).
    Dataset {
        #[arg(long, default_value = "data/events")]
        events: PathBuf,
        #[command(flatten)]
        exec: ExecArgs,
        #[arg(long, default_value = "data/dataset.parquet")]
        out: PathBuf,
    },
    /// Check the Rust scorer reproduces Python's predictions for an exported model.
    VerifyModel { model: PathBuf },
    /// Live paper trading: tail finalized blocks, score decisions, record hypothetical trades.
    Paper {
        #[arg(long, default_value = "https://solana-rpc.publicnode.com")]
        url: String,
        #[arg(long)]
        model: PathBuf,
        #[arg(long, allow_negative_numbers = true)]
        threshold: f64,
        #[arg(long, default_value_t = 120.0)]
        hold_secs: f64,
        #[command(flatten)]
        exec: ExecArgs,
        /// Slots to submit and land after a decision, on top of measured data lag.
        #[arg(long, default_value_t = 1)]
        submit_slots: u64,
        #[arg(long, default_value_t = 4)]
        workers: usize,
        #[arg(long, default_value = "data/paper")]
        out: PathBuf,
        /// Replay this historical store first so wallet skill starts warm.
        #[arg(long)]
        warm: Option<PathBuf>,
        #[arg(long, default_value_t = 3600.0)]
        duration_secs: f64,
        #[arg(long, default_value_t = 0.217)]
        slot_secs: f64,
        /// Block commitment for the live tail: confirmed (what a bot would use) or finalized.
        #[arg(long, default_value = "confirmed")]
        commitment: String,
    },
    /// Measure replay throughput, per-event latency and memory.
    Bench {
        #[arg(long, default_value = "data/events")]
        events: PathBuf,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
    },
}

fn load(events: &Path) -> Result<(Tape, Timeline)> {
    let t0 = Instant::now();
    let tape = read_tape(events)?;
    let tl = Timeline::build(&tape);
    eprintln!(
        "loaded {} events, {} mints in {:.2}s",
        tape.len(),
        tape.mints.names.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok((tape, tl))
}

fn slot_secs(t: &Tape) -> f64 {
    let (s0, s1) = (t.slot[0], *t.slot.last().unwrap());
    let (b0, b1) = (t.block_time[0], *t.block_time.last().unwrap());
    if s1 > s0 && b1 > b0 {
        (b1 - b0) as f64 / (s1 - s0) as f64
    } else {
        0.4
    }
}

fn engine_cfg(t: &Tape, exec: ExecConfig, w: &Window) -> EngineConfig {
    let (s0, s1) = (t.slot[0], *t.slot.last().unwrap());
    let span = (s1 - s0) as f64;
    let ss = slot_secs(t);
    EngineConfig {
        exec,
        hold_slots: w
            .hold_secs
            .map(|h| dataset::horizon_slots(h, ss))
            .unwrap_or(w.hold),
        skill_horizon: dataset::horizon_slots(120.0, ss),
        max_positions: w.max_positions,
        decide_from: w.from_slot.unwrap_or(s0 + (span * w.from_frac) as u64),
        decide_to: w.to_slot.unwrap_or(s0 + (span * w.to_frac) as u64),
    }
}

fn write_trades(path: &Path, tape: &Tape, trades: &[slotfish::exec::RoundTrip]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "mint,decision_slot,entry_slot,exit_slot,entered,skip_reason,censored,sol_in,sol_out,swap_fees,tx_costs,pnl,exit_attempts,exit_on_amm")?;
    for t in trades {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            tape.mints.names[t.mint as usize],
            t.decision_slot,
            t.entry_slot,
            t.exit_slot,
            t.entered,
            t.skip_reason,
            t.censored,
            t.sol_in,
            t.sol_out,
            t.swap_fees,
            t.tx_costs,
            t.pnl(),
            t.exit_attempts,
            t.exit_on_amm
        )?;
    }
    Ok(())
}

fn percentile(v: &mut [u32], q: f64) -> u32 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() - 1) as f64 * q) as usize]
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Fetch {
            url,
            out,
            start,
            end,
            workers,
        } => fetch::run(&url, &out, start, end, workers),
        Cmd::DecodeRaw { input, output } => {
            let f =
                std::io::BufReader::new(flate2::read::GzDecoder::new(std::fs::File::open(&input)?));
            let mut evs = Vec::new();
            for line in f.lines() {
                let v: serde_json::Value = serde_json::from_str(&line?)?;
                decode::decode_slot(&v, &mut evs);
            }
            store::write_events(&output, &evs)?;
            println!("{} events -> {}", evs.len(), output.display());
            Ok(())
        }
        Cmd::Info { events } => {
            let (t, tl) = load(&events)?;
            let mut kinds = [0usize; 7];
            for k in &t.kind {
                kinds[*k as usize] += 1;
            }
            let cohort = tl.info.iter().filter(|m| m.in_cohort()).count();
            let created = tl.info.iter().filter(|m| m.create_slot.is_some()).count();
            let mayhem = tl
                .info
                .iter()
                .filter(|m| m.create_slot.is_some() && m.mayhem)
                .count();
            let zero_fee_unflagged = tl
                .info
                .iter()
                .filter(|m| m.in_cohort() && m.zero_fee_slot.is_some())
                .count();
            let grads = tl
                .info
                .iter()
                .filter(|m| m.in_cohort() && m.complete_slot.is_some())
                .count();
            let info = serde_json::json!({
                "events": t.len(), "first_slot": t.slot[0], "last_slot": t.slot.last(),
                "first_time": t.block_time[0], "last_time": t.block_time.last(), "slot_secs": slot_secs(&t),
                "kinds": {"create": kinds[0], "trade": kinds[1], "complete": kinds[2], "migrate": kinds[3],
                          "amm_buy": kinds[4], "amm_sell": kinds[5], "failed": kinds[6]},
                "mints": t.mints.names.len(), "wallets": t.users.names.len(),
                "launched_in_window": created, "mayhem_launches": mayhem, "cohort_zero_fee_tainted": zero_fee_unflagged, "cohort": cohort,
                "cohort_graduated": grads,
            });
            println!("{}", serde_json::to_string_pretty(&info)?);
            Ok(())
        }
        Cmd::Backtest {
            events,
            exec,
            strat,
            window,
            trades,
        } => {
            let (t, tl) = load(&events)?;
            let cfg = engine_cfg(&t, exec.cfg(), &window);
            let mut s = strat.build(&t)?;
            let replay = Replay {
                tape: &t,
                tl: &tl,
                cfg: cfg.clone(),
            };
            let (tr, st) = replay.run(s.as_mut(), false);
            let rep = summarize(&s.name(), &tr, cfg.exec.size_lamports, slot_secs(&t));
            eprintln!(
                "replayed {} events, {} decisions in {:.2}s",
                st.events, st.decisions, st.elapsed_s
            );
            println!("{}", serde_json::to_string_pretty(&rep)?);
            if let Some(p) = trades {
                write_trades(&p, &t, &tr)?;
            }
            Ok(())
        }
        Cmd::Sweep {
            events,
            exec,
            strat,
            window,
            latencies,
            sizes,
            holds,
            p_fails,
            out,
        } => {
            let (t, tl) = load(&events)?;
            let mut grid = Vec::new();
            for &l in &latencies {
                for &s in &sizes {
                    for &h in &holds {
                        for &p in &p_fails {
                            grid.push((l, s, h, p));
                        }
                    }
                }
            }
            let ss = slot_secs(&t);
            let lines: Vec<String> = grid
                .par_iter()
                .map(|&(l, s, h, p)| {
                    let mut e = exec.clone();
                    e.latency = l;
                    e.size = s;
                    e.p_fail = p;
                    let mut w = window.clone();
                    w.hold_secs = Some(h);
                    let cfg = engine_cfg(&t, e.cfg(), &w);
                    let mut st = strat.build(&t).expect("strategy");
                    let (tr, _) = Replay {
                        tape: &t,
                        tl: &tl,
                        cfg: cfg.clone(),
                    }
                    .run(st.as_mut(), false);
                    let rep = summarize(&st.name(), &tr, cfg.exec.size_lamports, ss);
                    let mut v = serde_json::to_value(&rep).unwrap();
                    v["latency"] = l.into();
                    v["size"] = s.into();
                    v["hold"] = h.into();
                    v["p_fail"] = p.into();
                    v.to_string()
                })
                .collect();
            std::fs::write(&out, lines.join("\n") + "\n")?;
            println!("{} runs -> {}", lines.len(), out.display());
            Ok(())
        }
        Cmd::Dataset { events, exec, out } => {
            let (t, tl) = load(&events)?;
            let t0 = Instant::now();
            let n = dataset::build(&t, &tl, &exec.cfg(), slot_secs(&t), &out)?;
            println!(
                "{n} rows -> {} in {:.1}s",
                out.display(),
                t0.elapsed().as_secs_f64()
            );
            Ok(())
        }
        Cmd::VerifyModel { model } => {
            let sc = slotfish::model::Scorer::load(&model)?;
            let (n, worst) = sc.verify(&model.with_extension("verify.json"))?;
            println!(
                "{}",
                serde_json::json!({"model": model, "rows": n, "max_abs_diff": worst})
            );
            if worst > 1e-9 {
                bail!("scorer disagrees with Python by {worst}");
            }
            Ok(())
        }
        Cmd::Paper {
            url,
            model,
            threshold,
            hold_secs,
            exec,
            submit_slots,
            workers,
            out,
            warm,
            duration_secs,
            slot_secs,
            commitment,
        } => slotfish::paper::run(&slotfish::paper::PaperConfig {
            url,
            model,
            threshold,
            hold_secs,
            exec: exec.cfg(),
            submit_slots,
            workers,
            out_dir: out,
            warm,
            duration_secs,
            slot_secs,
            commitment,
        }),
        Cmd::Bench { events, repeats } => {
            let t0 = Instant::now();
            let tape = read_tape(&events)?;
            let load_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let tl = Timeline::build(&tape);
            let index_s = t1.elapsed().as_secs_f64();
            let mut out = serde_json::json!({
                "events": tape.len(), "load_s": load_s, "index_s": index_s,
                "tape_column_bytes": tape.column_bytes(),
                "threads": rayon::current_num_threads(),
            });
            let cfg = engine_cfg(
                &tape,
                ExecConfig::default(),
                &Window {
                    from_frac: 0.0,
                    to_frac: 1.0,
                    from_slot: None,
                    to_slot: None,
                    hold: 554,
                    hold_secs: None,
                    max_positions: 20,
                },
            );
            for (name, mk) in [("replay_no_trade", 0u8), ("replay_momentum_features", 1u8)] {
                let mut best = f64::MAX;
                let mut lat_all = Vec::new();
                for r in 0..repeats {
                    let mut s: Box<dyn Strategy> = if mk == 0 {
                        Box::new(NoTrade)
                    } else {
                        Box::new(Momentum {
                            min_ret: 0.2,
                            min_buys: 5.0,
                        })
                    };
                    // Throughput runs without per-event timers; the last repeat measures latency.
                    let timed = r + 1 == repeats;
                    let (_, st) = Replay {
                        tape: &tape,
                        tl: &tl,
                        cfg: cfg.clone(),
                    }
                    .run(s.as_mut(), timed);
                    if !timed {
                        best = best.min(st.elapsed_s);
                    } else {
                        lat_all = st.latency_ns;
                    }
                }
                if repeats == 1 {
                    best = f64::NAN;
                }
                let p50 = percentile(&mut lat_all, 0.5);
                let p99 = percentile(&mut lat_all, 0.99);
                let p999 = percentile(&mut lat_all, 0.999);
                out[name] = serde_json::json!({
                    "best_s": best, "events_per_s": tape.len() as f64 / best,
                    "slot_latency_us_p50": p50 as f64 / 1e3, "slot_latency_us_p99": p99 as f64 / 1e3,
                    "slot_latency_us_p999": p999 as f64 / 1e3, "slots_timed": lat_all.len(),
                });
            }
            if let Some(m) = memory_stats::memory_stats() {
                out["physical_mem_bytes"] = m.physical_mem.into();
            }
            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(())
        }
    }
}
