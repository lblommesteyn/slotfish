//! Contiguous-slot block fetcher + streaming decoder.
//!
//! Pulls full blocks over JSON-RPC (`getBlock`), decodes pump.fun and PumpSwap
//! events in memory and writes one compact parquet file per CHUNK slots. Raw
//! JSON is never stored: at ~2.5 MB per block the archive would be ~50 GB/day.
//! Every row keeps (slot, tx_idx), an exact on-chain pointer, so any event can
//! be re-fetched and audited.
//!
//! Contiguous slot ranges (rather than per-token history) give a cohort of
//! every token launched in the window, including all that died: no
//! survivorship filter is possible by construction. Failed transactions are
//! decoded too; they are the evidence for execution-failure rates.

use crate::decode::decode_slot;
use crate::events::Event;
use crate::store::write_events;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const CHUNK: u64 = 1000;

/// JSON-RPC codes meaning "no block in this slot" (leader skipped it).
const SKIPPED_CODES: [i64; 2] = [-32007, -32009];

pub enum SlotResult {
    Block(Value),
    Skipped,
    /// Pruned from the node's ledger: a permanent hole, recorded in the manifest.
    Missing,
}

pub struct Rpc {
    url: String,
    /// "finalized" for history (no forks); "confirmed" for the live path.
    pub commitment: String,
    client: reqwest::blocking::Client,
}

impl Rpc {
    pub fn new(url: &str) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .gzip(true)
            .build()?;
        Ok(Rpc {
            url: url.to_string(),
            commitment: "finalized".into(),
            client,
        })
    }

    pub fn call(&self, method: &str, params: Value) -> Result<Value> {
        let body = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
        let r = self.client.post(&self.url).json(&body).send()?;
        if r.status().as_u16() == 429 {
            bail!("429 rate limited");
        }
        Ok(serde_json::from_slice(&r.bytes()?)?)
    }

    /// Fetch one block, retrying transient failures with capped exponential backoff.
    pub fn get_block(&self, slot: u64) -> SlotResult {
        let mut backoff = 1.0f64;
        loop {
            let params = json!([slot, {
                "maxSupportedTransactionVersion": 1, "encoding": "json",
                "transactionDetails": "full", "rewards": false, "commitment": self.commitment}]);
            match self.call("getBlock", params) {
                Ok(mut j) => {
                    if let Some(err) = j.get("error") {
                        let code = err["code"].as_i64().unwrap_or(0);
                        let msg = err["message"].as_str().unwrap_or("");
                        if SKIPPED_CODES.contains(&code) {
                            return SlotResult::Skipped;
                        }
                        if msg.contains("cleaned up") {
                            return SlotResult::Missing;
                        }
                        eprintln!("[retry] slot {slot}: rpc error {code}: {msg}");
                    } else if !j["result"].is_null() {
                        return SlotResult::Block(j["result"].take());
                    }
                }
                Err(e) => eprintln!("[retry] slot {slot}: {e}"),
            }
            std::thread::sleep(Duration::from_secs_f64(backoff));
            backoff = (backoff * 2.0).min(30.0);
        }
    }
}

/// Wrap a getBlock result in the slot-record shape `decode_slot` expects,
/// numbering every transaction by its position in the block.
pub fn slot_record(slot: u64, mut block: Value) -> Value {
    let mut txs = block["transactions"].take();
    if let Some(arr) = txs.as_array_mut() {
        for (i, tx) in arr.iter_mut().enumerate() {
            tx["idx"] = json!(i);
        }
    }
    json!({"slot": slot, "blockTime": block["blockTime"], "txs": txs})
}

pub fn chunk_path(out: &Path, start: u64) -> PathBuf {
    out.join(format!("events_{start:012}.parquet"))
}

pub struct ChunkStats {
    pub slots: u64,
    pub skipped: u64,
    pub missing: u64,
    pub events: usize,
}

/// Fetch and decode slots [lo, hi) with `workers` threads; returns events in chain order.
pub fn fetch_range(rpc: &Rpc, lo: u64, hi: u64, workers: usize) -> (Vec<Event>, ChunkStats) {
    let next = AtomicU64::new(lo);
    let per_slot: Mutex<Vec<(u64, Vec<Event>)>> = Mutex::new(Vec::new());
    let skipped = AtomicU64::new(0);
    let missing = AtomicU64::new(0);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let slot = next.fetch_add(1, Ordering::SeqCst);
                if slot >= hi {
                    return;
                }
                let evs = match rpc.get_block(slot) {
                    SlotResult::Block(b) => {
                        let mut v = Vec::new();
                        decode_slot(&slot_record(slot, b), &mut v);
                        v
                    }
                    SlotResult::Skipped => {
                        skipped.fetch_add(1, Ordering::Relaxed);
                        Vec::new()
                    }
                    SlotResult::Missing => {
                        missing.fetch_add(1, Ordering::Relaxed);
                        Vec::new()
                    }
                };
                per_slot.lock().unwrap().push((slot, evs));
            });
        }
    });
    let mut per_slot = per_slot.into_inner().unwrap();
    per_slot.sort_by_key(|(s, _)| *s);
    let events: Vec<Event> = per_slot.into_iter().flat_map(|(_, v)| v).collect();
    let stats = ChunkStats {
        slots: hi - lo,
        skipped: skipped.into_inner(),
        missing: missing.into_inner(),
        events: events.len(),
    };
    (events, stats)
}

/// Resumable chunked backfill. Existing chunk files are skipped; each chunk is
/// written to a temp file and renamed, so a crash never leaves a partial chunk.
pub fn run(url: &str, out: &Path, start: u64, end: u64, workers: usize) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let rpc = Rpc::new(url)?;
    let first = start - start % CHUNK;
    let manifest = out.join("manifest.jsonl");
    for c in (first..end).step_by(CHUNK as usize) {
        let path = chunk_path(out, c);
        if path.exists() {
            continue;
        }
        let (lo, hi) = (c.max(start), (c + CHUNK).min(end));
        let t0 = Instant::now();
        let (events, st) = fetch_range(&rpc, lo, hi, workers);
        let tmp = path.with_extension("tmp");
        write_events(&tmp, &events)?;
        std::fs::rename(&tmp, &path)?;
        let dt = t0.elapsed().as_secs_f64();
        let line = json!({"chunk": c, "lo": lo, "hi": hi, "skipped": st.skipped, "missing": st.missing,
                          "events": st.events, "secs": dt, "bytes": std::fs::metadata(&path)?.len()});
        use std::io::Write;
        let mut m = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&manifest)?;
        writeln!(m, "{line}")?;
        println!(
            "chunk {c}: {} slots ({} skipped, {} missing), {} events, {:.0}s ({:.2} slots/s), {:.1} MB",
            st.slots,
            st.skipped,
            st.missing,
            st.events,
            dt,
            st.slots as f64 / dt,
            std::fs::metadata(&path)?.len() as f64 / 1e6
        );
    }
    Ok(())
}
