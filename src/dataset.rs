//! Supervised dataset: one row per decision point (cohort mint, slot) with
//! point-in-time features and *executable* labels: the net return of a real
//! round trip through the execution model at each horizon, not a mid-price
//! move. Labels deliberately use p_fail = 0 (failures are a cost the backtest
//! charges, not something a model should learn to predict).

use crate::amm::Pool;
use crate::engine::next_same_mint_in_slot;
use crate::events::Kind;
use crate::exec::{round_trip, ExecConfig};
use crate::features::{FeatureEngine, FEATURE_NAMES};
use crate::market::Timeline;
use crate::store::Tape;
use anyhow::Result;
use arrow::array::{
    ArrayRef, BooleanBuilder, Float32Builder, Int64Builder, StringBuilder, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::path::Path;
use std::sync::Arc;

/// Prediction horizons in seconds; converted to slots with the measured slot time.
pub const HORIZONS: [(f64, &str); 3] = [(30.0, "30s"), (120.0, "2m"), (600.0, "10m")];

pub fn horizon_slots(secs: f64, slot_secs: f64) -> u64 {
    (secs / slot_secs).round() as u64
}

pub fn build(
    t: &Tape,
    tl: &Timeline,
    cfg: &ExecConfig,
    slot_secs: f64,
    out: &Path,
) -> Result<usize> {
    let mut cfg = cfg.clone();
    cfg.p_fail = 0.0;
    let data_end = *t.slot.last().unwrap_or(&0);
    let mut fe = FeatureEngine::new(horizon_slots(120.0, slot_secs));
    let hslots: Vec<u64> = HORIZONS
        .iter()
        .map(|(s, _)| horizon_slots(*s, slot_secs))
        .collect();
    let nf = FEATURE_NAMES.len();
    let mut fields = vec![
        Field::new("slot", DataType::UInt64, false),
        Field::new("block_time", DataType::Int64, false),
        Field::new("mint", DataType::Utf8, false),
    ];
    for n in FEATURE_NAMES {
        fields.push(Field::new(*n, DataType::Float32, true));
    }
    for (_, h) in HORIZONS {
        fields.push(Field::new(format!("ret_net_{h}"), DataType::Float32, true));
        fields.push(Field::new(format!("ret_mid_{h}"), DataType::Float32, true));
        fields.push(Field::new(format!("entered_{h}"), DataType::Boolean, false));
    }
    let schema = Arc::new(Schema::new(fields));
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut w = ArrowWriter::try_new(std::fs::File::create(out)?, schema.clone(), Some(props))?;

    let mut feats = Vec::with_capacity(nf);
    let mut rows = 0usize;
    // Column buffers, flushed every BATCH rows.
    const BATCH: usize = 1 << 16;
    let mut slot_b = UInt64Builder::new();
    let mut bt_b = Int64Builder::new();
    let mut mint_b = StringBuilder::new();
    let mut f_b: Vec<Float32Builder> = (0..nf).map(|_| Float32Builder::new()).collect();
    let mut l_b: Vec<(Float32Builder, Float32Builder, BooleanBuilder)> = HORIZONS
        .iter()
        .map(|_| {
            (
                Float32Builder::new(),
                Float32Builder::new(),
                BooleanBuilder::new(),
            )
        })
        .collect();
    let mut pending = 0usize;

    let mut flush = |slot_b: &mut UInt64Builder,
                     bt_b: &mut Int64Builder,
                     mint_b: &mut StringBuilder,
                     f_b: &mut Vec<Float32Builder>,
                     l_b: &mut Vec<(Float32Builder, Float32Builder, BooleanBuilder)>|
     -> Result<()> {
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(slot_b.finish()),
            Arc::new(bt_b.finish()),
            Arc::new(mint_b.finish()),
        ];
        for b in f_b.iter_mut() {
            cols.push(Arc::new(b.finish()));
        }
        for (a, b, c) in l_b.iter_mut() {
            cols.push(Arc::new(a.finish()));
            cols.push(Arc::new(b.finish()));
            cols.push(Arc::new(c.finish()));
        }
        w.write(&RecordBatch::try_new(schema.clone(), cols)?)?;
        Ok(())
    };

    for i in 0..t.len() {
        fe.update(t, tl, i);
        let m = t.mint[i];
        let slot = t.slot[i];
        let info = &tl.info[m as usize];
        if t.kind[i] != Kind::Trade || !info.tradable_at(slot) {
            continue;
        }
        if i + 1 < t.len() && t.slot[i + 1] == slot && next_same_mint_in_slot(t, i) {
            continue;
        }
        fe.features(t, tl, i, &mut feats);
        slot_b.append_value(slot);
        bt_b.append_value(t.block_time[i]);
        mint_b.append_value(&t.mints.names[m as usize]);
        for (b, v) in f_b.iter_mut().zip(feats.iter()) {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(*v);
            }
        }
        let quote = Pool {
            sol: t.vsol[i],
            tok: t.vtok[i],
            fee_bps: t.fee_bps[i],
        };
        for (k, h) in hslots.iter().enumerate() {
            let rt = round_trip(tl, &cfg, m, slot, quote, *h, data_end);
            let (net, mid, ent) = &mut l_b[k];
            if rt.entered && !rt.censored {
                net.append_value(rt.ret(cfg.size_lamports) as f32);
                let p0 = tl.state_at_end_of(m, rt.entry_slot).unwrap().0.price();
                let p1 = tl.state_at_end_of(m, rt.exit_slot).unwrap().0.price();
                mid.append_value((p1 / p0).ln() as f32);
            } else {
                net.append_null();
                mid.append_null();
            }
            ent.append_value(rt.entered);
        }
        rows += 1;
        pending += 1;
        if pending == BATCH {
            flush(&mut slot_b, &mut bt_b, &mut mint_b, &mut f_b, &mut l_b)?;
            pending = 0;
        }
    }
    if pending > 0 {
        flush(&mut slot_b, &mut bt_b, &mut mint_b, &mut f_b, &mut l_b)?;
    }
    w.close()?;
    Ok(rows)
}
