//! Parquet IO. Events are written sorted by (slot, tx_idx, ev_idx) in row
//! groups of ~1M rows; readers load them into a columnar, interned
//! [`Tape`] that the replay engine iterates without allocation.

use crate::events::{Event, Kind};
use anyhow::{Context, Result};
use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Int64Array, Int64Builder, StringArray,
    StringBuilder, UInt16Array, UInt16Builder, UInt32Array, UInt32Builder, UInt64Array,
    UInt64Builder, UInt8Array, UInt8Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub fn schema() -> Arc<Schema> {
    use DataType::*;
    Arc::new(Schema::new(vec![
        Field::new("slot", UInt64, false),
        Field::new("block_time", Int64, false),
        Field::new("tx_idx", UInt32, false),
        Field::new("ev_idx", UInt16, false),
        Field::new("kind", UInt8, false),
        Field::new("mint", Utf8, false),
        Field::new("user", Utf8, false),
        Field::new("pool", Utf8, false),
        Field::new("symbol", Utf8, false),
        Field::new("is_buy", Boolean, false),
        Field::new("sol_amount", UInt64, false),
        Field::new("token_amount", UInt64, false),
        Field::new("user_sol", UInt64, false),
        Field::new("vsol", UInt64, false),
        Field::new("vtok", UInt64, false),
        Field::new("rsol", UInt64, false),
        Field::new("rtok", UInt64, false),
        Field::new("fee", UInt64, false),
        Field::new("fee_bps", UInt32, false),
        Field::new("ts", Int64, false),
        Field::new("tx_fee", UInt64, false),
        Field::new("priority_fee", UInt64, false),
        Field::new("cu", UInt64, false),
    ]))
}

fn to_batch(evs: &[Event]) -> Result<RecordBatch> {
    macro_rules! col {
        ($b:ty, $f:ident) => {{
            let mut b = <$b>::with_capacity(evs.len());
            for e in evs {
                b.append_value(e.$f);
            }
            Arc::new(b.finish()) as ArrayRef
        }};
    }
    macro_rules! scol {
        ($f:ident) => {{
            let mut b = StringBuilder::new();
            for e in evs {
                b.append_value(&e.$f);
            }
            Arc::new(b.finish()) as ArrayRef
        }};
    }
    let mut kind = UInt8Builder::with_capacity(evs.len());
    for e in evs {
        kind.append_value(e.kind as u8);
    }
    let cols: Vec<ArrayRef> = vec![
        col!(UInt64Builder, slot),
        col!(Int64Builder, block_time),
        col!(UInt32Builder, tx_idx),
        col!(UInt16Builder, ev_idx),
        Arc::new(kind.finish()),
        scol!(mint),
        scol!(user),
        scol!(pool),
        scol!(symbol),
        col!(BooleanBuilder, is_buy),
        col!(UInt64Builder, sol_amount),
        col!(UInt64Builder, token_amount),
        col!(UInt64Builder, user_sol),
        col!(UInt64Builder, vsol),
        col!(UInt64Builder, vtok),
        col!(UInt64Builder, rsol),
        col!(UInt64Builder, rtok),
        col!(UInt64Builder, fee),
        col!(UInt32Builder, fee_bps),
        col!(Int64Builder, ts),
        col!(UInt64Builder, tx_fee),
        col!(UInt64Builder, priority_fee),
        col!(UInt64Builder, cu),
    ];
    Ok(RecordBatch::try_new(schema(), cols)?)
}

pub fn write_events(path: &Path, evs: &[Event]) -> Result<()> {
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .set_max_row_group_size(1 << 20)
        .build();
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = ArrowWriter::try_new(f, schema(), Some(props))?;
    for chunk in evs.chunks(1 << 16) {
        w.write(&to_batch(chunk)?)?;
    }
    w.close()?;
    Ok(())
}

/// String interner: base58 address -> dense u32 id.
#[derive(Default, Debug)]
pub struct Interner {
    pub map: HashMap<String, u32>,
    pub names: Vec<String>,
}

impl Interner {
    pub fn id(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.map.get(s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(s.to_string());
        self.map.insert(s.to_string(), i);
        i
    }
}

/// Columnar, interned event tape: what the replay engine consumes.
#[derive(Default, Debug)]
pub struct Tape {
    pub slot: Vec<u64>,
    pub block_time: Vec<i64>,
    pub tx_idx: Vec<u32>,
    pub kind: Vec<Kind>,
    pub mint: Vec<u32>,
    pub user: Vec<u32>,
    pub pool: Vec<u32>,
    pub is_buy: Vec<bool>,
    pub sol_amount: Vec<u64>,
    pub token_amount: Vec<u64>,
    pub vsol: Vec<u64>,
    pub vtok: Vec<u64>,
    pub rsol: Vec<u64>,
    pub fee_bps: Vec<u32>,
    pub priority_fee: Vec<u64>,
    pub mints: Interner,
    pub users: Interner,
    pub pools: Interner,
}

impl Tape {
    /// The first `n` events as an independent tape (interners cloned whole, so
    /// ids are identical). Used by the lookahead test.
    pub fn prefix(&self, n: usize) -> Tape {
        Tape {
            slot: self.slot[..n].to_vec(),
            block_time: self.block_time[..n].to_vec(),
            tx_idx: self.tx_idx[..n].to_vec(),
            kind: self.kind[..n].to_vec(),
            mint: self.mint[..n].to_vec(),
            user: self.user[..n].to_vec(),
            pool: self.pool[..n].to_vec(),
            is_buy: self.is_buy[..n].to_vec(),
            sol_amount: self.sol_amount[..n].to_vec(),
            token_amount: self.token_amount[..n].to_vec(),
            vsol: self.vsol[..n].to_vec(),
            vtok: self.vtok[..n].to_vec(),
            rsol: self.rsol[..n].to_vec(),
            fee_bps: self.fee_bps[..n].to_vec(),
            priority_fee: self.priority_fee[..n].to_vec(),
            mints: Interner {
                map: self.mints.map.clone(),
                names: self.mints.names.clone(),
            },
            users: Interner {
                map: self.users.map.clone(),
                names: self.users.names.clone(),
            },
            pools: Interner {
                map: self.pools.map.clone(),
                names: self.pools.names.clone(),
            },
        }
    }

    pub fn len(&self) -> usize {
        self.slot.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slot.is_empty()
    }
    /// Approximate resident bytes of the columnar arrays (excludes interner strings).
    pub fn column_bytes(&self) -> usize {
        self.len() * (8 + 8 + 4 + 1 + 4 + 4 + 4 + 1 + 8 + 8 + 8 + 8 + 8 + 4 + 8)
    }
}

/// Read one parquet file, or every `events_*.parquet` in a directory (chunk
/// names sort in slot order, so concatenation preserves chain order).
pub fn read_tape(path: &Path) -> Result<Tape> {
    let mut files: Vec<std::path::PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    let mut t = Tape::default();
    for f in &files {
        read_into(f, &mut t)?;
    }
    for w in t.slot.windows(2) {
        anyhow::ensure!(w[0] <= w[1], "events out of slot order");
    }
    Ok(t)
}

fn read_into(path: &Path, t: &mut Tape) -> Result<()> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(f)?
        .with_batch_size(1 << 16)
        .build()?;
    for batch in reader {
        let b = batch?;
        macro_rules! get {
            ($name:literal, $ty:ty) => {
                b.column_by_name($name)
                    .context($name)?
                    .as_any()
                    .downcast_ref::<$ty>()
                    .context($name)?
            };
        }
        let slot = get!("slot", UInt64Array);
        let bt = get!("block_time", Int64Array);
        let txi = get!("tx_idx", UInt32Array);
        let kind = get!("kind", UInt8Array);
        let mint = get!("mint", StringArray);
        let user = get!("user", StringArray);
        let pool = get!("pool", StringArray);
        let isb = get!("is_buy", BooleanArray);
        let sol = get!("sol_amount", UInt64Array);
        let tok = get!("token_amount", UInt64Array);
        let vsol = get!("vsol", UInt64Array);
        let vtok = get!("vtok", UInt64Array);
        let rsol = get!("rsol", UInt64Array);
        let fbps = get!("fee_bps", UInt32Array);
        let pf = get!("priority_fee", UInt64Array);
        let _ = get!("ev_idx", UInt16Array);
        for i in 0..b.num_rows() {
            t.slot.push(slot.value(i));
            t.block_time.push(bt.value(i));
            t.tx_idx.push(txi.value(i));
            t.kind.push(Kind::from_u8(kind.value(i)));
            t.mint.push(t.mints.id(mint.value(i)));
            t.user.push(t.users.id(user.value(i)));
            t.pool.push(t.pools.id(pool.value(i)));
            t.is_buy.push(isb.value(i));
            t.sol_amount.push(sol.value(i));
            t.token_amount.push(tok.value(i));
            t.vsol.push(vsol.value(i));
            t.vtok.push(vtok.value(i));
            t.rsol.push(rsol.value(i));
            t.fee_bps.push(fbps.value(i));
            t.priority_fee.push(pf.value(i));
        }
        debug_assert!(slot.len() == b.num_rows());
    }
    Ok(())
}
