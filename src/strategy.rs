//! Strategy plugins. Entries only: exits are the engine's fixed-horizon rule,
//! so every strategy is measured on the same holding period.

use crate::engine::{Ctx, Strategy};
use crate::exec::uniform;
use crate::features::FEATURE_NAMES;
use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, StringArray, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::path::Path;

/// Baseline: never trades.
pub struct NoTrade;
impl Strategy for NoTrade {
    fn name(&self) -> String {
        "no_trade".into()
    }
    fn on_decision(&mut self, _: &Ctx) -> bool {
        false
    }
}

/// Baseline: enters at a random decision point with probability `p`.
pub struct RandomEntry {
    pub p: f64,
    pub seed: u64,
}
impl Strategy for RandomEntry {
    fn name(&self) -> String {
        format!("random_p{}", self.p)
    }
    fn on_decision(&mut self, c: &Ctx) -> bool {
        uniform(self.seed, c.i as u64, 0xAB) < self.p
    }
}

/// Baseline: simple momentum. Enter when the last `window` slots had at least
/// `min_buys` buys and the price rose at least `min_ret` (log) over the window.
pub struct Momentum {
    pub min_ret: f32,
    pub min_buys: f32,
}
impl Strategy for Momentum {
    fn name(&self) -> String {
        format!("momentum_r{}_b{}", self.min_ret, self.min_buys)
    }
    fn needs_features(&self) -> bool {
        true
    }
    fn on_decision(&mut self, c: &Ctx) -> bool {
        let f = |n: &str| c.features[FEATURE_NAMES.iter().position(|x| *x == n).unwrap()];
        f("ret_75") >= self.min_ret && f("buys_75") >= self.min_buys
    }
}

/// Model scores produced offline (Python), keyed by (slot, mint). Enters when
/// score >= threshold. Scores must have been computed from features at that
/// same decision point; the dataset/signal pipeline guarantees the keys match.
pub struct Signal {
    pub label: String,
    pub threshold: f64,
    pub scores: HashMap<(u64, u32), f64>,
}

impl Signal {
    pub fn load(
        path: &Path,
        mints: &crate::store::Interner,
        threshold: f64,
        label: String,
    ) -> Result<Signal> {
        let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let r = ParquetRecordBatchReaderBuilder::try_new(f)?.build()?;
        let mut scores = HashMap::new();
        for b in r {
            let b = b?;
            let slot = b
                .column_by_name("slot")
                .context("slot")?
                .as_any()
                .downcast_ref::<UInt64Array>()
                .context("slot u64")?;
            let mint = b.column_by_name("mint").context("mint")?;
            let mint = mint
                .as_any()
                .downcast_ref::<StringArray>()
                .context("mint utf8")?;
            let score = b
                .column_by_name("score")
                .context("score")?
                .as_any()
                .downcast_ref::<Float64Array>()
                .context("score f64")?;
            for i in 0..b.num_rows() {
                if let Some(&m) = mints.map.get(mint.value(i)) {
                    scores.insert((slot.value(i), m), score.value(i));
                }
            }
        }
        Ok(Signal {
            label,
            threshold,
            scores,
        })
    }
}

impl Strategy for Signal {
    fn name(&self) -> String {
        format!("signal_{}_t{}", self.label, self.threshold)
    }
    fn on_decision(&mut self, c: &Ctx) -> bool {
        let key = (c.tape.slot[c.i], c.tape.mint[c.i]);
        self.scores.get(&key).is_some_and(|&s| s >= self.threshold)
    }
}
