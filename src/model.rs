//! Scoring of models trained in Python, so the live paper trader runs the
//! exact research model without a Python process in the loop.
//!
//! Format: JSON written by `python/slotfish/research.py`:
//!   {"kind": "lgbm", "features": [...], "dump": <booster.dump_model()>}
//!   {"kind": "ridge", "features": [...], "median": [...], "mu": [...], "sd": [...],
//!    "coef": [...], "intercept": x}
//! `verify` compares against predictions Python saved next to the model.

use crate::features::FEATURE_NAMES;
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

pub enum Node {
    Split {
        feature: usize,
        threshold: f64,
        default_left: bool,
        /// LightGBM missing_type: 0 None (NaN -> 0.0), 1 Zero, 2 NaN.
        missing: u8,
        left: Box<Node>,
        right: Box<Node>,
    },
    Leaf(f64),
}

impl Node {
    fn parse(v: &Value) -> Result<Node> {
        if let Some(leaf) = v.get("leaf_value") {
            return Ok(Node::Leaf(leaf.as_f64().context("leaf_value")?));
        }
        if v["decision_type"].as_str() != Some("<=") {
            bail!("unsupported decision_type {}", v["decision_type"]);
        }
        let missing = match v["missing_type"].as_str() {
            Some("None") => 0,
            Some("Zero") => 1,
            Some("NaN") => 2,
            m => bail!("unknown missing_type {m:?}"),
        };
        Ok(Node::Split {
            feature: v["split_feature"].as_u64().context("split_feature")? as usize,
            threshold: v["threshold"].as_f64().context("threshold")?,
            default_left: v["default_left"].as_bool().unwrap_or(true),
            missing,
            left: Box::new(Node::parse(&v["left_child"])?),
            right: Box::new(Node::parse(&v["right_child"])?),
        })
    }

    fn eval(&self, x: &[f64]) -> f64 {
        let mut n = self;
        loop {
            match n {
                Node::Leaf(v) => return *v,
                Node::Split {
                    feature,
                    threshold,
                    default_left,
                    missing,
                    left,
                    right,
                } => {
                    let mut v = x[*feature];
                    let is_missing = match missing {
                        2 => v.is_nan(),
                        1 => v.is_nan() || v == 0.0,
                        _ => {
                            if v.is_nan() {
                                v = 0.0;
                            }
                            false
                        }
                    };
                    let go_left = if is_missing {
                        *default_left
                    } else {
                        v <= *threshold
                    };
                    n = if go_left { left } else { right };
                }
            }
        }
    }
}

pub enum Model {
    Gbdt {
        trees: Vec<Node>,
    },
    Ridge {
        median: Vec<f64>,
        mu: Vec<f64>,
        sd: Vec<f64>,
        coef: Vec<f64>,
        intercept: f64,
    },
}

pub struct Scorer {
    pub model: Model,
    /// Index into the engine's FEATURE_NAMES for each model input.
    map: Vec<usize>,
    buf: std::cell::RefCell<Vec<f64>>,
}

impl Scorer {
    pub fn load(path: &Path) -> Result<Scorer> {
        let v: Value = serde_json::from_slice(
            &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
        )?;
        let names: Vec<String> = serde_json::from_value(v["features"].clone())?;
        let map = names
            .iter()
            .map(|n| {
                FEATURE_NAMES
                    .iter()
                    .position(|f| f == n)
                    .with_context(|| format!("unknown feature {n}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let f64s = |k: &str| -> Result<Vec<f64>> { Ok(serde_json::from_value(v[k].clone())?) };
        let model = match v["kind"].as_str() {
            Some("lgbm") => {
                let trees = v["dump"]["tree_info"]
                    .as_array()
                    .context("tree_info")?
                    .iter()
                    .map(|t| Node::parse(&t["tree_structure"]))
                    .collect::<Result<Vec<_>>>()?;
                Model::Gbdt { trees }
            }
            Some("ridge") => Model::Ridge {
                median: f64s("median")?,
                mu: f64s("mu")?,
                sd: f64s("sd")?,
                coef: f64s("coef")?,
                intercept: v["intercept"].as_f64().context("intercept")?,
            },
            k => bail!("unknown model kind {k:?}"),
        };
        Ok(Scorer {
            model,
            map,
            buf: Default::default(),
        })
    }

    /// Score one engine feature vector (FEATURE_NAMES order; NaN = missing).
    pub fn score(&self, feats: &[f32]) -> f64 {
        let mut x = self.buf.borrow_mut();
        x.clear();
        // f32 -> f64 exactly as numpy does when LightGBM receives float32 input.
        x.extend(self.map.iter().map(|&k| feats[k] as f64));
        match &self.model {
            Model::Gbdt { trees } => trees.iter().map(|t| t.eval(&x)).sum(),
            Model::Ridge {
                median,
                mu,
                sd,
                coef,
                intercept,
            } => {
                let mut s = *intercept;
                for k in 0..x.len() {
                    let v = if x[k].is_nan() { median[k] } else { x[k] };
                    s += coef[k] * (v - mu[k]) / sd[k];
                }
                s
            }
        }
    }

    /// Check against Python predictions: {"features": [[...]], "pred": [...]} in model input order.
    pub fn verify(&self, path: &Path) -> Result<(usize, f64)> {
        let v: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let rows: Vec<Vec<Option<f64>>> = serde_json::from_value(v["features"].clone())?;
        let preds: Vec<f64> = serde_json::from_value(v["pred"].clone())?;
        let mut worst = 0.0f64;
        let mut feats = vec![f32::NAN; FEATURE_NAMES.len()];
        for (r, p) in rows.iter().zip(preds.iter()) {
            feats.iter_mut().for_each(|f| *f = f32::NAN);
            for (k, val) in r.iter().enumerate() {
                feats[self.map[k]] = val.map(|x| x as f32).unwrap_or(f32::NAN);
            }
            worst = worst.max((self.score(&feats) - p).abs());
        }
        Ok((rows.len(), worst))
    }
}
