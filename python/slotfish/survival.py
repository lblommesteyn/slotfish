"""Token survival: graduation, liquidity collapse, extreme drawdown.

Landmark design (avoids immortal-time bias and lookahead):
  * Cohort: every non-Mayhem token launched in the window.
  * Landmark: LANDMARK_S seconds after launch. Tokens that already hit an
    endpoint before the landmark are excluded *for that endpoint*; every other
    token is at risk from the landmark, including tokens with zero trades
    (they get zero-activity features, not dropped).
  * Covariates: the token's state at the landmark, taken from the last
    point-in-time feature row with age <= landmark (Rust dataset), else zeros.
  * Time-to-event is measured from the landmark, right-censored at the end of
    data.

Endpoints:
  graduation   pump.fun Complete event
  collapse     real SOL reserves fall below COLLAPSE_FRAC of their running
               peak, after the peak reached COLLAPSE_MIN_SOL
  drawdown50/80  price falls 50% / 80% below its post-landmark running peak.
               The curve's 30 SOL virtual floor means price never drops below
               the launch price, so -80% is only reachable after a >=5x pump.

Models: Kaplan-Meier curves, a Cox model (statsmodels PHReg) on a few
interpretable covariates, and a LightGBM classifier for "event within
WINDOW_S of the landmark", trained on early launches and tested on later ones
(chronological split by launch slot).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.duration.hazard_regression import PHReg

LANDMARK_S = 60.0
WINDOW_S = 600.0
COLLAPSE_FRAC = 0.2
COLLAPSE_MIN_SOL = 1.0
DD_LEVELS = {"drawdown50": 0.5, "drawdown80": 0.2}
COX_COVARIATES = ["log_n_trades", "n_buyers_log", "progress", "creator_sold", "launch_snipers_log", "ret_300"]
CLF_FEATURES = [
    "n_trades", "n_buyers", "progress", "log_mcap_sol", "creator_sold", "creator_hold_frac", "launch_snipers",
    "netflow_75_sol", "netflow_300_sol", "buys_75", "sells_75", "uniq_buyers_75", "ret_75", "ret_300",
    "drawdown_from_peak", "skill_wavg_75", "smart_buys_75",
]
SEED = 20261012


def kaplan_meier(t: np.ndarray, e: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """S(t) on `grid` (product-limit estimator)."""
    order = np.argsort(t)
    t, e = t[order], e[order]
    uniq = np.unique(t[e == 1])
    s, surv_at = 1.0, []
    at_risk = len(t)
    idx = 0
    steps = []
    for u in uniq:
        while idx < len(t) and t[idx] < u:
            at_risk -= 1
            idx += 1
        d = int(((t == u) & (e == 1)).sum())
        s *= 1 - d / max(at_risk, 1)
        steps.append((u, s))
    for g in grid:
        v = 1.0
        for u, sv in steps:
            if u <= g:
                v = sv
            else:
                break
        surv_at.append(v)
    return np.array(surv_at)


def token_table(events_dir: Path) -> tuple[pl.DataFrame, float, int]:
    ev = pl.scan_parquet(str(events_dir / "events_*.parquet"))
    bounds = ev.select(pl.col("slot").min().alias("s0"), pl.col("slot").max().alias("s1"),
                       pl.col("block_time").min().alias("b0"), pl.col("block_time").max().alias("b1")).collect()
    s0, s1, b0, b1 = bounds.row(0)
    slot_secs = (b1 - b0) / (s1 - s0)
    creates = (ev.filter((pl.col("kind") == 0) & ~pl.col("is_buy"))
               .select(pl.col("mint"), pl.col("slot").alias("create_slot")).collect())
    trades = (ev.filter(pl.col("kind") == 1).select("mint", "slot", "tx_idx", "ev_idx", "vsol", "vtok", "rsol", "fee_bps")
              .join(creates.lazy(), on="mint").sort(["mint", "slot", "tx_idx", "ev_idx"]).collect())
    # Tokens touched by zero-fee (Mayhem-agent) trades are excluded entirely here:
    # their reserves are not reconstructible, so endpoints would be unreliable.
    tainted = trades.filter(pl.col("fee_bps") == 0)["mint"].unique()
    trades = trades.filter(~pl.col("mint").is_in(tainted.implode()))
    creates = creates.filter(~pl.col("mint").is_in(tainted.implode()))
    complete = ev.filter(pl.col("kind") == 2).select("mint", pl.col("slot").alias("grad_slot")).collect()
    t = trades.with_columns(
        price=pl.col("vsol") / pl.col("vtok"),
        peak_rsol=pl.col("rsol").cum_max().over("mint"),
    ).with_columns(peak_price=pl.col("price").cum_max().over("mint"))
    collapse = (t.filter((pl.col("peak_rsol") >= COLLAPSE_MIN_SOL * 1e9) & (pl.col("rsol") < COLLAPSE_FRAC * pl.col("peak_rsol")))
                .group_by("mint").agg(pl.col("slot").min().alias("collapse_slot")))
    landmark = int(LANDMARK_S / slot_secs)
    # Drawdown measured on the post-landmark path only.
    tl = t.filter(pl.col("slot") >= pl.col("create_slot") + landmark).with_columns(
        peak_after=pl.col("price").cum_max().over("mint"))
    dds = []
    for name, frac in DD_LEVELS.items():
        dds.append(tl.filter(pl.col("price") < frac * pl.col("peak_after"))
                   .group_by("mint").agg(pl.col("slot").min().alias(f"{name}_slot")))
    tok = creates.join(complete, on="mint", how="left").join(collapse, on="mint", how="left")
    for d in dds:
        tok = tok.join(d, on="mint", how="left")
    tok = tok.with_columns(landmark_slot=pl.col("create_slot") + landmark)
    return tok, slot_secs, int(s1)


def landmark_features(tok: pl.DataFrame, dataset: Path) -> pl.DataFrame:
    ds = pl.read_parquet(dataset, columns=["slot", "mint"] + CLF_FEATURES)
    j = ds.join(tok.select("mint", "landmark_slot"), on="mint").filter(pl.col("slot") <= pl.col("landmark_slot"))
    last = j.sort("slot").group_by("mint").last().drop("slot", "landmark_slot")
    out = tok.join(last, on="mint", how="left")
    # No trade before the landmark = zero activity (not missing): keep these tokens.
    fill = {c: 0.0 for c in CLF_FEATURES if c not in ("skill_wavg_75",)}
    out = out.with_columns([pl.col(c).fill_null(v) for c, v in fill.items()])
    return out.with_columns(
        log_n_trades=(pl.col("n_trades") + 1).log(),
        n_buyers_log=(pl.col("n_buyers") + 1).log(),
        launch_snipers_log=(pl.col("launch_snipers") + 1).log(),
    )


def endpoint_frame(df: pl.DataFrame, col: str, end_slot: int, slot_secs: float) -> pl.DataFrame:
    """At-risk set and (time, event) from the landmark for one endpoint."""
    d = df.filter(pl.col("landmark_slot") <= end_slot)
    d = d.filter(pl.col(col).is_null() | (pl.col(col) > pl.col("landmark_slot")))
    return d.with_columns(
        event=pl.col(col).is_not_null().cast(pl.Int8),
        time_s=((pl.when(pl.col(col).is_not_null()).then(pl.col(col)).otherwise(end_slot) - pl.col("landmark_slot"))
                * slot_secs),
    )


def run(events_dir: Path, dataset: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, slot_secs, end_slot = token_table(events_dir)
    df = landmark_features(tok, dataset)
    res: dict = {"cohort_tokens": tok.height, "slot_secs": slot_secs, "landmark_s": LANDMARK_S,
                 "window_s": WINDOW_S, "endpoints": {}}
    grid = np.array([60, 300, 600, 1800, 3600, 3 * 3600, 6 * 3600])
    endpoints = [("graduation", "grad_slot"), ("collapse", "collapse_slot")] + [(n, f"{n}_slot") for n in DD_LEVELS]
    for name, col in endpoints:
        d = endpoint_frame(df, col, end_slot, slot_secs)
        t, e = d["time_s"].to_numpy(), d["event"].to_numpy()
        er: dict = {"at_risk": int(len(d)), "events": int(e.sum()),
                    "km_cum_incidence": dict(zip([f"{g}s" for g in grid], (1 - kaplan_meier(t, e, grid)).round(4).tolist()))}
        # Cox PH on interpretable covariates (standardized).
        X = d.select(COX_COVARIATES).fill_null(0.0).to_numpy()
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        try:
            ph = PHReg(t, X, status=e, ties="efron").fit()
            er["cox"] = {c: {"hr_per_sd": float(np.exp(b)), "p": float(p)}
                         for c, b, p in zip(COX_COVARIATES, ph.params, ph.pvalues)}
        except Exception as exc:  # noqa: BLE001
            er["cox_error"] = str(exc)
        # Classifier: event within WINDOW_S of landmark; only tokens observed for the full window.
        full = d.filter(pl.col("landmark_slot") + int(WINDOW_S / slot_secs) <= end_slot).sort("create_slot")
        yw = ((full["event"] == 1) & (full["time_s"] <= WINDOW_S)).to_numpy().astype(int)
        cut = int(0.7 * full.height)
        if full.height > 200 and 0 < yw[:cut].sum() and 0 < yw[cut:].sum() < len(yw[cut:]):
            Xc = full.select(CLF_FEATURES).to_numpy().astype(np.float32)
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                                   verbose=-1, deterministic=True, force_row_wise=True)
            m.fit(Xc[:cut], yw[:cut])
            p = m.predict_proba(Xc[cut:])[:, 1]
            yt = yw[cut:]
            base = yw[:cut].mean()
            bins = np.quantile(p, np.linspace(0, 1, 6))
            calib = []
            for k in range(5):
                mk = (p >= bins[k]) & ((p <= bins[k + 1]) if k == 4 else (p < bins[k + 1]))
                if mk.sum():
                    calib.append({"pred": float(p[mk].mean()), "obs": float(yt[mk].mean()), "n": int(mk.sum())})
            er["classifier_10m"] = {
                "train_n": cut, "test_n": int(len(yt)), "test_rate": float(yt.mean()),
                "auc": float(roc_auc_score(yt, p)), "brier": float(brier_score_loss(yt, p)),
                "brier_baseline": float(brier_score_loss(yt, np.full_like(p, base))),
                "calibration": calib,
            }
        res["endpoints"][name] = er
        print(f"[{name}] at_risk={er['at_risk']} events={er['events']} clf={er.get('classifier_10m', {}).get('auc')}", flush=True)
    (out_dir / "survival.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=Path("data/events"))
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/survival"))
    a = ap.parse_args()
    run(a.events, a.dataset, a.out)
