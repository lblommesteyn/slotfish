"""Alpha research on the Rust-built dataset.

Pipeline (all choices fixed before looking at test):
  1. Chronological split by slot into train / validation / test with a purge
     gap of one maximum horizon between them, so no training label's exit
     price falls inside the validation period (and likewise val -> test).
  2. Models per horizon on the *executable* net return label:
       ridge   standardized linear regression (interpretable baseline)
       lgbm    LightGBM regressor on a clipped label
  3. Evaluate on test: Spearman IC, AUC of the sign, mean realized net return
     in score buckets, with hour-block bootstrap confidence intervals.
  4. Choose an entry threshold on VALIDATION only (best mean net return with at
     least MIN_VAL_TRADES), write test-period scores as a signal file; the Rust
     engine then backtests it with portfolio limits and random failures.

Feature groups support the ablations and the core question (does wallet
intelligence add anything beyond momentum and order flow?).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

HORIZONS = {"30s": 30.0, "2m": 120.0, "10m": 600.0}

GROUPS: dict[str, list[str]] = {
    "momentum": ["ret_10", "ret_75", "ret_300", "drawdown_from_peak"],
    "orderflow": [
        "buys_10", "sells_10", "buys_75", "sells_75", "netflow_10_sol", "netflow_75_sol",
        "netflow_300_sol", "vol_75_sol", "uniq_buyers_75", "buy_share_75", "slots_since_last",
        "last_size_sol", "last_is_buy", "fails_75", "same_slot_buyers",
    ],
    "token": [
        "age_slots", "log_mcap_sol", "progress", "n_trades", "n_buyers", "creator_sold",
        "creator_hold_frac", "launch_snipers",
    ],
    "wallet": ["skill_last_buyer", "skill_n_last_buyer", "skill_wavg_75", "smart_buys_75", "smart_sol_75",
               "coord_pairs_75", "coord_wallets_75"],
}
ALL = [f for g in GROUPS.values() for f in g]

FEATURE_SETS: dict[str, list[str]] = {
    "all": ALL,
    "no_wallet": [f for f in ALL if f not in GROUPS["wallet"]],
    "momentum_only": GROUPS["momentum"],
    "momentum+wallet": GROUPS["momentum"] + GROUPS["wallet"],
    "no_orderflow": [f for f in ALL if f not in GROUPS["orderflow"]],
    "no_token": [f for f in ALL if f not in GROUPS["token"]],
}

MIN_VAL_TRADES = 30
LABEL_CLIP = (-1.0, 3.0)
SEED = 20261012


@dataclass
class Split:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    bounds: dict


def load(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def slot_secs(df: pl.DataFrame) -> float:
    s0, s1 = df["slot"].min(), df["slot"].max()
    b0, b1 = df["block_time"].min(), df["block_time"].max()
    return (b1 - b0) / (s1 - s0) if s1 > s0 else 0.4


def split(df: pl.DataFrame, fr=(0.6, 0.2), purge_secs: float = 600.0 + 30.0) -> Split:
    s0, s1 = int(df["slot"].min()), int(df["slot"].max())
    span = s1 - s0
    purge = int(purge_secs / slot_secs(df))
    a = s0 + int(span * fr[0])
    b = s0 + int(span * (fr[0] + fr[1]))
    tr = df.filter(pl.col("slot") < a - purge)
    va = df.filter((pl.col("slot") >= a) & (pl.col("slot") < b - purge))
    te = df.filter(pl.col("slot") >= b)
    bounds = {"train": [s0, a - purge], "val": [a, b - purge], "test": [b, s1], "purge_slots": purge}
    return Split(tr, va, te, bounds)


def xy(df: pl.DataFrame, feats: list[str], h: str) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    d = df.filter(pl.col(f"ret_net_{h}").is_not_null())
    x = d.select(feats).to_numpy().astype(np.float32)
    y = d[f"ret_net_{h}"].to_numpy().astype(np.float64)
    return x, y, d


def fit_ridge(x, y):
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    xf = np.where(np.isnan(x), med, x)
    mu, sd = xf.mean(0), xf.std(0) + 1e-9
    m = Ridge(alpha=10.0).fit((xf - mu) / sd, np.clip(y, *LABEL_CLIP))

    def predict(xn):
        xn = np.where(np.isnan(xn), med, xn).astype(np.float64)
        return m.predict((xn - mu) / sd)

    params = {"kind": "ridge", "median": med.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
              "coef": m.coef_.tolist(), "intercept": float(m.intercept_)}
    return predict, params


def fit_lgbm(x, y, xv, yv):
    params = dict(
        objective="huber", alpha=0.5, learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
        seed=SEED, deterministic=True, force_row_wise=True, num_threads=8,
    )
    dt = lgb.Dataset(x, np.clip(y, *LABEL_CLIP))
    dv = lgb.Dataset(xv, np.clip(yv, *LABEL_CLIP), reference=dt)
    m = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return (lambda xn: m.predict(xn, num_iteration=m.best_iteration)), m


def export_model(path: Path, payload: dict, feats: list[str], predict, x_sample: np.ndarray) -> None:
    """Write a model the Rust scorer can load, plus Python predictions to verify it against."""
    payload = {**payload, "features": feats}
    path.write_text(json.dumps(payload))
    rows = [[None if np.isnan(v) else float(v) for v in r] for r in x_sample]
    verify = {"features": rows, "pred": [float(p) for p in predict(x_sample)]}
    path.with_suffix(".verify.json").write_text(json.dumps(verify))


def hour_block_bootstrap(values: np.ndarray, hours: np.ndarray, n=2000, seed=SEED) -> tuple[float, float]:
    """95% CI of the mean, resampling whole hours (trades within an hour are dependent)."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    uh = np.unique(hours)
    groups = [values[hours == h] for h in uh]
    sums = np.array([g.sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(uh), len(uh))
        c = cnts[idx].sum()
        if c:
            means.append(sums[idx].sum() / c)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(pred: np.ndarray, d: pl.DataFrame, h: str) -> dict:
    y = d[f"ret_net_{h}"].to_numpy()
    hours = (d["block_time"].to_numpy() // 3600)
    out = {"n": int(len(y)), "base_mean_ret": float(y.mean()), "base_hit": float((y > 0).mean())}
    if len(y) < 50 or np.std(pred) == 0:
        return out
    out["ic_spearman"] = float(spearmanr(pred, y).statistic)
    out["auc_sign"] = float(roc_auc_score(y > 0, pred)) if 0 < (y > 0).mean() < 1 else float("nan")
    # IC per hour -> t-stat across hours (robust to within-hour dependence)
    ics = []
    for hh in np.unique(hours):
        m = hours == hh
        if m.sum() >= 30 and np.std(pred[m]) > 0:
            ics.append(spearmanr(pred[m], y[m]).statistic)
    if len(ics) >= 3:
        ics = np.array(ics)
        out["ic_hourly_mean"] = float(ics.mean())
        out["ic_hourly_t"] = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))))
        out["ic_hours"] = len(ics)
    q = np.quantile(pred, [0.9, 0.99])
    for name, thr in (("top10", q[0]), ("top1", q[1])):
        m = pred >= thr
        out[f"{name}_n"] = int(m.sum())
        out[f"{name}_mean_ret"] = float(y[m].mean())
        out[f"{name}_ci95"] = hour_block_bootstrap(y[m], hours[m])
        out[f"{name}_hit"] = float((y[m] > 0).mean())
    return out


def paired_ic_test(pa: np.ndarray, pb: np.ndarray, y: np.ndarray, hours: np.ndarray) -> dict:
    """Is model A's rank IC higher than B's? Paired over hours (each hour one observation)."""
    diffs = []
    for hh in np.unique(hours):
        m = hours == hh
        if m.sum() >= 30 and np.std(pa[m]) > 0 and np.std(pb[m]) > 0:
            diffs.append(spearmanr(pa[m], y[m]).statistic - spearmanr(pb[m], y[m]).statistic)
    if len(diffs) < 3:
        return {"hours": len(diffs)}
    d = np.array(diffs)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {"hours": len(d), "mean_ic_diff": float(d.mean()), "t": float(t), "frac_hours_positive": float((d > 0).mean())}


def calibration(pred: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Reliability of the regression score as a return forecast: mean predicted vs realized by decile."""
    qs = np.quantile(pred, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        m = (pred >= qs[i]) & (pred <= qs[i + 1]) if i == bins - 1 else (pred >= qs[i]) & (pred < qs[i + 1])
        if m.sum():
            rows.append({"bin": i, "n": int(m.sum()), "pred": float(pred[m].mean()), "realized": float(y[m].mean()),
                         "hit": float((y[m] > 0).mean())})
    return rows


def choose_threshold(pred_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Pick the score cutoff maximizing mean validation net return (>= MIN_VAL_TRADES rows)."""
    best = {"threshold": float("inf"), "val_mean_ret": float("nan"), "val_n": 0}
    order = np.argsort(-pred_val)
    ys = y_val[order]
    csum = np.cumsum(ys)
    for k in range(MIN_VAL_TRADES, len(ys) + 1, max(1, len(ys) // 400)):
        mean = csum[k - 1] / k
        if not np.isfinite(best["val_mean_ret"]) or mean > best["val_mean_ret"]:
            best = {"threshold": float(pred_val[order][k - 1]), "val_mean_ret": float(mean), "val_n": int(k)}
    return best


def write_signal(path: Path, d: pl.DataFrame, score: np.ndarray) -> None:
    tbl = pa.table({
        "slot": pa.array(d["slot"].to_numpy().astype(np.uint64), pa.uint64()),
        "mint": pa.array(d["mint"].to_list(), pa.string()),
        "score": pa.array(score.astype(np.float64), pa.float64()),
    })
    pq.write_table(tbl, path)


def run(dataset: Path, out_dir: Path, purge_secs: float = 630.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(dataset)
    sp = split(df, purge_secs=purge_secs)
    res: dict = {"rows": df.height, "slot_secs": slot_secs(df), "bounds": sp.bounds,
                 "split_rows": {"train": sp.train.height, "val": sp.val.height, "test": sp.test.height},
                 "horizons": {}}
    for h in HORIZONS:
        hr: dict = {"models": {}, "ablation": {}}
        x, y, _ = xy(sp.train, ALL, h)
        xv, yv, dv = xy(sp.val, ALL, h)
        # Test rows: keep all decision rows (signal file needs every decision point);
        # evaluation uses rows with a label.
        xt, yt, dt = xy(sp.test, ALL, h)
        if min(len(y), len(yv), len(yt)) < 200:
            res["horizons"][h] = {"skipped": "insufficient labelled rows",
                                  "n": [int(len(y)), int(len(yv)), int(len(yt))]}
            print(f"[{h}] skipped: labelled rows train/val/test = {len(y)}/{len(yv)}/{len(yt)}", flush=True)
            continue
        hr["label_stats_train"] = {"n": int(len(y)), "mean": float(y.mean()), "median": float(np.median(y)),
                                   "p_pos": float((y > 0).mean())}
        ridge, ridge_params = fit_ridge(x, y)
        lg, model = fit_lgbm(x, y, xv, yv)
        sample = xt[:: max(1, len(xt) // 500)][:500]
        export_model(out_dir / f"model_ridge_{h}.json", ridge_params, ALL, ridge, sample)
        export_model(out_dir / f"model_lgbm_{h}.json",
                     {"kind": "lgbm", "dump": model.dump_model(num_iteration=model.best_iteration)}, ALL, lg, sample)
        for name, f in (("ridge", ridge), ("lgbm", lg)):
            pv, pt = f(xv), f(xt)
            thr = choose_threshold(pv, yv)
            ev = evaluate(pt, dt, h)
            ev["val"] = evaluate(pv, dv, h)
            ev["threshold"] = thr
            m = pt >= thr["threshold"]
            ev["test_at_val_threshold"] = {"n": int(m.sum()), "mean_ret": float(yt[m].mean()) if m.any() else None,
                                           "ci95": hour_block_bootstrap(yt[m], dt["block_time"].to_numpy()[m] // 3600)}
            ev["calibration_test"] = calibration(pt, yt)
            hr["models"][name] = ev
            # Signal over ALL test decision rows (labelled or not), for the engine.
            test_all = sp.test
            xa = test_all.select(ALL).to_numpy().astype(np.float32)
            write_signal(out_dir / f"signal_{name}_{h}.parquet", test_all, f(xa))
        hr["ridge_coefs"] = dict(zip(ALL, ridge_params["coef"]))
        imp = model.feature_importance("gain")
        hr["lgbm_gain"] = dict(sorted(zip(ALL, (imp / imp.sum()).round(4).tolist()), key=lambda kv: -kv[1]))
        # Ablations (LightGBM): which feature groups carry the test IC?
        preds = {}
        for fs_name, feats in FEATURE_SETS.items():
            a, b, _ = xy(sp.train, feats, h)
            av, bv, _ = xy(sp.val, feats, h)
            at, _, dta = xy(sp.test, feats, h)
            f, _ = fit_lgbm(a, b, av, bv)
            preds[fs_name] = f(at)
            hr["ablation"][fs_name] = evaluate(preds[fs_name], dta, h)
        hours_t = dt["block_time"].to_numpy() // 3600
        hr["wallet_increment"] = {
            "momentum+wallet_vs_momentum_only": paired_ic_test(preds["momentum+wallet"], preds["momentum_only"], yt, hours_t),
            "all_vs_no_wallet": paired_ic_test(preds["all"], preds["no_wallet"], yt, hours_t),
        }
        res["horizons"][h] = hr
        print(f"[{h}] lgbm IC={hr['models']['lgbm'].get('ic_spearman')}, ridge IC={hr['models']['ridge'].get('ic_spearman')}", flush=True)
    (out_dir / "research.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/research"))
    a = ap.parse_args()
    run(a.dataset, a.out)
