"""Summarize a paper-trading session (data/paper) into results/paper.json.

Compares each paper trade at the *measured* latency with the backtest's
assumed latency, and checks simulated entry fills against real buys of the
same token that landed in the same slot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/paper")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/paper.json")

trades = [json.loads(l) for l in (src / "paper_trades.jsonl").read_text().splitlines() if l.strip()]
lags = [json.loads(l)["lag_s"] for l in (src / "lag.jsonl").read_text().splitlines() if l.strip()]

real = [t for t in trades if t["real"]["entered"] and not t["real"]["censored"]]
assumed = [t for t in trades if t["assumed"]["entered"]]


def stats(xs):
    xs = np.array(xs, dtype=float)
    if len(xs) == 0:
        return {"n": 0}
    se = xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else float("nan")
    return {"n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)), "se": float(se),
            "hit": float((xs > 0).mean()), "sum_sol": None}


fills = []
for t in real:
    fc = t["fill_check"]
    if fc["real_buys_in_landing_slot"] > 0 and fc["sim_lamports_per_token"] == fc["sim_lamports_per_token"]:
        sim, lo, hi = fc["sim_lamports_per_token"], fc["real_min_px"], fc["real_max_px"]
        fills.append({"within": lo * 0.999 <= sim <= hi * 1.001 or sim >= hi, "sim_vs_max": sim / hi - 1,
                      "sim_vs_min": sim / lo - 1})

summary = {
    "paper_signals": len(trades),
    "data_lag_s": {"p50": float(np.percentile(lags, 50)), "p90": float(np.percentile(lags, 90)),
                   "p99": float(np.percentile(lags, 99)), "n": len(lags)} if lags else None,
    "latency_slots_real_p50": float(np.median([t["latency_slots_real"] for t in trades])) if trades else None,
    "ret_at_measured_latency": stats([t["real"]["ret"] for t in real]),
    "ret_at_assumed_latency": stats([t["assumed"]["ret"] for t in assumed]),
    "skips_at_measured_latency": {k: sum(1 for t in trades if t["real"]["skip"] == k)
                                  for k in {t["real"]["skip"] for t in trades}},
    "fill_check": {
        "entries_with_real_buys_in_landing_slot": len(fills),
        "sim_price_at_or_above_real_min": float(np.mean([x["sim_vs_min"] >= -1e-3 for x in fills])) if fills else None,
        "median_sim_vs_real_max": float(np.median([x["sim_vs_max"] for x in fills])) if fills else None,
    },
}
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
