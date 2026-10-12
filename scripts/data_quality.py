"""Data-quality report for an event store -> results/data_quality.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/events")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/data_quality.json")
ev = pl.scan_parquet(str(src / "events_*.parquet"))

man = [json.loads(l) for l in (src / "manifest.jsonl").read_text().splitlines() if l.strip()]
s = ev.select(pl.col("slot").min().alias("s0"), pl.col("slot").max().alias("s1"),
              pl.col("block_time").min().alias("t0"), pl.col("block_time").max().alias("t1"), pl.len().alias("n")).collect().row(0, named=True)

kinds = {0: "create", 1: "trade", 2: "complete", 3: "migrate", 4: "amm_buy", 5: "amm_sell", 6: "failed"}
by_kind = ev.group_by("kind").len().collect()
hourly = (ev.with_columns(hour=(pl.col("block_time") - s["t0"]) // 3600)
          .group_by("hour", "kind").len().collect().pivot(on="kind", index="hour", values="len").sort("hour"))

creates = ev.filter(pl.col("kind") == 0).select("mint", "vsol", mayhem=pl.col("is_buy")).collect()
curve = ev.filter(pl.col("kind") == 1)
zero_fee_mints = curve.filter(pl.col("fee_bps") == 0).select("mint").unique().collect()["mint"]
sol_q = creates.filter(pl.col("vsol") == 30_000_000_000)
cohort = sol_q.filter(~pl.col("mayhem") & ~pl.col("mint").is_in(zero_fee_mints.implode()))
completes = ev.filter(pl.col("kind") == 2).select("mint").collect()["mint"]

fails = ev.filter(pl.col("kind") == 6).with_columns(
    code=pl.col("symbol").str.extract(r'Custom":(\d+)').cast(pl.Int64)).collect()
ok_curve_buys = curve.filter(pl.col("is_buy")).select(pl.len()).collect().item()
curve_buy_fails = fails.filter((pl.col("pool") == "curve") & pl.col("is_buy"))
anchor_coded = curve_buy_fails.filter(pl.col("code").is_between(6000, 6099))

pf = curve.select(pl.col("priority_fee").quantile(q).alias(f"p{int(q*100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).collect()
trades_per_mint = (curve.join(cohort.lazy().select("mint"), on="mint").group_by("mint").len().collect()["len"])

rep = {
    "slot_range": [s["s0"], s["s1"]], "utc_range": [s["t0"], s["t1"]], "hours": (s["t1"] - s["t0"]) / 3600,
    "slot_secs": (s["t1"] - s["t0"]) / (s["s1"] - s["s0"]), "events": s["n"],
    "manifest": {"chunks": len(man), "slots": sum(m["hi"] - m["lo"] for m in man),
                 "skipped_slots": sum(m["skipped"] for m in man), "missing_slots": sum(m["missing"] for m in man),
                 "bytes": sum(m["bytes"] for m in man)},
    "events_by_kind": {kinds[k]: n for k, n in by_kind.iter_rows()},
    "events_per_hour": hourly.to_dicts(),
    "launches": {"total": creates.height, "sol_quoted": sol_q.height, "mayhem_flagged": int(creates["mayhem"].sum()),
                 "non_sol_quote": creates.height - sol_q.height, "cohort": cohort.height,
                 "cohort_graduated": int(cohort["mint"].is_in(completes.implode()).sum())},
    "cohort_curve_trades_per_token": {"zero_trades": cohort.height - trades_per_mint.len(),
                                      "median": float(trades_per_mint.median() or 0), "p90": float(trades_per_mint.quantile(0.9) or 0),
                                      "max": int(trades_per_mint.max() or 0)},
    "failed_curve_buys": {"total": curve_buy_fails.height, "successful_curve_buys": ok_curve_buys,
                          "raw_fail_share": curve_buy_fails.height / max(curve_buy_fails.height + ok_curve_buys, 1),
                          "anchor_60xx_share": anchor_coded.height / max(anchor_coded.height + ok_curve_buys, 1),
                          "top_codes": fails.filter(pl.col("pool") == "curve").group_by("code").len()
                          .sort("len", descending=True).head(8).to_dicts()},
    "priority_fee_lamports_curve_trades": pf.to_dicts()[0],
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rep, indent=2, default=str))
print(json.dumps({k: v for k, v in rep.items() if k != "events_per_hour"}, indent=2, default=str))
