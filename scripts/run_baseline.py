"""Reproduce the baseline experiment end to end.

    python scripts/run_baseline.py [--events data/events] [--out results]

Steps: build -> info -> dataset -> research (models, ablations, signals) ->
test-window backtests (no-trade, random, momentum, models) -> execution
sensitivity sweeps -> benchmark -> survival -> results/summary.json.
Every step is deterministic given the event store.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from slotfish import research, survival  # noqa: E402

EXE = ROOT / "target" / "release" / ("slotfish.exe" if sys.platform == "win32" else "slotfish")


def sh(args: list[str], capture=True) -> str:
    print("+", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([str(a) for a in args], cwd=ROOT, capture_output=capture, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {args}")
    return r.stdout


def backtest(events: Path, window: dict, hold_secs: float, strategy: list[str], extra: list[str] | None = None) -> dict:
    args = [EXE, "backtest", "--events", events, "--from-slot", window[0], "--to-slot", window[1],
            "--hold-secs", hold_secs, *strategy, *(extra or [])]
    return json.loads(sh(args))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=Path("data/events"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_build:
        sh(["cargo", "build", "--release"], capture=False)

    info = json.loads(sh([EXE, "info", "--events", a.events]))
    (out / "info.json").write_text(json.dumps(info, indent=2))

    ds = Path("data/dataset.parquet")
    sh([EXE, "dataset", "--events", a.events, "--out", ds])
    res = research.run(ds, out / "research")

    test = res["bounds"]["test"]
    val = res["bounds"]["val"]
    n_test_decisions = res["split_rows"]["test"]
    bt: dict = {}
    for h, secs in research.HORIZONS.items():
        rows = {}
        rows["no_trade"] = backtest(a.events, test, secs, ["--strategy", "none"])
        rows["momentum"] = backtest(a.events, test, secs, ["--strategy", "momentum"])
        for model in ("ridge", "lgbm"):
            thr = res["horizons"][h]["models"][model]["threshold"]["threshold"]
            sig = out / "research" / f"signal_{model}_{h}.parquet"
            rows[model] = backtest(a.events, test, secs,
                                   ["--strategy", "signal", "--signal", sig, "--threshold", thr, "--label", f"{model}_{h}"])
        # Random entry matched to the LightGBM strategy's signal count (at least 1%).
        p = max(rows["lgbm"]["signals"] / max(n_test_decisions, 1), 0.01)
        rows["random"] = backtest(a.events, test, secs, ["--strategy", "random", "--random-p", f"{p:.6f}"])
        bt[h] = rows
    (out / "backtests.json").write_text(json.dumps({"test_window": test, "val_window": val, "results": bt}, indent=2))

    # Execution sensitivity on the 2-minute horizon: latency, size (capacity), failure rate.
    sweeps = {}
    sig = out / "research" / "signal_lgbm_2m.parquet"
    thr = res["horizons"]["2m"]["models"]["lgbm"]["threshold"]["threshold"]
    for name, strat in (("lgbm_2m", ["--strategy", "signal", "--signal", sig, "--threshold", thr]),
                        ("momentum", ["--strategy", "momentum"]),
                        ("random", ["--strategy", "random", "--random-p", "0.01"])):
        f = out / f"sweep_{name}.jsonl"
        sh([EXE, "sweep", "--events", a.events, "--from-slot", test[0], "--to-slot", test[1], *strat,
            "--latencies", "1,2,4,8,16,32", "--sizes", "0.05,0.1,0.5,1,2", "--holds", "120",
            "--p-fails", "0,0.05,0.2", "--out", f])
        sweeps[name] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]

    bench = json.loads(sh([EXE, "bench", "--events", a.events, "--repeats", "3"]))
    (out / "bench.json").write_text(json.dumps(bench, indent=2))

    surv = survival.run(a.events, ds, out / "survival")

    summary = {"info": info, "research_bounds": res["bounds"], "backtests": bt, "bench": bench,
               "sweep_files": {k: f"sweep_{k}.jsonl" for k in sweeps}, "survival": surv["endpoints"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("done ->", out / "summary.json")


if __name__ == "__main__":
    main()
