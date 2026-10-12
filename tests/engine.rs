//! Correctness invariants on a real 1000-slot fixture (curve events + launched-mint AMM/failed rows).

use slotfish::engine::{EngineConfig, Replay};
use slotfish::events::Kind;
use slotfish::exec::{round_trip, ExecConfig};
use slotfish::features::FeatureEngine;
use slotfish::market::Timeline;
use slotfish::store::{read_tape, Tape};
use slotfish::strategy::RandomEntry;
use std::path::Path;

fn fixture() -> (Tape, Timeline) {
    let t = read_tape(Path::new("tests/fixtures/events_1000slots.parquet")).unwrap();
    let tl = Timeline::build(&t);
    (t, tl)
}

fn cfg(t: &Tape, p_fail: f64) -> EngineConfig {
    EngineConfig {
        exec: ExecConfig {
            p_fail,
            ..Default::default()
        },
        hold_slots: 40,
        max_positions: 1000,
        decide_from: t.slot[0],
        decide_to: *t.slot.last().unwrap(),
        skill_horizon: 50,
    }
}

#[test]
fn fixture_is_in_chain_order_and_has_cohort() {
    let (t, tl) = fixture();
    assert!(t.len() > 10_000);
    for i in 1..t.len() {
        assert!(
            (t.slot[i - 1], t.tx_idx[i - 1]) <= (t.slot[i], t.tx_idx[i]),
            "order broken at {i}"
        );
    }
    assert!(tl.info.iter().filter(|m| m.in_cohort()).count() > 10);
}

#[test]
fn curve_reserves_are_continuous_for_cohort() {
    // Each curve trade's pre-state (post minus its own delta) equals the previous
    // event's post-state: proves no missing or misordered trades for the cohort.
    let (t, tl) = fixture();
    let mut last: Vec<Option<(u64, u64)>> = vec![None; t.mints.names.len()];
    let (mut ok, mut n) = (0, 0);
    for i in 0..t.len() {
        if t.kind[i] != Kind::Trade || !tl.info[t.mint[i] as usize].tradable_at(t.slot[i]) {
            continue;
        }
        let m = t.mint[i] as usize;
        // Standard curves can never go below their 30 SOL virtual floor.
        assert!(
            t.vsol[i] >= slotfish::market::STANDARD_VIRTUAL_SOL,
            "vsol below floor at {i}"
        );
        let (pre_sol, pre_tok) = if t.is_buy[i] {
            (t.vsol[i] - t.sol_amount[i], t.vtok[i] + t.token_amount[i])
        } else {
            (t.vsol[i] + t.sol_amount[i], t.vtok[i] - t.token_amount[i])
        };
        if let Some(prev) = last[m] {
            n += 1;
            if prev == (pre_sol, pre_tok) {
                ok += 1;
            }
        }
        last[m] = Some((t.vsol[i], t.vtok[i]));
    }
    assert!(n > 1000);
    assert_eq!(ok, n, "{} of {n} curve transitions inconsistent", n - ok);
}

#[test]
fn engine_matches_reference_round_trip() {
    // The streaming engine and the timeline-based reference must agree exactly,
    // including random failures and exit retries.
    let (t, tl) = fixture();
    for p_fail in [0.0, 0.3] {
        let c = cfg(&t, p_fail);
        let mut s = RandomEntry { p: 0.2, seed: 3 };
        let (trades, _) = Replay {
            tape: &t,
            tl: &tl,
            cfg: c.clone(),
        }
        .run(&mut s, false);
        let data_end = *t.slot.last().unwrap();
        let mut compared = 0;
        for tr in trades.iter().filter(|x| !x.censored) {
            let quote = tl.state_at_end_of(tr.mint, tr.decision_slot).unwrap().0;
            let r = round_trip(
                &tl,
                &c.exec,
                tr.mint,
                tr.decision_slot,
                quote,
                c.hold_slots,
                data_end,
            );
            assert_eq!(&r, tr, "mismatch for decision at slot {}", tr.decision_slot);
            compared += 1;
        }
        assert!(compared > 20, "only {compared} trades compared");
    }
}

#[test]
fn replay_is_deterministic() {
    let (t, tl) = fixture();
    let c = cfg(&t, 0.1);
    let run = || {
        Replay {
            tape: &t,
            tl: &tl,
            cfg: c.clone(),
        }
        .run(&mut RandomEntry { p: 0.1, seed: 9 }, false)
        .0
    };
    assert_eq!(run(), run());
}

#[test]
fn features_use_no_future_information() {
    // Features at event i computed on the full tape must equal features computed
    // on a tape that physically ends at event i.
    let (t, tl) = fixture();
    let mut full = FeatureEngine::new(50);
    let mut fa = Vec::new();
    let mut fb = Vec::new();
    let mut checked = 0;
    let probes: Vec<usize> = (0..t.len())
        .filter(|&i| t.kind[i] == Kind::Trade)
        .step_by(97)
        .collect();
    let mut next_probe = probes.iter().peekable();
    for i in 0..t.len() {
        full.update(&t, &tl, i);
        if next_probe.peek() == Some(&&i) {
            next_probe.next();
            full.features(&t, &tl, i, &mut fa);
            let pt = t.prefix(i + 1);
            let ptl = Timeline::build(&pt);
            let mut fe = FeatureEngine::new(50);
            for j in 0..=i {
                fe.update(&pt, &ptl, j);
            }
            fe.features(&pt, &ptl, i, &mut fb);
            for (k, (a, b)) in fa.iter().zip(fb.iter()).enumerate() {
                assert!(
                    a == b || (a.is_nan() && b.is_nan()),
                    "feature {k} differs at event {i}: {a} vs {b}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 50);
}
