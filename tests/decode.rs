//! Decoder regression test on three real blocks (slots 455560000-455560002,
//! filtered to transactions that invoke pump.fun or PumpSwap).

use slotfish::decode::decode_slot;
use slotfish::events::{Event, Kind};
use std::io::BufRead;

fn decode_fixture() -> Vec<Event> {
    let f = std::fs::File::open("tests/fixtures/blocks_3.jsonl.gz").unwrap();
    let r = std::io::BufReader::new(flate2::read::GzDecoder::new(f));
    let mut out = Vec::new();
    for line in r.lines() {
        let v: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        decode_slot(&v, &mut out);
    }
    out
}

#[test]
fn decoded_counts_are_stable() {
    let evs = decode_fixture();
    let count = |k: Kind| evs.iter().filter(|e| e.kind == k).count();
    assert_eq!(count(Kind::Create), 2);
    assert_eq!(count(Kind::Trade), 23);
    assert_eq!(count(Kind::AmmBuy), 29);
    assert_eq!(count(Kind::AmmSell), 16);
    assert_eq!(count(Kind::Failed), 9);
}

#[test]
fn decoded_events_are_well_formed() {
    let evs = decode_fixture();
    for w in evs.windows(2) {
        assert!((w[0].slot, w[0].tx_idx, w[0].ev_idx) < (w[1].slot, w[1].tx_idx, w[1].ev_idx));
    }
    for e in &evs {
        assert_eq!(
            bs58::decode(&e.mint).into_vec().map(|v| v.len()).ok(),
            Some(32),
            "bad mint {}",
            e.mint
        );
        match e.kind {
            Kind::Trade => {
                assert!(e.token_amount > 0 && e.vtok > 0);
                assert!(e.vsol >= e.rsol);
                assert!(e.fee_bps <= 2_000);
            }
            Kind::AmmBuy | Kind::AmmSell => {
                assert_eq!(e.is_buy, e.kind == Kind::AmmBuy);
                assert!(e.vsol > 0 && e.vtok > 0 && !e.pool.is_empty());
            }
            Kind::Failed => assert!(e.symbol.contains("InstructionError") || !e.symbol.is_empty()),
            _ => {}
        }
    }
}
