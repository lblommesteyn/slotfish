//! Raw block JSON -> typed events.
//!
//! pump.fun (bonding curve) and PumpSwap (AMM) are Anchor programs that emit
//! events through a self-CPI ("emit_cpi!"): an inner instruction to the
//! program itself whose data is `EVENT_IX_TAG ++ sha256("event:<Name>")[..8]
//! ++ borsh(event)`. Every field we need, including post-trade reserves, is in
//! those events, so the AMM state is reconstructed exactly rather than inferred
//! from token balance deltas.
//!
//! Failed transactions emit no events. For them we decode the attempted
//! instruction (buy/sell discriminator + account list) so execution-failure
//! rates can be measured.

use crate::events::{Event, Kind};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub const PUMP: &str = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P";
pub const PAMM: &str = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA";
pub const WSOL: &str = "So11111111111111111111111111111111111111112";
const EVENT_IX_TAG: [u8; 8] = [0xe4, 0x45, 0xa5, 0x2e, 0x51, 0xcb, 0x9a, 0x1d];

fn disc(prefix: &str, name: &str) -> [u8; 8] {
    let h = Sha256::digest(format!("{prefix}:{name}").as_bytes());
    h[..8].try_into().unwrap()
}

struct Discs {
    trade: [u8; 8],
    create: [u8; 8],
    complete: [u8; 8],
    migrate: [u8; 8],
    amm_buy: [u8; 8],
    amm_sell: [u8; 8],
    ix_buy: [u8; 8],
    ix_buy_exact_sol: [u8; 8],
    ix_sell: [u8; 8],
}

fn discs() -> &'static Discs {
    static D: OnceLock<Discs> = OnceLock::new();
    D.get_or_init(|| Discs {
        trade: disc("event", "TradeEvent"),
        create: disc("event", "CreateEvent"),
        complete: disc("event", "CompleteEvent"),
        migrate: disc("event", "CompletePumpAmmMigrationEvent"),
        amm_buy: disc("event", "BuyEvent"),
        amm_sell: disc("event", "SellEvent"),
        ix_buy: disc("global", "buy"),
        ix_buy_exact_sol: disc("global", "buy_exact_sol_in"),
        ix_sell: disc("global", "sell"),
    })
}

/// Little-endian borsh reader that fails soft (returns None past the end).
struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> R<'a> {
    fn new(b: &'a [u8]) -> Self {
        R { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.p..self.p + n)?;
        self.p += n;
        Some(s)
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn key(&mut self) -> Option<String> {
        Some(bs58::encode(self.take(32)?).into_string())
    }
    fn skip(&mut self, n: usize) -> Option<()> {
        self.take(n).map(|_| ())
    }
    fn string(&mut self) -> Option<String> {
        let n = u32::from_le_bytes(self.take(4)?.try_into().ok()?) as usize;
        Some(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

/// One instruction with its resolved program id and account keys.
struct Ix<'a> {
    program: &'a str,
    accounts: Vec<&'a str>,
    data: Vec<u8>,
    height: usize,
}

fn as_ix<'a>(v: &'a Value, keys: &[&'a str], default_height: usize) -> Option<Ix<'a>> {
    let program = *keys.get(v["programIdIndex"].as_u64()? as usize)?;
    let accounts = v["accounts"]
        .as_array()?
        .iter()
        .filter_map(|a| keys.get(a.as_u64()? as usize).copied())
        .collect();
    let data = bs58::decode(v["data"].as_str()?).into_vec().ok()?;
    let height = v["stackHeight"]
        .as_u64()
        .map(|h| h as usize)
        .unwrap_or(default_height);
    Some(Ix {
        program,
        accounts,
        data,
        height,
    })
}

/// Per-transaction context copied onto every event it produces.
struct TxCtx {
    slot: u64,
    block_time: i64,
    tx_idx: u32,
    tx_fee: u64,
    cu: u64,
    n_sigs: u64,
}

impl TxCtx {
    fn base(&self, kind: Kind, ev_idx: u16) -> Event {
        Event {
            slot: self.slot,
            block_time: self.block_time,
            tx_idx: self.tx_idx,
            ev_idx,
            kind,
            tx_fee: self.tx_fee,
            priority_fee: self.tx_fee.saturating_sub(5000 * self.n_sigs),
            cu: self.cu,
            ..Default::default()
        }
    }
}

/// Decode one event-CPI payload (bytes after EVENT_IX_TAG). `parent` is the
/// instruction that emitted it, used to recover the AMM base mint.
fn decode_event(
    ctx: &TxCtx,
    ev_idx: u16,
    program: &str,
    payload: &[u8],
    parent: Option<&Ix>,
) -> Option<Event> {
    let d = discs();
    let (tag, body) = payload.split_at_checked(8)?;
    let mut r = R::new(body);
    if program == PUMP {
        if tag == d.trade {
            let mut e = ctx.base(Kind::Trade, ev_idx);
            e.mint = r.key()?;
            e.sol_amount = r.u64()?;
            e.token_amount = r.u64()?;
            e.is_buy = r.u8()? != 0;
            e.user = r.key()?;
            e.ts = r.i64()?;
            e.vsol = r.u64()?;
            e.vtok = r.u64()?;
            e.rsol = r.u64()?;
            e.rtok = r.u64()?;
            // Fee fields were appended in later program versions; tolerate their absence.
            let mut fee_bps = 0;
            let mut fee = 0;
            if r.skip(32).is_some() {
                fee_bps += r.u64().unwrap_or(0);
                fee += r.u64().unwrap_or(0);
                if r.skip(32).is_some() {
                    fee_bps += r.u64().unwrap_or(0);
                    fee += r.u64().unwrap_or(0);
                }
            }
            e.fee_bps = fee_bps as u32;
            e.fee = fee;
            return Some(e);
        }
        if tag == d.create {
            let mut e = ctx.base(Kind::Create, ev_idx);
            let _name = r.string()?;
            e.symbol = r.string()?;
            let _uri = r.string()?;
            e.mint = r.key()?;
            let _curve = r.key()?;
            e.user = r.key()?;
            let _creator = r.key();
            e.ts = r.i64().unwrap_or(0);
            e.vtok = r.u64().unwrap_or(0);
            e.vsol = r.u64().unwrap_or(0);
            e.rtok = r.u64().unwrap_or(0);
            let _supply = r.u64();
            let _token_program = r.key();
            // is_mayhem_mode (Create rows store it in `is_buy`): known at launch, so
            // excluding Mayhem tokens from the universe uses no future information.
            e.is_buy = r.u8().unwrap_or(0) != 0;
            return Some(e);
        }
        if tag == d.complete {
            let mut e = ctx.base(Kind::Complete, ev_idx);
            e.user = r.key()?;
            e.mint = r.key()?;
            let _curve = r.key()?;
            e.ts = r.i64()?;
            return Some(e);
        }
        if tag == d.migrate {
            let mut e = ctx.base(Kind::Migrate, ev_idx);
            e.user = r.key()?;
            e.mint = r.key()?;
            e.token_amount = r.u64()?;
            e.sol_amount = r.u64()?;
            e.fee = r.u64()?;
            let _curve = r.key()?;
            e.ts = r.i64()?;
            e.pool = r.key()?;
            return Some(e);
        }
    } else if program == PAMM && (tag == d.amm_buy || tag == d.amm_sell) {
        let buy = tag == d.amm_buy;
        let parent = parent?;
        // buy/sell accounts: [pool, user, global_config, base_mint, quote_mint, ...]
        if parent.accounts.get(4) != Some(&WSOL) {
            return None;
        }
        let mut e = ctx.base(if buy { Kind::AmmBuy } else { Kind::AmmSell }, ev_idx);
        e.is_buy = buy;
        e.mint = parent.accounts.get(3)?.to_string();
        e.ts = r.i64()?;
        e.token_amount = r.u64()?; // base_amount_out (buy) / base_amount_in (sell)
        let _limit = r.u64()?;
        let _ub = r.u64()?;
        let _uq = r.u64()?;
        let pool_base = r.u64()?;
        let pool_quote = r.u64()?;
        let _quote_amount = r.u64()?; // curve-side amount before LP fee
        let lp_bps = r.u64()?;
        let lp_fee = r.u64()?;
        let proto_bps = r.u64()?;
        let proto_fee = r.u64()?;
        // Pool quote reserve moves by exactly this amount (verified empirically,
        // scripts/dev/amm_layout.py): in_with_lp_fee (buy) / out_without_lp_fee (sell).
        let pool_delta = r.u64()?;
        let user_quote = r.u64()?; // total SOL paid (buy) / received (sell) by user, all fees included
        e.pool = r.key()?;
        e.user = r.key()?;
        r.skip(32 * 4)?;
        let _creator = r.key();
        let creator_bps = r.u64().unwrap_or(0);
        let creator_fee = r.u64().unwrap_or(0);
        e.sol_amount = pool_delta;
        e.user_sol = user_quote;
        e.fee_bps = (lp_bps + proto_bps + creator_bps) as u32;
        e.fee = lp_fee + proto_fee + creator_fee;
        // Event reports PRE-trade pool reserves; store POST-trade so that every
        // event row means "state after this event", like the curve events.
        if buy {
            e.vtok = pool_base.saturating_sub(e.token_amount);
            e.vsol = pool_quote + pool_delta;
        } else {
            e.vtok = pool_base + e.token_amount;
            e.vsol = pool_quote.saturating_sub(pool_delta);
        }
        e.rsol = e.vsol;
        e.rtok = e.vtok;
        return Some(e);
    }
    None
}

/// Decode an attempted buy/sell instruction from a failed transaction.
fn decode_attempt(ctx: &TxCtx, ix: &Ix, err: &str) -> Option<Event> {
    let d = discs();
    let tag = ix.data.get(..8)?;
    let (is_buy, mint, user) = if ix.program == PUMP {
        // buy/sell accounts: [global, fee_recipient, mint, bonding_curve, abc, abu, user, ...]
        let b = if tag == d.ix_buy || tag == d.ix_buy_exact_sol {
            true
        } else if tag == d.ix_sell {
            false
        } else {
            return None;
        };
        (b, ix.accounts.get(2)?, ix.accounts.get(6)?)
    } else if ix.program == PAMM {
        let b = if tag == d.ix_buy {
            true
        } else if tag == d.ix_sell {
            false
        } else {
            return None;
        };
        (b, ix.accounts.get(3)?, ix.accounts.get(1)?)
    } else {
        return None;
    };
    let mut e = ctx.base(Kind::Failed, 0);
    e.is_buy = is_buy;
    e.mint = mint.to_string();
    e.user = user.to_string();
    e.pool = if ix.program == PAMM {
        "amm".into()
    } else {
        "curve".into()
    };
    e.symbol = err.chars().take(64).collect();
    // First u64 arg: token amount (buy/sell) or sol amount (buy_exact_sol_in).
    e.token_amount = ix
        .data
        .get(8..16)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .unwrap_or(0);
    Some(e)
}

/// Decode one filtered slot record (one JSONL line written by fetch.py).
pub fn decode_slot(rec: &Value, out: &mut Vec<Event>) {
    let Some(slot) = rec["slot"].as_u64() else {
        return;
    };
    let Some(txs) = rec["txs"].as_array() else {
        return;
    };
    let block_time = rec["blockTime"].as_i64().unwrap_or(0);
    for tx in txs {
        let meta = &tx["meta"];
        let msg = &tx["transaction"]["message"];
        let mut keys: Vec<&str> = msg["accountKeys"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        for k in ["writable", "readonly"] {
            keys.extend(
                meta["loadedAddresses"][k]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str),
            );
        }
        let ctx = TxCtx {
            slot,
            block_time,
            tx_idx: tx["idx"].as_u64().unwrap_or(0) as u32,
            tx_fee: meta["fee"].as_u64().unwrap_or(0),
            cu: meta["computeUnitsConsumed"].as_u64().unwrap_or(0),
            n_sigs: tx["transaction"]["signatures"]
                .as_array()
                .map(|a| a.len() as u64)
                .unwrap_or(1),
        };
        let failed = !meta["err"].is_null();

        // Flatten [top-level ix, its inner ixs...] in execution order.
        let top = msg["instructions"].as_array().cloned().unwrap_or_default();
        let inner = meta["innerInstructions"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut ev_idx: u16 = 0;
        for (i, t) in top.iter().enumerate() {
            let mut seq: Vec<Ix> = Vec::new();
            if let Some(ix) = as_ix(t, &keys, 1) {
                seq.push(ix);
            }
            for g in inner
                .iter()
                .filter(|g| g["index"].as_u64() == Some(i as u64))
            {
                for v in g["instructions"].as_array().into_iter().flatten() {
                    if let Some(ix) = as_ix(v, &keys, 2) {
                        seq.push(ix);
                    }
                }
            }
            if failed {
                let err = meta["err"].to_string();
                // Count one attempt per transaction: the first buy/sell ix found.
                if let Some(e) = seq.iter().find_map(|ix| decode_attempt(&ctx, ix, &err)) {
                    out.push(e);
                    break;
                }
                continue;
            }
            // parents[h] = latest non-event instruction seen at stack height h.
            let mut parents: Vec<Option<usize>> = vec![None; 8];
            for (j, ix) in seq.iter().enumerate() {
                let is_event = ix.data.len() >= 16 && ix.data[..8] == EVENT_IX_TAG;
                if is_event && (ix.program == PUMP || ix.program == PAMM) {
                    let parent = ix
                        .height
                        .checked_sub(1)
                        .and_then(|h| parents.get(h).copied().flatten())
                        .map(|p| &seq[p]);
                    if let Some(e) = decode_event(&ctx, ev_idx, ix.program, &ix.data[8..], parent) {
                        out.push(e);
                        ev_idx += 1;
                    }
                } else if ix.height < parents.len() {
                    parents[ix.height] = Some(j);
                    for p in parents.iter_mut().skip(ix.height + 1) {
                        *p = None;
                    }
                }
            }
        }
    }
}
