//! Per-mint market timeline and universe metadata.
//!
//! The timeline answers "what was this token's pool state at the END of slot
//! s?" by binary search. Both the label generator and the event-driven engine
//! fill orders through [`Timeline::state_at_end_of`]-equivalent semantics, and
//! a test asserts they agree.

use crate::amm::Pool;
use crate::events::Kind;
use crate::store::Tape;

#[derive(Clone, Debug, Default)]
pub struct MintInfo {
    /// Slot of the Create event, if the launch is inside the window.
    pub create_slot: Option<u64>,
    /// Creator wallet (interned user id).
    pub creator: Option<u32>,
    /// Launched in Mayhem mode (CreateEvent flag, known at launch). The Mayhem
    /// agent's zero-fee trades break constant-product reconstruction, so these
    /// tokens are excluded from trading.
    pub mayhem: bool,
    /// First slot with a zero-fee curve trade (Mayhem-agent activity on a token
    /// not flagged at launch). New decisions stop from this slot on.
    pub zero_fee_slot: Option<u64>,
    pub complete_slot: Option<u64>,
    /// Canonical PumpSwap pool (from the migration event), interned in `pools`.
    pub pool: Option<String>,
    pub n_curve_trades: u32,
}

impl MintInfo {
    /// Launched inside the window (full history known) and not Mayhem mode.
    pub fn in_cohort(&self) -> bool {
        self.create_slot.is_some() && !self.mayhem
    }

    /// Point-in-time tradability at the end of `slot`: in cohort, still on the
    /// bonding curve, and no zero-fee (non-reconstructible) trade seen yet.
    pub fn tradable_at(&self, slot: u64) -> bool {
        self.in_cohort()
            && self.complete_slot.is_none_or(|c| c > slot)
            && self.zero_fee_slot.is_none_or(|z| z > slot)
    }
}

/// Per-mint state history: CSR layout (offsets into flat arrays).
#[derive(Default)]
pub struct Timeline {
    pub offsets: Vec<usize>,
    pub slot: Vec<u64>,
    pub pool: Vec<Pool>,
    /// Venue of each state: false = curve, true = PumpSwap.
    pub amm: Vec<bool>,
    pub info: Vec<MintInfo>,
}

impl Timeline {
    pub fn build(t: &Tape) -> Timeline {
        let n_mints = t.mints.names.len();
        let mut info = vec![MintInfo::default(); n_mints];
        // Pass 1: metadata.
        for i in 0..t.len() {
            let m = t.mint[i] as usize;
            match t.kind[i] {
                Kind::Create => {
                    info[m].create_slot = Some(t.slot[i]);
                    info[m].creator = Some(t.user[i]);
                    info[m].mayhem = t.is_buy[i];
                }
                Kind::Trade => {
                    info[m].n_curve_trades += 1;
                    if t.fee_bps[i] == 0 && info[m].zero_fee_slot.is_none() {
                        info[m].zero_fee_slot = Some(t.slot[i]);
                    }
                }
                Kind::Complete => info[m].complete_slot = Some(t.slot[i]),
                Kind::Migrate => info[m].pool = Some(t.pools.names[t.pool[i] as usize].clone()),
                _ => {}
            }
        }
        // Pass 2: count then fill per-mint states (curve trades + canonical-pool AMM swaps).
        let keep = |i: usize| -> bool {
            match t.kind[i] {
                Kind::Trade => true,
                Kind::AmmBuy | Kind::AmmSell => {
                    let m = t.mint[i] as usize;
                    info[m].pool.as_deref() == Some(t.pools.names[t.pool[i] as usize].as_str())
                }
                _ => false,
            }
        };
        let mut counts = vec![0usize; n_mints + 1];
        for i in 0..t.len() {
            if keep(i) {
                counts[t.mint[i] as usize + 1] += 1;
            }
        }
        for m in 0..n_mints {
            counts[m + 1] += counts[m];
        }
        let total = counts[n_mints];
        let mut cursor = counts.clone();
        let mut slot = vec![0u64; total];
        let mut pool = vec![Pool::default(); total];
        let mut amm = vec![false; total];
        for i in 0..t.len() {
            if !keep(i) {
                continue;
            }
            let m = t.mint[i] as usize;
            let k = cursor[m];
            cursor[m] += 1;
            slot[k] = t.slot[i];
            pool[k] = Pool {
                sol: t.vsol[i],
                tok: t.vtok[i],
                fee_bps: t.fee_bps[i],
            };
            amm[k] = t.kind[i] != Kind::Trade;
        }
        Timeline {
            offsets: counts,
            slot,
            pool,
            amm,
            info,
        }
    }

    pub fn range(&self, mint: u32) -> std::ops::Range<usize> {
        self.offsets[mint as usize]..self.offsets[mint as usize + 1]
    }

    /// Last state of `mint` with state.slot <= `slot` (i.e. at the end of `slot`).
    pub fn state_at_end_of(&self, mint: u32, slot: u64) -> Option<(Pool, bool)> {
        let r = self.range(mint);
        let s = &self.slot[r.clone()];
        let n = s.partition_point(|&x| x <= slot);
        if n == 0 {
            return None;
        }
        let k = r.start + n - 1;
        Some((self.pool[k], self.amm[k]))
    }

    /// Last slot with any state for this mint.
    pub fn last_slot(&self, mint: u32) -> Option<u64> {
        let r = self.range(mint);
        if r.is_empty() {
            None
        } else {
            Some(self.slot[r.end - 1])
        }
    }
}
