//! Per-mint market timeline and universe metadata.
//!
//! The timeline answers "what was this token's pool state at the END of slot
//! s?" by binary search. Both the label generator and the event-driven engine
//! fill orders through [`Timeline::state_at_end_of`]-equivalent semantics, and
//! a test asserts they agree.

use crate::amm::Pool;
use crate::events::Kind;
use crate::store::Tape;

/// Initial virtual SOL reserves of a standard SOL-quoted pump.fun curve.
pub const STANDARD_VIRTUAL_SOL: u64 = 30_000_000_000;

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
    /// Launched as a standard SOL-quoted curve (CreateEvent virtual SOL = 30 SOL).
    /// Curves quoted in other assets report quote-asset units in the SOL fields, so
    /// SOL-denominated execution cannot be simulated on them.
    pub sol_quoted: bool,
    /// First slot with a zero-fee curve trade (Mayhem-agent activity on a token
    /// not flagged at launch). New decisions stop from this slot on.
    pub zero_fee_slot: Option<u64>,
    pub complete_slot: Option<u64>,
    /// Canonical PumpSwap pool (from the migration event), interned in `pools`.
    pub pool: Option<String>,
    pub n_curve_trades: u32,
}

impl MintInfo {
    /// Launched inside the window (full history known), SOL-quoted, not Mayhem
    /// mode. Every input is a launch-time fact.
    pub fn in_cohort(&self) -> bool {
        self.create_slot.is_some() && self.sol_quoted && !self.mayhem
    }

    /// Point-in-time tradability at the end of `slot`: in cohort, still on the
    /// bonding curve, and no zero-fee (non-reconstructible) trade seen yet.
    pub fn tradable_at(&self, slot: u64) -> bool {
        self.in_cohort()
            && self.complete_slot.is_none_or(|c| c > slot)
            && self.zero_fee_slot.is_none_or(|z| z > slot)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct State {
    pub slot: u64,
    pub pool: Pool,
    /// Venue: false = bonding curve, true = PumpSwap.
    pub amm: bool,
}

/// Per-mint state history. Append-only, so the same structure serves batch
/// replay (`build`) and the live paper trader (`push` as events arrive).
/// Metadata in `info` is only ever compared against the query slot, so a
/// timeline built from the full tape answers every query exactly as an
/// incrementally built one would at that point in time.
#[derive(Default)]
pub struct Timeline {
    pub states: Vec<Vec<State>>,
    pub info: Vec<MintInfo>,
}

impl Timeline {
    pub fn build(t: &Tape) -> Timeline {
        let mut tl = Timeline::default();
        for i in 0..t.len() {
            tl.push(t, i);
        }
        tl.ensure(t.mints.names.len());
        tl
    }

    fn ensure(&mut self, n_mints: usize) {
        if self.info.len() < n_mints {
            self.info.resize(n_mints, MintInfo::default());
            self.states.resize(n_mints, Vec::new());
        }
    }

    /// Ingest event i (must be called in tape order).
    pub fn push(&mut self, t: &Tape, i: usize) {
        let m = t.mint[i] as usize;
        self.ensure(m + 1);
        let info = &mut self.info[m];
        let state_event = match t.kind[i] {
            Kind::Create => {
                info.create_slot = Some(t.slot[i]);
                info.creator = Some(t.user[i]);
                info.mayhem = t.is_buy[i];
                info.sol_quoted = t.vsol[i] == STANDARD_VIRTUAL_SOL;
                false
            }
            Kind::Trade => {
                info.n_curve_trades += 1;
                if t.fee_bps[i] == 0 && info.zero_fee_slot.is_none() {
                    info.zero_fee_slot = Some(t.slot[i]);
                }
                true
            }
            Kind::Complete => {
                info.complete_slot = Some(t.slot[i]);
                false
            }
            Kind::Migrate => {
                info.pool = Some(t.pools.names[t.pool[i] as usize].clone());
                false
            }
            // Only the canonical (migration) pool defines a graduated token's state.
            Kind::AmmBuy | Kind::AmmSell => {
                info.pool.as_deref() == Some(t.pools.names[t.pool[i] as usize].as_str())
            }
            Kind::Failed => false,
        };
        if state_event {
            self.states[m].push(State {
                slot: t.slot[i],
                pool: Pool {
                    sol: t.vsol[i],
                    tok: t.vtok[i],
                    fee_bps: t.fee_bps[i],
                },
                amm: t.kind[i] != Kind::Trade,
            });
        }
    }

    /// Whether event i updates its mint's state (same rule as `push`).
    pub fn is_state_event(&self, t: &Tape, i: usize) -> bool {
        match t.kind[i] {
            Kind::Trade => true,
            Kind::AmmBuy | Kind::AmmSell => {
                self.info[t.mint[i] as usize].pool.as_deref()
                    == Some(t.pools.names[t.pool[i] as usize].as_str())
            }
            _ => false,
        }
    }

    /// Last state of `mint` with state.slot <= `slot` (i.e. at the end of `slot`).
    pub fn state_at_end_of(&self, mint: u32, slot: u64) -> Option<(Pool, bool)> {
        let s = self.states.get(mint as usize)?;
        let n = s.partition_point(|x| x.slot <= slot);
        if n == 0 {
            return None;
        }
        Some((s[n - 1].pool, s[n - 1].amm))
    }

    /// Last slot with any state for this mint.
    pub fn last_slot(&self, mint: u32) -> Option<u64> {
        self.states.get(mint as usize)?.last().map(|s| s.slot)
    }
}
