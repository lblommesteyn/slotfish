//! Streaming, point-in-time feature engine.
//!
//! `FeatureEngine::update(t, tl, i)` must be called for every event in tape order.
//! `FeatureEngine::features(t, tl, i)` may only be called right after `update` and
//! uses nothing but events 0..=i plus *past* timeline states, so every feature
//! is available at the decision instant by construction.
//!
//! Wallet skill is a Beta-Binomial posterior on "a curve buy by this wallet is
//! followed by a +HURDLE price move within `skill_horizon` slots". An outcome is
//! only revealed to the posterior once the replay clock passes
//! buy_slot + skill_horizon, so skill never peeks at the future.
//!
//! Coordination: a wallet pair is "linked" when both wallets were among the
//! first EARLY_N distinct buyers of at least COORD_MIN earlier tokens. Pair
//! counts are incremented the moment the second wallet's early buy happens,
//! and the current token's own contribution is subtracted, so the feature
//! only reflects co-buying in *previous* tokens.

use crate::events::Kind;
use crate::market::Timeline;
use crate::store::Tape;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

pub const SKILL_HURDLE: f64 = 0.03; // must beat round-trip fees to count as a win
pub const PRIOR_A: f64 = 1.0;
pub const PRIOR_B: f64 = 4.0;
const WINDOW: u64 = 300;
const EARLY_N: usize = 20;
const COORD_MIN: u32 = 2;
const COORD_RECENT: usize = 20;

pub const FEATURE_NAMES: &[&str] = &[
    "age_slots",
    "log_mcap_sol",
    "progress",
    "n_trades",
    "n_buyers",
    "ret_10",
    "ret_75",
    "ret_300",
    "drawdown_from_peak",
    "buys_10",
    "sells_10",
    "buys_75",
    "sells_75",
    "netflow_10_sol",
    "netflow_75_sol",
    "netflow_300_sol",
    "vol_75_sol",
    "uniq_buyers_75",
    "buy_share_75",
    "slots_since_last",
    "last_size_sol",
    "last_is_buy",
    "fails_75",
    "creator_sold",
    "creator_hold_frac",
    "same_slot_buyers",
    "launch_snipers",
    "skill_last_buyer",
    "skill_n_last_buyer",
    "skill_wavg_75",
    "smart_buys_75",
    "smart_sol_75",
    "coord_pairs_75",
    "coord_wallets_75",
];

#[derive(Clone, Copy)]
struct Tick {
    slot: u64,
    is_buy: bool,
    sol: u64,
    user: u32,
}

#[derive(Default)]
struct MintState {
    n_trades: u32,
    buyers: HashSet<u32>,
    ticks: VecDeque<Tick>,
    fails: VecDeque<u64>,
    peak_sol: u64,
    creator_tokens: i128,
    creator_sold: bool,
    snipers: u32,
    /// First EARLY_N distinct non-creator buyers.
    early: Vec<u32>,
}

#[derive(Default, Clone, Copy)]
struct Skill {
    wins: f64,
    n: f64,
}

impl Skill {
    fn mean(&self) -> f64 {
        (PRIOR_A + self.wins) / (PRIOR_A + PRIOR_B + self.n)
    }
}

pub struct FeatureEngine {
    mints: Vec<MintState>,
    skill: HashMap<u32, Skill>,
    /// (resolve_slot, mint, user, entry_price) pending skill outcomes.
    pending: BinaryHeap<Reverse<(u64, u32, u32, u64)>>,
    /// Wallet pair (lo << 32 | hi) -> (tokens co-bought early, last mint counted).
    pairs: HashMap<u64, (u32, u32)>,
    pub skill_horizon: u64,
    pub min_smart_n: f64,
    pub smart_threshold: f64,
}

const TOTAL_SUPPLY: f64 = 1e15; // 1B tokens, 6 decimals

fn pair_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

impl FeatureEngine {
    pub fn new(skill_horizon: u64) -> Self {
        FeatureEngine {
            mints: Vec::new(),
            skill: HashMap::new(),
            pending: BinaryHeap::new(),
            pairs: HashMap::new(),
            skill_horizon,
            min_smart_n: 5.0,
            smart_threshold: 0.35,
        }
    }

    fn price_bits(p: f64) -> u64 {
        p.to_bits()
    }

    fn resolve_until(&mut self, tl: &Timeline, slot: u64) {
        while let Some(Reverse((rs, mint, user, pbits))) = self.pending.peek().copied() {
            if rs > slot {
                break;
            }
            self.pending.pop();
            let entry = f64::from_bits(pbits);
            let win = match tl.state_at_end_of(mint, rs) {
                Some((p, _)) => p.price() > entry * (1.0 + SKILL_HURDLE),
                None => false,
            };
            let s = self.skill.entry(user).or_default();
            s.n += 1.0;
            if win {
                s.wins += 1.0;
            }
        }
    }

    pub fn wallet_skill(&self, user: u32) -> (f64, f64) {
        let s = self.skill.get(&user).copied().unwrap_or_default();
        (s.mean(), s.n)
    }

    /// Ingest event i (must be called in tape order, after `tl` has seen it).
    pub fn update(&mut self, t: &Tape, tl: &Timeline, i: usize) {
        let slot = t.slot[i];
        // Outcomes resolving strictly before this slot are known now. Using `slot - 1`
        // keeps same-slot information out: the end-of-slot state is not yet final.
        self.resolve_until(tl, slot.saturating_sub(1));
        let m = t.mint[i] as usize;
        if self.mints.len() <= m {
            self.mints.resize_with(m + 1, MintState::default);
        }
        match t.kind[i] {
            Kind::Trade => {
                let info = &tl.info[m];
                let ms = &mut self.mints[m];
                ms.n_trades += 1;
                ms.peak_sol = ms.peak_sol.max(t.vsol[i]);
                let user = t.user[i];
                if t.is_buy[i] {
                    let new_buyer = ms.buyers.insert(user);
                    if info.create_slot == Some(slot) && info.creator != Some(user) {
                        ms.snipers += 1;
                    }
                    if new_buyer
                        && info.in_cohort()
                        && info.creator != Some(user)
                        && ms.early.len() < EARLY_N
                    {
                        for &e in &ms.early {
                            let c = self.pairs.entry(pair_key(e, user)).or_insert((0, u32::MAX));
                            if c.1 != m as u32 {
                                c.0 += 1;
                                c.1 = m as u32;
                            }
                        }
                        ms.early.push(user);
                    }
                }
                if info.creator == Some(user) {
                    if t.is_buy[i] {
                        ms.creator_tokens += t.token_amount[i] as i128;
                    } else {
                        ms.creator_tokens -= t.token_amount[i] as i128;
                        ms.creator_sold = true;
                    }
                }
                ms.ticks.push_back(Tick {
                    slot,
                    is_buy: t.is_buy[i],
                    sol: t.sol_amount[i],
                    user,
                });
                while ms.ticks.front().is_some_and(|k| k.slot + WINDOW < slot) {
                    ms.ticks.pop_front();
                }
                if t.is_buy[i] && info.in_cohort() && t.vtok[i] > 0 {
                    let p = t.vsol[i] as f64 / t.vtok[i] as f64;
                    self.pending.push(Reverse((
                        slot + self.skill_horizon,
                        m as u32,
                        user,
                        Self::price_bits(p),
                    )));
                }
            }
            Kind::Failed => {
                let ms = &mut self.mints[m];
                ms.fails.push_back(slot);
                while ms.fails.front().is_some_and(|&s| s + 75 < slot) {
                    ms.fails.pop_front();
                }
            }
            _ => {}
        }
    }

    /// Feature vector for the state right after event i (a curve Trade).
    pub fn features(&self, t: &Tape, tl: &Timeline, i: usize, out: &mut Vec<f32>) {
        let slot = t.slot[i];
        let m = t.mint[i];
        let ms = &self.mints[m as usize];
        let info = &tl.info[m as usize];
        let price = t.vsol[i] as f64 / t.vtok[i].max(1) as f64;
        let past_ret = |w: u64| -> f64 {
            match tl.state_at_end_of(m, slot.saturating_sub(w + 1)) {
                Some((p, _)) => (price / p.price()).ln(),
                None => 0.0,
            }
        };
        let (mut b10, mut s10, mut b75, mut s75) = (0u32, 0u32, 0u32, 0u32);
        let (mut nf10, mut nf75, mut nf300, mut vol75) = (0f64, 0f64, 0f64, 0f64);
        let mut uniq75: Vec<u32> = Vec::new();
        let mut same_slot: Vec<u32> = Vec::new();
        let (mut skill_wsum, mut skill_w, mut smart_buys, mut smart_sol) = (0f64, 0f64, 0u32, 0f64);
        for k in ms.ticks.iter() {
            let sol = k.sol as f64 / 1e9;
            let signed = if k.is_buy { sol } else { -sol };
            let age = slot - k.slot;
            nf300 += signed;
            if age <= 75 {
                nf75 += signed;
                vol75 += sol;
                if k.is_buy {
                    b75 += 1;
                    uniq75.push(k.user);
                    let (sm, sn) = self.wallet_skill(k.user);
                    skill_wsum += sm * sol;
                    skill_w += sol;
                    if sn >= self.min_smart_n && sm >= self.smart_threshold {
                        smart_buys += 1;
                        smart_sol += sol;
                    }
                } else {
                    s75 += 1;
                }
            }
            if age <= 10 {
                nf10 += signed;
                if k.is_buy {
                    b10 += 1;
                } else {
                    s10 += 1;
                }
            }
            if age == 0 && k.is_buy {
                same_slot.push(k.user);
            }
        }
        uniq75.sort_unstable();
        uniq75.dedup();
        same_slot.sort_unstable();
        same_slot.dedup();
        // Coordination among the most recent distinct buyers in the 75-slot window.
        let mut recent: Vec<u32> = Vec::with_capacity(COORD_RECENT);
        for k in ms.ticks.iter().rev() {
            if slot - k.slot > 75 || recent.len() == COORD_RECENT {
                break;
            }
            if k.is_buy && !recent.contains(&k.user) {
                recent.push(k.user);
            }
        }
        let (mut coord_pairs, mut coord_wallets) = (0u32, Vec::new());
        for a in 0..recent.len() {
            for b in a + 1..recent.len() {
                if let Some(&(c, last_m)) = self.pairs.get(&pair_key(recent[a], recent[b])) {
                    let prev = c - u32::from(last_m == m);
                    if prev >= COORD_MIN {
                        coord_pairs += 1;
                        coord_wallets.push(recent[a]);
                        coord_wallets.push(recent[b]);
                    }
                }
            }
        }
        coord_wallets.sort_unstable();
        coord_wallets.dedup();
        let last = ms.ticks.back().copied();
        let prev_slot = ms.ticks.iter().rev().nth(1).map(|k| k.slot);
        let (sk, skn) = if t.is_buy[i] {
            self.wallet_skill(t.user[i])
        } else {
            (f64::NAN, f64::NAN)
        };
        let mcap_sol = price * TOTAL_SUPPLY / 1e9;
        let real_sol = t.rsol[i] as f64 / 1e9;
        let fails75 = ms.fails.iter().filter(|&&s| s + 75 >= slot).count();
        out.clear();
        out.extend_from_slice(&[
            info.create_slot
                .map(|c| (slot - c) as f32)
                .unwrap_or(f32::NAN),
            mcap_sol.ln() as f32,
            (real_sol / 85.0) as f32,
            ms.n_trades as f32,
            ms.buyers.len() as f32,
            past_ret(10) as f32,
            past_ret(75) as f32,
            past_ret(300) as f32,
            (t.vsol[i] as f64 / ms.peak_sol.max(1) as f64 - 1.0) as f32,
            b10 as f32,
            s10 as f32,
            b75 as f32,
            s75 as f32,
            nf10 as f32,
            nf75 as f32,
            nf300 as f32,
            vol75 as f32,
            uniq75.len() as f32,
            if b75 + s75 > 0 {
                b75 as f32 / (b75 + s75) as f32
            } else {
                f32::NAN
            },
            prev_slot.map(|p| (slot - p) as f32).unwrap_or(f32::NAN),
            last.map(|k| k.sol as f32 / 1e9).unwrap_or(f32::NAN),
            last.map(|k| k.is_buy as u8 as f32).unwrap_or(f32::NAN),
            fails75 as f32,
            ms.creator_sold as u8 as f32,
            (ms.creator_tokens.max(0) as f64 / TOTAL_SUPPLY) as f32,
            same_slot.len() as f32,
            ms.snipers as f32,
            sk as f32,
            skn as f32,
            if skill_w > 0.0 {
                (skill_wsum / skill_w) as f32
            } else {
                f32::NAN
            },
            smart_buys as f32,
            smart_sol as f32,
            coord_pairs as f32,
            coord_wallets.len() as f32,
        ]);
        debug_assert_eq!(out.len(), FEATURE_NAMES.len());
    }
}
