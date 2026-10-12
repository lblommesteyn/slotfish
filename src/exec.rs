//! Execution model: latency, fees, priority tips, slippage limits, failures.
//!
//! Semantics (documented in docs/EXECUTION.md):
//!   * A decision taken after observing an event in slot `s` lands in slot
//!     `s + latency_slots` and fills against the token's state at the END of
//!     that slot, i.e. after every observed trade in the landing slot. Bots
//!     that land in the same slot are assumed to be ahead of us.
//!   * Our own trades do not feed back into the replayed pool (the observed
//!     history is fixed). For a round trip this is conservative: the exit sells
//!     into a pool that never received our buy, so impact is paid twice.
//!   * Every landed transaction pays base fee + priority fee whether or not it
//!     succeeds. Entries fail if the fill is worse than the decision-time quote
//!     by more than `max_slippage_bps`, or at random with probability `p_fail`.
//!     Exits never fail on slippage (sell at any price) but can fail at random;
//!     a failed exit retries one slot later and pays fees again.

use crate::market::Timeline;

#[derive(Clone, Debug)]
pub struct ExecConfig {
    pub latency_slots: u64,
    pub size_lamports: u64,
    pub base_fee: u64,
    pub priority_fee: u64,
    pub p_fail: f64,
    pub max_slippage_bps: u32,
    pub seed: u64,
    pub max_exit_retries: u32,
}

impl Default for ExecConfig {
    fn default() -> Self {
        ExecConfig {
            latency_slots: 2,
            size_lamports: 100_000_000, // 0.1 SOL
            base_fee: 5_000,
            priority_fee: 100_000,
            p_fail: 0.05,
            max_slippage_bps: 2_000,
            seed: 7,
            max_exit_retries: 20,
        }
    }
}

impl ExecConfig {
    pub fn tx_cost(&self) -> u64 {
        self.base_fee + self.priority_fee
    }
}

/// splitmix64: deterministic per-(seed, key) uniform draws, independent of replay order.
pub fn uniform(seed: u64, a: u64, b: u64) -> f64 {
    let mut z =
        seed ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoundTrip {
    pub mint: u32,
    pub decision_slot: u64,
    pub entry_slot: u64,
    pub exit_slot: u64,
    pub entered: bool,
    /// Reason an entry did not happen: "fail", "slippage", "graduated", "nostate".
    pub skip_reason: &'static str,
    pub tokens: u64,
    pub sol_in: u64,
    pub sol_out: u64,
    pub swap_fees: u64,
    pub tx_costs: u64,
    pub exit_attempts: u32,
    pub exit_on_amm: bool,
    /// Exit slot is beyond the data: outcome unknown, must be excluded.
    pub censored: bool,
}

impl RoundTrip {
    /// Net P&L in lamports, all costs included.
    pub fn pnl(&self) -> i64 {
        self.sol_out as i64 - self.sol_in as i64 - self.tx_costs as i64
    }
    pub fn ret(&self, size: u64) -> f64 {
        self.pnl() as f64 / size as f64
    }
}

/// Entry leg. `quote_pool` is the state the decision observed.
pub fn enter(
    tl: &Timeline,
    cfg: &ExecConfig,
    mint: u32,
    decision_slot: u64,
    quote: crate::amm::Pool,
) -> RoundTrip {
    let mut rt = RoundTrip {
        mint,
        decision_slot,
        ..Default::default()
    };
    let entry_slot = decision_slot + cfg.latency_slots;
    rt.entry_slot = entry_slot;
    let Some((pool, on_amm)) = tl.state_at_end_of(mint, entry_slot) else {
        rt.skip_reason = "nostate";
        return rt;
    };
    if on_amm
        || tl.info[mint as usize]
            .complete_slot
            .is_some_and(|c| c <= entry_slot)
    {
        rt.skip_reason = "graduated";
        return rt;
    }
    rt.tx_costs = cfg.tx_cost();
    if uniform(cfg.seed, mint as u64, decision_slot) < cfg.p_fail {
        rt.skip_reason = "fail";
        return rt;
    }
    let (quoted, _) = quote.buy_with_budget(cfg.size_lamports);
    let (tokens, fee) = pool.buy_with_budget(cfg.size_lamports);
    if (tokens as f64) < quoted as f64 * (1.0 - cfg.max_slippage_bps as f64 / 1e4) {
        rt.skip_reason = "slippage";
        return rt;
    }
    rt.entered = true;
    rt.tokens = tokens;
    rt.sol_in = cfg.size_lamports;
    rt.swap_fees = fee;
    rt
}

/// Exit leg: decide to sell at `decide_slot`; lands latency later; retries on failure.
pub fn exit(
    tl: &Timeline,
    cfg: &ExecConfig,
    rt: &mut RoundTrip,
    decide_slot: u64,
    data_end_slot: u64,
) {
    let mut land = decide_slot + cfg.latency_slots;
    for attempt in 0..=cfg.max_exit_retries {
        if land > data_end_slot {
            rt.censored = true;
            return;
        }
        rt.exit_attempts = attempt + 1;
        rt.tx_costs += cfg.tx_cost();
        if uniform(cfg.seed ^ 0x5E11, rt.mint as u64, land) >= cfg.p_fail {
            let (pool, on_amm) = tl
                .state_at_end_of(rt.mint, land)
                .expect("entered => state exists");
            let (out, fee) = pool.sell(rt.tokens);
            rt.sol_out = out;
            rt.swap_fees += fee;
            rt.exit_slot = land;
            rt.exit_on_amm = on_amm;
            return;
        }
        land += 1;
    }
    // Exhausted retries: position is written off at zero (conservative).
    rt.exit_slot = land;
}

/// Full fixed-horizon round trip, the unit of every label and backtest trade.
pub fn round_trip(
    tl: &Timeline,
    cfg: &ExecConfig,
    mint: u32,
    decision_slot: u64,
    quote: crate::amm::Pool,
    hold_slots: u64,
    data_end_slot: u64,
) -> RoundTrip {
    let mut rt = enter(tl, cfg, mint, decision_slot, quote);
    if rt.entered {
        let decide = rt.entry_slot + hold_slots;
        exit(tl, cfg, &mut rt, decide, data_end_slot);
    }
    rt
}
