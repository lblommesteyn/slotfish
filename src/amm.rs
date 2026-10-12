//! Constant-product swap math for the pump.fun bonding curve and PumpSwap pools.
//!
//! Both venues price as x*y=k on (sol_reserve, token_reserve): the curve on
//! its *virtual* reserves, PumpSwap on real pool balances. Fees are charged on
//! the SOL side, on top of the curve amount for buys and out of the proceeds
//! for sells. Integer rounding follows the programs (floor on the output
//! side), so a simulated fill is never better than the chain would give.

/// Market state for one token at one point in time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pool {
    pub sol: u64,
    pub tok: u64,
    /// Total swap fee in basis points (protocol + creator [+ LP]).
    pub fee_bps: u32,
}

impl Pool {
    /// Price in lamports per raw token unit.
    pub fn price(&self) -> f64 {
        if self.tok == 0 {
            return f64::NAN;
        }
        self.sol as f64 / self.tok as f64
    }

    /// Spend `budget` lamports in total (fee included). Returns (tokens_out, fee_paid).
    pub fn buy_with_budget(&self, budget: u64) -> (u64, u64) {
        // budget = net * (1 + fee) => net = budget * 10000 / (10000 + fee_bps)
        let net = (budget as u128 * 10_000 / (10_000 + self.fee_bps as u128)) as u64;
        let fee = budget - net;
        let out = (net as u128 * self.tok as u128 / (self.sol as u128 + net as u128)) as u64;
        (out, fee)
    }

    /// Sell `tokens`. Returns (lamports received after fee, fee_paid).
    pub fn sell(&self, tokens: u64) -> (u64, u64) {
        let gross =
            (tokens as u128 * self.sol as u128 / (self.tok as u128 + tokens as u128)) as u64;
        let fee = (gross as u128 * self.fee_bps as u128).div_ceil(10_000) as u64;
        (gross.saturating_sub(fee), fee)
    }

    /// State after a buy of `net` lamports (fee excluded) into the pool.
    pub fn after_buy(&self, net: u64) -> Pool {
        let out = (net as u128 * self.tok as u128 / (self.sol as u128 + net as u128)) as u64;
        Pool {
            sol: self.sol + net,
            tok: self.tok - out,
            fee_bps: self.fee_bps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_curve() -> Pool {
        // Initial pump.fun curve: 30 SOL virtual, 1.073e15 raw token units virtual.
        Pool {
            sol: 30_000_000_000,
            tok: 1_073_000_000_000_000,
            fee_bps: 125,
        }
    }

    #[test]
    fn round_trip_loses_exactly_fees_and_rounding() {
        let p = fresh_curve();
        let budget = 1_000_000_000; // 1 SOL
        let (tok, fee_in) = p.buy_with_budget(budget);
        let after = p.after_buy(budget - fee_in);
        let (back, fee_out) = after.sell(tok);
        assert!(back < budget);
        let lost = budget - back;
        // Loss is the two fees plus at most a couple of lamports of rounding.
        assert!(lost >= fee_in + fee_out);
        assert!(
            lost <= fee_in + fee_out + 2,
            "lost {lost} fees {}",
            fee_in + fee_out
        );
    }

    #[test]
    fn price_impact_increases_with_size() {
        let p = fresh_curve();
        let (a, _) = p.buy_with_budget(100_000_000);
        let (b, _) = p.buy_with_budget(1_000_000_000);
        // Tokens per lamport falls with size.
        assert!((b as f64 / 1e9) < (a as f64 / 1e8));
    }

    #[test]
    fn sell_into_unmoved_pool_is_conservative() {
        // Selling against the pre-buy state (no feedback from our own buy)
        // returns less than selling against the post-buy state.
        let p = fresh_curve();
        let (tok, fee) = p.buy_with_budget(500_000_000);
        let (no_feedback, _) = p.sell(tok);
        let (with_feedback, _) = p.after_buy(500_000_000 - fee).sell(tok);
        assert!(no_feedback < with_feedback);
    }
}
