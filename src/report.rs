//! Backtest metrics. All P&L is net of swap fees, base + priority fees and
//! failed-transaction costs. Censored trades (exit beyond the data) are
//! excluded and counted.

use crate::exec::RoundTrip;
use serde::Serialize;

#[derive(Serialize, Debug, Default)]
pub struct Report {
    pub strategy: String,
    pub signals: usize,
    pub entered: usize,
    pub failed_entries: usize,
    pub slippage_rejects: usize,
    pub graduated_skips: usize,
    pub censored: usize,
    pub net_pnl_sol: f64,
    pub cost_sol: f64,
    pub mean_ret: f64,
    pub median_ret: f64,
    pub ret_t_stat: f64,
    pub hit_rate: f64,
    /// Sharpe of hourly P&L (sum of trade P&L by exit hour), annualised by sqrt(24*365).
    pub sharpe_hourly_ann: f64,
    pub max_drawdown_sol: f64,
    pub turnover_sol: f64,
    pub exits_on_amm: usize,
}

pub fn summarize(name: &str, trades: &[RoundTrip], size: u64, slot_secs: f64) -> Report {
    let mut r = Report {
        strategy: name.to_string(),
        signals: trades.len(),
        ..Default::default()
    };
    let mut rets = Vec::new();
    let mut by_exit: Vec<(u64, f64)> = Vec::new();
    for t in trades {
        match t.skip_reason {
            "fail" => r.failed_entries += 1,
            "slippage" => r.slippage_rejects += 1,
            "graduated" => r.graduated_skips += 1,
            _ => {}
        }
        if t.censored {
            r.censored += 1;
            continue;
        }
        let pnl = t.pnl() as f64 / 1e9;
        r.net_pnl_sol += pnl;
        r.cost_sol += (t.swap_fees + t.tx_costs) as f64 / 1e9;
        if t.entered {
            r.entered += 1;
            r.turnover_sol += (t.sol_in + t.sol_out) as f64 / 1e9;
            rets.push(t.ret(size));
            if t.exit_on_amm {
                r.exits_on_amm += 1;
            }
            by_exit.push((t.exit_slot, pnl));
        } else if t.tx_costs > 0 {
            by_exit.push((t.entry_slot, pnl));
        }
    }
    if !rets.is_empty() {
        let n = rets.len() as f64;
        r.mean_ret = rets.iter().sum::<f64>() / n;
        let var = rets.iter().map(|x| (x - r.mean_ret).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        r.ret_t_stat = r.mean_ret / (var.sqrt() / n.sqrt()).max(1e-12);
        r.hit_rate = rets.iter().filter(|&&x| x > 0.0).count() as f64 / n;
        let mut s = rets.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        r.median_ret = s[s.len() / 2];
    }
    by_exit.sort_by_key(|x| x.0);
    let (mut cum, mut peak) = (0.0f64, 0.0f64);
    for &(_, p) in &by_exit {
        cum += p;
        peak = peak.max(cum);
        r.max_drawdown_sol = r.max_drawdown_sol.max(peak - cum);
    }
    if let (Some(first), Some(last)) = (by_exit.first(), by_exit.last()) {
        let slots_per_hour = (3600.0 / slot_secs) as u64;
        let nh = ((last.0 - first.0) / slots_per_hour + 1) as usize;
        let mut hourly = vec![0.0f64; nh];
        for &(s, p) in &by_exit {
            hourly[((s - first.0) / slots_per_hour) as usize] += p;
        }
        if nh > 1 {
            let mu = hourly.iter().sum::<f64>() / nh as f64;
            let sd =
                (hourly.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (nh as f64 - 1.0)).sqrt();
            if sd > 0.0 {
                r.sharpe_hourly_ann = mu / sd * (24.0f64 * 365.0).sqrt();
            }
        }
    }
    r
}
