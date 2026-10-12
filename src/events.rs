//! Canonical event row. One table, ordered by (slot, tx_idx, ev_idx), which is
//! the on-chain execution order. Every reserve field means "state AFTER this
//! event", so a replay never needs to look ahead to know the current state.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Kind {
    /// Token launched on the bonding curve.
    #[default]
    Create = 0,
    /// Bonding-curve buy/sell.
    Trade = 1,
    /// Curve reached its graduation threshold; no more curve trades.
    Complete = 2,
    /// Liquidity migrated to a PumpSwap pool (`pool` set).
    Migrate = 3,
    /// PumpSwap buy (SOL -> token).
    AmmBuy = 4,
    /// PumpSwap sell.
    AmmSell = 5,
    /// Failed transaction that attempted a buy/sell (`symbol` holds the error).
    Failed = 6,
}

impl Kind {
    pub fn from_u8(v: u8) -> Kind {
        match v {
            0 => Kind::Create,
            1 => Kind::Trade,
            2 => Kind::Complete,
            3 => Kind::Migrate,
            4 => Kind::AmmBuy,
            5 => Kind::AmmSell,
            _ => Kind::Failed,
        }
    }
    pub fn is_swap(self) -> bool {
        matches!(self, Kind::Trade | Kind::AmmBuy | Kind::AmmSell)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Event {
    pub slot: u64,
    pub block_time: i64,
    pub tx_idx: u32,
    pub ev_idx: u16,
    pub kind: Kind,
    pub mint: String,
    pub user: String,
    /// PumpSwap pool address (AMM/migrate), or "curve"/"amm" venue tag for failed attempts.
    pub pool: String,
    /// Token symbol (Create) or error string (Failed).
    pub symbol: String,
    /// Swap direction. For Create rows: the token's is_mayhem_mode flag.
    pub is_buy: bool,
    /// Lamports into (buy) / out of (sell) the curve, excluding fees.
    pub sol_amount: u64,
    /// Raw token units (6 decimals).
    pub token_amount: u64,
    /// AMM only: total lamports paid/received by the user including fees.
    pub user_sol: u64,
    /// Post-event virtual reserves (curve) or pool reserves (AMM).
    pub vsol: u64,
    pub vtok: u64,
    /// Post-event real reserves (curve). Equal to v* for AMM rows.
    pub rsol: u64,
    pub rtok: u64,
    /// Total swap fee in lamports and its rate in bps (protocol + creator [+ LP]).
    pub fee: u64,
    pub fee_bps: u32,
    /// Event's own unix timestamp (program clock).
    pub ts: i64,
    /// Transaction-level fee paid (lamports) and the part above the 5000/sig base fee.
    pub tx_fee: u64,
    pub priority_fee: u64,
    pub cu: u64,
}
