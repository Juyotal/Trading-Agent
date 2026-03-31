from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.backtest.fill_sim import SimulatedTrade


@dataclass
class BacktestStats:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_consecutive_losses: int
    stop_exits: int
    target_exits: int
    eod_exits: int


def compute_stats(trades: List[SimulatedTrade]) -> BacktestStats:
    if not trades:
        return BacktestStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in trades)
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))

    # Equity curve → max drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    # Consecutive losses
    streak = 0
    worst_streak = 0
    for t in trades:
        if t.pnl <= 0:
            streak += 1
            worst_streak = max(worst_streak, streak)
        else:
            streak = 0

    return BacktestStats(
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trades),
        total_pnl=total_pnl,
        avg_win=(gross_profit / len(wins)) if wins else 0.0,
        avg_loss=(-gross_loss / len(losses)) if losses else 0.0,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        max_drawdown=max_dd,
        max_consecutive_losses=worst_streak,
        stop_exits=sum(1 for t in trades if t.exit_reason == "STOP"),
        target_exits=sum(1 for t in trades if t.exit_reason == "TARGET"),
        eod_exits=sum(1 for t in trades if t.exit_reason == "END_OF_DATA"),
    )


def format_report(stats: BacktestStats) -> str:
    if stats.total_trades == 0:
        return "No trades executed."

    lines = [
        "=" * 60,
        "BACKTEST RESULTS",
        "=" * 60,
        f"Total trades:          {stats.total_trades}",
        f"Win rate:              {stats.win_rate:.1%} ({stats.wins}W / {stats.losses}L)",
        f"Total PnL:             ${stats.total_pnl:,.2f}",
        f"Avg win:               ${stats.avg_win:,.2f}",
        f"Avg loss:              ${stats.avg_loss:,.2f}",
        f"Profit factor:         {stats.profit_factor:.2f}",
        f"Max drawdown:          ${stats.max_drawdown:,.2f}",
        f"Max consec losses:     {stats.max_consecutive_losses}",
        f"Exits: stop={stats.stop_exits} target={stats.target_exits} eod={stats.eod_exits}",
        "=" * 60,
    ]
    return "\n".join(lines)
