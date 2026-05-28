from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import snapshot_from_series
from .signals import TradeSignal


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    risk_level: str
    position_usd: float
    max_loss_usd: float
    notes: list[str]


def evaluate_risk(signal: TradeSignal, series: pd.DataFrame, capital: float = 10_000, risk_per_trade: float = 0.01) -> RiskDecision:
    s = snapshot_from_series(series)
    notes: list[str] = []
    allowed = True

    if signal.action == "HOLD":
        allowed = False
        notes.append("HOLD signal: stay flat and wait for a cleaner setup.")
    if signal.confidence < 0.48:
        allowed = False
        notes.append("Confidence is below the 48% execution threshold.")
    if s.volatility > 95:
        allowed = False
        notes.append(f"Volatility is too high: {s.volatility:.1f}% annualized.")
    if s.sentiment < 22 and signal.action == "BUY":
        allowed = False
        notes.append("Panic filter: sentiment is too low for bottom-fishing.")
    if s.volume_ratio > 2.25:
        notes.append("Strong volume spike: size is reduced to avoid chasing price.")

    risk_dollars = capital * risk_per_trade
    distance = abs(signal.entry - signal.stop_loss) / max(signal.entry, 1e-9)
    raw_size = risk_dollars / max(distance, 0.005)
    confidence_adj = 0.35 + signal.confidence
    vol_adj = 0.60 if s.volatility > 70 else 1.0
    spike_adj = 0.70 if s.volume_ratio > 2.25 else 1.0
    position = min(capital * 0.35, raw_size * confidence_adj * vol_adj * spike_adj)
    if not allowed:
        position = 0

    if not notes:
        notes.append("Risk filter passed; size is still capped by stop distance and available capital.")
    risk_level = "LOW" if allowed and s.volatility < 45 else "MEDIUM" if allowed else "BLOCKED"
    return RiskDecision(allowed=allowed, risk_level=risk_level, position_usd=round(position, 2), max_loss_usd=round(risk_dollars, 2), notes=notes)
