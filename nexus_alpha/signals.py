from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import AssetSnapshot, snapshot_from_series


@dataclass(frozen=True)
class TradeSignal:
    asset: str
    action: str
    confidence: float
    score: float
    entry: float
    stop_loss: float
    take_profit: float
    thesis: str
    reasons: list[str]


def score_snapshot(s: AssetSnapshot) -> tuple[float, list[str]]:
    reasons: list[str] = []
    flow_delta = s.etf_flow_7d - s.etf_flow_prev_7d
    flow_strength = flow_delta / max(abs(s.etf_flow_prev_7d), 150)
    price_score = np.tanh(s.change_24h / 4)
    sentiment_score = (s.sentiment - 50) / 50
    volume_score = min(max((s.volume_ratio - 1) / 1.5, -1), 1)

    score = 0.42 * np.tanh(flow_strength) + 0.25 * price_score + 0.23 * sentiment_score + 0.10 * volume_score

    if flow_delta > 0:
        reasons.append(f"7-day ETF flow improved by {flow_delta:,.1f}M USD versus the previous 7-day window.")
    else:
        reasons.append(f"7-day ETF flow weakened by {abs(flow_delta):,.1f}M USD versus the previous 7-day window.")
    reasons.append(f"24h price change is {s.change_24h:+.2f}% and current sentiment is {s.sentiment:.1f}/100.")
    reasons.append(f"Volume ratio is {s.volume_ratio:.2f}x, indicating current market attention.")
    return float(score), reasons


def generate_signal(df: pd.DataFrame) -> TradeSignal:
    s = snapshot_from_series(df)
    score, reasons = score_snapshot(s)
    confidence = round(min(0.95, max(0.05, abs(score) * 1.35 + 0.20)), 2)

    if score >= 0.22:
        action = "BUY"
    elif score <= -0.22:
        action = "SELL"
    else:
        action = "HOLD"

    vol_band = min(max(s.volatility / 100 / 8, 0.018), 0.075)
    if action == "SELL":
        stop = s.price * (1 + vol_band)
        take = s.price * (1 - vol_band * 1.8)
    else:
        stop = s.price * (1 - vol_band)
        take = s.price * (1 + vol_band * 1.8)

    thesis = (
        f"{s.asset}: {action} bias with {confidence:.0%} confidence. "
        "The main drivers are 7-day flow momentum, price momentum, and sentiment. "
        "Enter only when the risk filter allows the setup."
    )
    return TradeSignal(
        asset=s.asset,
        action=action,
        confidence=confidence,
        score=round(score, 3),
        entry=round(s.price, 4),
        stop_loss=round(stop, 4),
        take_profit=round(take, 4),
        thesis=thesis,
        reasons=reasons,
    )


def generate_signals(data: dict[str, pd.DataFrame]) -> list[TradeSignal]:
    return [generate_signal(df) for df in data.values()]
