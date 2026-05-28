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
        reasons.append(f"ETF flow 7 ngày cải thiện {flow_delta:,.1f}M USD so với 7 ngày trước.")
    else:
        reasons.append(f"ETF flow 7 ngày yếu hơn {abs(flow_delta):,.1f}M USD so với 7 ngày trước.")
    reasons.append(f"Giá 24h biến động {s.change_24h:+.2f}% và sentiment hiện ở {s.sentiment:.1f}/100.")
    reasons.append(f"Volume ratio {s.volume_ratio:.2f}x cho biết mức độ chú ý hiện tại của thị trường.")
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

    # SL/TP expands with realized annualized volatility but stays usable.
    vol_band = min(max(s.volatility / 100 / 8, 0.018), 0.075)
    if action == "SELL":
        stop = s.price * (1 + vol_band)
        take = s.price * (1 - vol_band * 1.8)
    else:
        stop = s.price * (1 - vol_band)
        take = s.price * (1 + vol_band * 1.8)

    thesis = (
        f"{s.asset}: {action} bias với confidence {confidence:.0%}. "
        f"Động lực chính đến từ flow 7 ngày, momentum giá và sentiment; "
        f"chỉ nên vào lệnh nếu risk filter không chặn."
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
