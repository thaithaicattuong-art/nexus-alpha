from __future__ import annotations

import pandas as pd

from .risk import RiskDecision
from .signals import TradeSignal


def suggested_allocations(signals: list[TradeSignal], risks: dict[str, RiskDecision], capital: float = 10_000) -> pd.DataFrame:
    rows = []
    for sig in signals:
        risk = risks[sig.asset]
        if risk.allowed and sig.action in {"BUY", "SELL"}:
            conviction = sig.confidence * abs(sig.score)
            rows.append({
                "asset": sig.asset,
                "action": sig.action,
                "conviction": conviction,
                "position_usd": risk.position_usd,
                "risk_level": risk.risk_level,
            })
        else:
            rows.append({
                "asset": sig.asset,
                "action": "FLAT",
                "conviction": 0,
                "position_usd": 0,
                "risk_level": risk.risk_level,
            })
    df = pd.DataFrame(rows)
    total = df["position_usd"].sum()
    df["allocation_pct"] = 0 if total == 0 else (df["position_usd"] / capital * 100).round(2)
    return df.sort_values(["position_usd", "conviction"], ascending=False)


def decision_log(signals: list[TradeSignal], risks: dict[str, RiskDecision]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "asset": s.asset,
            "signal": s.action,
            "score": s.score,
            "confidence": f"{s.confidence:.0%}",
            "risk": risks[s.asset].risk_level,
            "allowed": risks[s.asset].allowed,
            "entry": s.entry,
            "sl": s.stop_loss,
            "tp": s.take_profit,
            "note": "; ".join(risks[s.asset].notes),
        }
        for s in signals
    ])
