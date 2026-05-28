from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .basket import Basket, TOKEN_UNIVERSE
from .data import snapshot_from_series
from .risk import RiskDecision
from .signals import TradeSignal


@dataclass(frozen=True)
class DemoScenario:
    name: str
    setup: str
    click_path: str
    success_signal: str


def signal_factor_table(data: dict[str, pd.DataFrame], signals: list[TradeSignal], risks: dict[str, RiskDecision]) -> pd.DataFrame:
    sig_map = {s.asset: s for s in signals}
    rows: list[dict] = []
    for asset, frame in data.items():
        snap = snapshot_from_series(frame)
        sig = sig_map[asset]
        flow_delta = snap.etf_flow_7d - snap.etf_flow_prev_7d
        rows.append({
            "asset": asset,
            "action": sig.action,
            "score": sig.score,
            "confidence_pct": round(sig.confidence * 100, 1),
            "risk_status": risks[asset].risk_level,
            "price": round(snap.price, 6),
            "change_24h_pct": round(snap.change_24h, 2),
            "flow_7d_musd": round(snap.etf_flow_7d, 2),
            "flow_delta_musd": round(flow_delta, 2),
            "sentiment": round(snap.sentiment, 1),
            "volume_ratio": round(snap.volume_ratio, 2),
            "annual_vol_pct": round(snap.volatility, 1),
            "position_usd": risks[asset].position_usd,
        })
    return pd.DataFrame(rows).sort_values(["position_usd", "confidence_pct"], ascending=False)


def market_regime(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    rows = []
    for asset, frame in data.items():
        snap = snapshot_from_series(frame)
        rows.append(snap)
    if not rows:
        return {"mode": "Unknown", "score": 0, "summary": "No market data loaded."}
    avg_sent = np.mean([r.sentiment for r in rows])
    avg_flow_delta = np.mean([r.etf_flow_7d - r.etf_flow_prev_7d for r in rows])
    avg_vol = np.mean([r.volatility for r in rows])
    raw = (avg_sent - 50) / 50 + np.tanh(avg_flow_delta / 250) - max(avg_vol - 65, 0) / 80
    score = float(np.clip(raw, -1.0, 1.0))
    if score > 0.25:
        mode = "Risk-on"
    elif score < -0.25:
        mode = "Risk-off"
    else:
        mode = "Neutral / selective"
    return {
        "mode": mode,
        "score": round(score, 2),
        "avg_sentiment": round(float(avg_sent), 1),
        "avg_flow_delta_musd": round(float(avg_flow_delta), 2),
        "avg_vol_pct": round(float(avg_vol), 1),
        "summary": f"{mode}: average sentiment {avg_sent:.1f}, average flow delta {avg_flow_delta:,.1f}M USD, average vol {avg_vol:.1f}%.",
    }


def flow_rotation_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for asset, frame in data.items():
        frame = frame.sort_values("date")
        recent = float(frame["net_flow_musd"].tail(7).sum())
        prev = float(frame["net_flow_musd"].tail(14).head(7).sum())
        accel = recent - prev
        price_chg = float(frame["price"].iloc[-1] / frame["price"].iloc[max(0, len(frame)-8)] - 1) * 100 if len(frame) > 8 else 0
        rows.append({
            "asset": asset,
            "recent_flow_7d_musd": round(recent, 2),
            "previous_flow_7d_musd": round(prev, 2),
            "flow_acceleration_musd": round(accel, 2),
            "price_change_7d_pct": round(price_chg, 2),
            "rotation_label": "Flow Leader" if accel > 50 else "Flow Lag" if accel < -50 else "Stable",
        })
    return pd.DataFrame(rows).sort_values("flow_acceleration_musd", ascending=False)


def rebalance_plan(basket: Basket, factor_table: pd.DataFrame, max_turnover_pct: float = 25.0) -> pd.DataFrame:
    if factor_table.empty:
        return pd.DataFrame()
    signal_map = factor_table.set_index("asset").to_dict("index")
    rows = []
    cash_raise = 0.0
    for c in basket.constituents:
        f = signal_map.get(c.symbol, {})
        risk_status = str(f.get("risk_status", "UNKNOWN"))
        score = float(f.get("score", 0))
        target = c.weight
        reason = "Keep weight"
        if risk_status == "BLOCKED" or score < -0.15:
            cut = min(c.weight * 0.5, max_turnover_pct / 100)
            target = max(0.0, c.weight - cut)
            cash_raise += cut
            reason = "Cut exposure because signal/risk is weak"
        elif score > 0.25 and risk_status != "BLOCKED":
            add = min(0.04, max_turnover_pct / 100)
            target = c.weight + add
            reason = "Increase weight because signal/risk is supportive"
        rows.append({
            "symbol": c.symbol,
            "current_weight_pct": round(c.weight * 100, 2),
            "target_weight_pct": round(target * 100, 2),
            "trade_weight_pct": round((target - c.weight) * 100, 2),
            "reason": reason,
        })
    total_target = sum(r["target_weight_pct"] for r in rows) / 100
    if total_target > 1:
        scale = 1 / total_target
        for r in rows:
            r["target_weight_pct"] = round(r["target_weight_pct"] * scale, 2)
            r["trade_weight_pct"] = round(r["target_weight_pct"] - r["current_weight_pct"], 2)
    if cash_raise > 0.001:
        rows.append({"symbol": "USDC", "current_weight_pct": 0.0, "target_weight_pct": round(cash_raise * 100, 2), "trade_weight_pct": round(cash_raise * 100, 2), "reason": "Raised from blocked or weak-risk assets"})
    return pd.DataFrame(rows)


def liquidity_map(basket: Basket) -> pd.DataFrame:
    rows = []
    for c in basket.constituents:
        meta = TOKEN_UNIVERSE[TOKEN_UNIVERSE.symbol == c.symbol]
        if meta.empty:
            continue
        row = meta.iloc[0]
        notional = c.weight * basket.amount_usd
        liquidity = float(row["liquidity"])
        price = float(row["price"])
        est_slip = max(1.5, min(180, (notional / 8000) * (1.08 - liquidity) * 45 + 2))
        rows.append({
            "symbol": c.symbol,
            "notional_usd": round(notional, 2),
            "liquidity_score": round(liquidity, 2),
            "reference_price": price,
            "estimated_slippage_bps": round(est_slip, 2),
            "execution_quality": "A" if est_slip < 15 else "B" if est_slip < 45 else "C",
        })
    return pd.DataFrame(rows)


def api_usage_scorecard(source_statuses) -> pd.DataFrame:
    rows = []
    for st in source_statuses:
        rows.append({
            "connector": st.name,
            "status": "OK" if st.ok else "Needs attention",
            "rows": st.rows,
            "message": st.message,
            "endpoint": st.endpoint,
        })
    if not rows:
        rows.append({"connector": "No source", "status": "Needs attention", "rows": 0, "message": "No connector status available", "endpoint": ""})
    return pd.DataFrame(rows)


def build_submission_checklist(source_statuses, basket: Basket, signals: list[TradeSignal]) -> pd.DataFrame:
    source_ok = any(s.ok and "SoSoValue" in s.name for s in source_statuses) and any(s.ok and "SoDEX" in s.name for s in source_statuses)
    private_ready = any(s.ok and "Auth" in s.name for s in source_statuses)
    checks = [
        ("Solid API usage", source_ok, "SoSoValue and SoDEX source status should show successful rows in the API tab."),
        ("Strong execution demo", True, "Execution Preview produces gated IOC-style paper orders with slippage estimates."),
        ("Agentic reasoning", bool(basket.constituents), "Agent Log explains observe, reason, propose, and confirm stages."),
        ("Risk management", any(s.action != "HOLD" for s in signals), "Signal Desk includes stop-loss, take-profit, confidence, and risk gating."),
        ("Private signing readiness", private_ready, "SoDEX private signing config is checked, but live order submission remains disabled."),
        ("Demo story", True, "Demo Center provides a walkthrough script for judges."),
    ]
    return pd.DataFrame([{"criterion": c, "ready": "YES" if ok else "REVIEW", "note": n} for c, ok, n in checks])


def demo_scenarios() -> list[DemoScenario]:
    return [
        DemoScenario("ETF-flow signal demo", "Choose BTC, ETH, SOL, XRP and keep live APIs enabled.", "Signal Desk → Flow Rotation → Daily Brief", "A judge can see why a signal is BUY/HOLD/SELL and what risk filter did."),
        DemoScenario("Thesis-to-basket demo", "Type a thesis such as 'AI infrastructure and RWA with ETF confirmation'.", "Thesis Basket → Rebalance Plan → Agent Log", "The app turns a human thesis into a weighted, explainable paper basket."),
        DemoScenario("Execution readiness demo", "Use the default basket amount and set max slippage to 35 bps.", "Execution Preview → Liquidity Map", "Every paper order has notional, slippage, and a blocked/ready status."),
        DemoScenario("Risk lab demo", "Run 90-day backtest, 1,000 Monte Carlo paths, and all stress scenarios.", "Backtest & Monte Carlo → Stress Test", "The demo shows return, max drawdown, probability of loss, and CVaR."),
    ]


def demo_script_text() -> str:
    return (
        "1. Start on the Command Center and explain that Nexus Alpha converts SoSoValue intelligence and SoDEX market data into paper-trading decisions.\n"
        "2. Open API Sources to prove live connector usage and signing-readiness checks.\n"
        "3. Open Signal Desk and explain one asset: flow acceleration, sentiment, confidence, risk status, entry, stop, and target.\n"
        "4. Open Thesis Basket and show how a thesis becomes a constrained basket.\n"
        "5. Open Execution Preview to show order sizing, slippage, and safety gates.\n"
        "6. Open Risk Lab to show backtest, Monte Carlo, stress tests, and what can go wrong.\n"
        "7. Finish in Agent Log / Ledger to show auditable reasoning and paper records."
    )
