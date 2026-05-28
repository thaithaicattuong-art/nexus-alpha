from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .basket import Basket, TOKEN_UNIVERSE, as_dict

DATA_DIR = Path(os.getenv("NEXUS_DATA_DIR", "data"))
LEDGER_FILE = DATA_DIR / "paper_ledger.json"


def _rng_for(*parts: str) -> np.random.Generator:
    seed = abs(hash("|".join(parts))) % (2**32)
    return np.random.default_rng(seed)


def generate_return_matrix(symbols: list[str], days: int = 180) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date()
    dates = [end - timedelta(days=i) for i in range(days - 1, -1, -1)]
    data: dict[str, np.ndarray] = {}
    common = _rng_for("common", str(days)).normal(0.0004, 0.018, days)
    for sym in symbols:
        row = TOKEN_UNIVERSE[TOKEN_UNIVERSE.symbol == sym]
        vol = float(row["volatility"].iloc[0]) if not row.empty else 0.7
        mom = float(row["momentum30d"].iloc[0]) if not row.empty else 0.02
        rng = _rng_for(sym, str(days))
        idio = rng.normal(mom / 30, max(vol / math.sqrt(365), 0.001), days)
        data[sym] = 0.45 * common + 0.55 * idio
    return pd.DataFrame(data, index=pd.to_datetime(dates))


def run_backtest(basket: Basket, days: int = 90) -> tuple[pd.DataFrame, dict]:
    symbols = [c.symbol for c in basket.constituents]
    weights = np.array([c.weight for c in basket.constituents], dtype=float)
    returns = generate_return_matrix(symbols, max(days, 30)).tail(days)
    port_ret = returns.values @ weights
    equity = basket.amount_usd * np.cumprod(1 + port_ret)
    benchmark = basket.amount_usd * np.cumprod(1 + returns[symbols[0]].values)
    frame = pd.DataFrame({"date": returns.index, "Nexus Basket": equity, f"{symbols[0]} benchmark": benchmark})
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    downside = port_ret[port_ret < 0]
    metrics = {
        "total_return_pct": float(round((equity[-1] / basket.amount_usd - 1) * 100, 2)),
        "benchmark_return_pct": float(round((benchmark[-1] / basket.amount_usd - 1) * 100, 2)),
        "max_drawdown_pct": float(round(drawdown.min() * 100, 2)),
        "annual_vol_pct": float(round(port_ret.std() * math.sqrt(365) * 100, 2)),
        "sharpe_demo": float(round((port_ret.mean() / max(port_ret.std(), 1e-9)) * math.sqrt(365), 2)),
        "sortino_demo": float(round((port_ret.mean() / max(downside.std(), 1e-9)) * math.sqrt(365), 2)) if len(downside) else 0.0,
    }
    return frame, metrics


def run_monte_carlo(basket: Basket, horizon_days: int = 30, paths: int = 1000) -> tuple[pd.DataFrame, dict]:
    symbols = [c.symbol for c in basket.constituents]
    weights = np.array([c.weight for c in basket.constituents], dtype=float)
    hist = generate_return_matrix(symbols, 240)
    port = hist.values @ weights
    rng = _rng_for("mc", basket.id, str(horizon_days), str(paths))
    terminals = []
    fan_rows = []
    for p in range(paths):
        sample = rng.choice(port, size=horizon_days, replace=True)
        eq = basket.amount_usd * np.cumprod(1 + sample)
        terminals.append(eq[-1])
        if p < 80:
            for d, v in enumerate(eq, 1):
                fan_rows.append({"path": p, "day": d, "value": v})
    arr = np.array(terminals)
    var95 = np.percentile(arr / basket.amount_usd - 1, 5) * 100
    cvar95 = (arr[arr <= np.percentile(arr, 5)] / basket.amount_usd - 1).mean() * 100
    stats = {
        "expected_terminal_usd": float(round(arr.mean(), 2)),
        "median_terminal_usd": float(round(np.median(arr), 2)),
        "prob_loss_pct": float(round((arr < basket.amount_usd).mean() * 100, 1)),
        "var_95_pct": float(round(var95, 2)),
        "cvar_95_pct": float(round(cvar95, 2)),
        "paths": paths,
        "horizon_days": horizon_days,
    }
    return pd.DataFrame(fan_rows), stats


SCENARIOS = [
    {"id":"covid", "name":"COVID crash 2020", "blurb":"Liquidity shock: correlation tăng mạnh, risk assets giảm đồng loạt.", "shock":-0.32, "vol_mult":1.6, "days":45},
    {"id":"ftx", "name":"FTX collapse 2022", "blurb":"Counterparty shock: low-cap, meme và high beta chịu tác động lớn.", "shock":-0.24, "vol_mult":1.35, "days":68},
    {"id":"eth_etf", "name":"ETH ETF launch 2024", "blurb":"Risk-on rotation: ETH ecosystem/L2/AI outperform nhưng vẫn biến động.", "shock":0.16, "vol_mult":1.15, "days":70},
    {"id":"flow_reversal", "name":"ETF flow reversal", "blurb":"Dòng tiền ETF đảo chiều 7 ngày, agent giảm beta và tăng cash.", "shock":-0.12, "vol_mult":1.1, "days":21},
]


def run_stress_tests(basket: Basket) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        drag = 0.0
        best = (None, -999.0)
        worst = (None, 999.0)
        for c in basket.constituents:
            meta = TOKEN_UNIVERSE[TOKEN_UNIVERSE.symbol == c.symbol].iloc[0]
            high_beta = float(meta["volatility"])
            theme = str(meta["themes"])
            theme_adj = 1.25 if "memes" in theme or "ai-infra" in theme else 0.75 if "rwa" in theme else 1.0
            token_ret = float(sc["shock"]) * theme_adj * (0.55 + high_beta * 0.45)
            drag += c.weight * token_ret
            if token_ret > best[1]: best = (c.symbol, token_ret)
            if token_ret < worst[1]: worst = (c.symbol, token_ret)
        max_dd = min(drag * float(sc["vol_mult"]), drag - 0.05 if drag < 0 else -0.04)
        rows.append({
            "scenario": sc["name"],
            "days": sc["days"],
            "basket_return_pct": round(drag * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "worst_constituent": worst[0],
            "best_constituent": best[0],
            "blurb": sc["blurb"],
        })
    return pd.DataFrame(rows)


def build_execution_preview(basket: Basket, max_slippage_bps: float = 35, confirm: bool = False) -> pd.DataFrame:
    rows = []
    for c in basket.constituents:
        meta = TOKEN_UNIVERSE[TOKEN_UNIVERSE.symbol == c.symbol].iloc[0]
        notional = c.weight * basket.amount_usd
        liquidity = float(meta["liquidity"])
        price = float(meta["price"])
        slippage = max(2, min(150, (notional / 10_000) * (1.05 - liquidity) * 40 + 3))
        status = "READY" if confirm and slippage <= max_slippage_bps else "REVIEW_ONLY" if slippage <= max_slippage_bps else "BLOCKED_SLIPPAGE"
        rows.append({
            "market": f"{c.symbol}/USDC",
            "side": "BUY",
            "notional_usd": round(notional, 2),
            "est_price": price,
            "est_slippage_bps": round(slippage, 2),
            "order_type": "IOC limit preview",
            "status": status,
        })
    return pd.DataFrame(rows)


def ensure_ledger() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LEDGER_FILE.exists():
        LEDGER_FILE.write_text("[]", encoding="utf-8")


def read_ledger() -> pd.DataFrame:
    ensure_ledger()
    try:
        data = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = []
    return pd.DataFrame(data)


def paper_log_basket(basket: Basket, note: str = "paper basket created") -> dict:
    ensure_ledger()
    data = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
    entry = {
        "id": f"PAPER-{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "basket_id": basket.id,
        "thesis": basket.thesis,
        "risk_profile": basket.risk_profile,
        "amount_usd": basket.amount_usd,
        "risk_score": basket.risk_score,
        "constituents": [asdict(c) for c in basket.constituents],
        "note": note,
        "status": "PAPER_ONLY",
    }
    data.insert(0, entry)
    LEDGER_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return entry


def agent_steps_for_basket(basket: Basket) -> list[dict]:
    return [
        {"stage":"Observe", "message":"Đọc thesis, demo market metrics, sentiment, liquidity và volatility.", "status":"done"},
        {"stage":"Reason", "message":basket.reasoning.replace("\n", " "), "status":"done"},
        {"stage":"Propose", "message":f"Đề xuất {len(basket.constituents)} assets, risk score {basket.risk_score}/100, expected vol {basket.expected_annual_vol:.2f}.", "status":"done"},
        {"stage":"Confirm", "message":"Không đặt lệnh thật. Mọi execution đều là preview/paper, cần xác nhận thủ công nếu nối broker/SoDEX sau.", "status":"safe"},
    ]


def export_basket_json(basket: Basket, folder: str | Path = DATA_DIR) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"basket_{basket.id}.json"
    out.write_text(json.dumps(as_dict(basket), indent=2), encoding="utf-8")
    return out
