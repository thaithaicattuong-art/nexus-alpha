from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

REQUIRED_COLS = ["opened_at", "closed_at", "symbol", "side", "entry_price", "exit_price", "size", "pnl"]


def demo_trades(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)
    symbols = rng.choice(["BTC", "ETH", "SOL", "XRP"], n, p=[.42, .28, .18, .12])
    sides = rng.choice(["long", "short"], n)
    opened = [now - timedelta(days=int(x), hours=int(rng.integers(0, 24))) for x in rng.integers(1, 95, n)]
    hold_h = rng.integers(1, 60, n)
    closed = [o + timedelta(hours=int(h)) for o, h in zip(opened, hold_h)]
    size = rng.uniform(200, 4000, n).round(2)
    base_edge = np.where(symbols == "BTC", 0.004, np.where(symbols == "SOL", -0.003, 0.001))
    side_edge = np.where(sides == "long", 0.002, -0.001)
    ret = rng.normal(base_edge + side_edge, 0.035, n)
    pnl = (size * ret).round(2)
    entry = np.where(symbols == "BTC", 95000, np.where(symbols == "ETH", 3600, np.where(symbols == "SOL", 170, .62)))
    entry = entry * (1 + rng.normal(0, .04, n))
    exitp = entry * (1 + ret * np.where(sides == "long", 1, -1))
    return pd.DataFrame({
        "opened_at": opened, "closed_at": closed, "symbol": symbols, "side": sides,
        "entry_price": entry.round(4), "exit_price": exitp.round(4), "size": size, "pnl": pnl,
    })


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["opened_at"] = pd.to_datetime(out["opened_at"], utc=True, errors="coerce")
    out["closed_at"] = pd.to_datetime(out["closed_at"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["side"] = out["side"].astype(str).str.lower()
    for c in ["entry_price", "exit_price", "size", "pnl"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLS)
    out["win"] = out["pnl"] > 0
    out["r_multiple"] = out["pnl"] / (out["size"].abs() * 0.01).replace(0, np.nan)
    out["hold_hours"] = (out["closed_at"] - out["opened_at"]).dt.total_seconds() / 3600
    out["hour"] = out["opened_at"].dt.hour
    out["session"] = pd.cut(out["hour"], bins=[-1, 6, 13, 20, 24], labels=["Asia night", "Asia/EU", "US open", "US late"])
    out["regime"] = np.where(out["pnl"].rolling(12, min_periods=1).mean() > 0, "hot streak", "cold streak")
    return out


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = normalize_trades(df).groupby(by, observed=True)
    res = g.agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        winrate=("win", "mean"),
        expectancy_r=("r_multiple", "mean"),
        avg_hold_h=("hold_hours", "mean"),
    ).reset_index()
    res["winrate"] = (res["winrate"] * 100).round(1)
    for c in ["pnl", "avg_pnl", "expectancy_r", "avg_hold_h"]:
        res[c] = res[c].round(2)
    return res.sort_values("pnl", ascending=False)


def find_edge_notes(df: pd.DataFrame) -> list[str]:
    trades = normalize_trades(df)
    notes = []
    for dim in ["symbol", "side", "session", "regime"]:
        s = summarize(trades, dim)
        if not s.empty:
            best = s.iloc[0]
            worst = s.iloc[-1]
            notes.append(f"Best {dim}: {best[dim]} with PnL {best['pnl']:,.2f} and winrate {best['winrate']}%.")
            notes.append(f"Worst {dim}: {worst[dim]} with PnL {worst['pnl']:,.2f} and winrate {worst['winrate']}%.")
    return notes[:6]
