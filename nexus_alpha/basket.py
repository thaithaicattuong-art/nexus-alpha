from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

THEMES: dict[str, list[tuple[str, int]]] = {
    "ai-infra": [("ai", 4), ("agent", 2), ("gpu", 3), ("compute", 2), ("tao", 4), ("render", 2)],
    "depin": [("depin", 5), ("infrastructure", 3), ("storage", 3), ("wireless", 3), ("akash", 3)],
    "defi-bluechip": [("defi", 4), ("dex", 3), ("lending", 3), ("staking", 2), ("bluechip", 4), ("yield", 2)],
    "memes": [("meme", 5), ("degen", 3), ("pepe", 3), ("doge", 3), ("bonk", 3), ("yolo", 3)],
    "rwa": [("rwa", 5), ("real world", 4), ("treasury", 3), ("bond", 3), ("ondo", 4)],
    "l2-scaling": [("l2", 4), ("layer 2", 4), ("rollup", 4), ("arbitrum", 3), ("optimism", 3), ("polygon", 3)],
}

TOKEN_UNIVERSE = pd.DataFrame([
    {"symbol":"BTC", "name":"Bitcoin", "themes":"defi-bluechip,rwa", "price":96500, "momentum30d":0.08, "sentiment":0.58, "volatility":0.46, "liquidity":0.95, "market_cap":1_900_000},
    {"symbol":"ETH", "name":"Ethereum", "themes":"defi-bluechip,l2-scaling", "price":3650, "momentum30d":0.04, "sentiment":0.52, "volatility":0.55, "liquidity":0.90, "market_cap":440_000},
    {"symbol":"SOL", "name":"Solana", "themes":"depin,memes,defi-bluechip", "price":170, "momentum30d":-0.02, "sentiment":0.36, "volatility":0.82, "liquidity":0.75, "market_cap":82_000},
    {"symbol":"XRP", "name":"XRP", "themes":"rwa", "price":0.62, "momentum30d":0.01, "sentiment":0.48, "volatility":0.72, "liquidity":0.70, "market_cap":35_000},
    {"symbol":"LINK", "name":"Chainlink", "themes":"rwa,defi-bluechip", "price":18.4, "momentum30d":0.05, "sentiment":0.57, "volatility":0.68, "liquidity":0.68, "market_cap":12_000},
    {"symbol":"ARB", "name":"Arbitrum", "themes":"l2-scaling,defi-bluechip", "price":1.12, "momentum30d":0.03, "sentiment":0.50, "volatility":0.78, "liquidity":0.62, "market_cap":4_200},
    {"symbol":"OP", "name":"Optimism", "themes":"l2-scaling", "price":2.30, "momentum30d":0.02, "sentiment":0.49, "volatility":0.80, "liquidity":0.57, "market_cap":3_600},
    {"symbol":"RNDR", "name":"Render", "themes":"ai-infra,depin", "price":9.2, "momentum30d":0.11, "sentiment":0.66, "volatility":0.94, "liquidity":0.55, "market_cap":4_600},
    {"symbol":"TAO", "name":"Bittensor", "themes":"ai-infra", "price":420, "momentum30d":0.13, "sentiment":0.70, "volatility":1.05, "liquidity":0.45, "market_cap":3_200},
    {"symbol":"ONDO", "name":"Ondo", "themes":"rwa", "price":1.25, "momentum30d":0.09, "sentiment":0.61, "volatility":0.86, "liquidity":0.50, "market_cap":2_000},
    {"symbol":"PEPE", "name":"Pepe", "themes":"memes", "price":0.000012, "momentum30d":0.16, "sentiment":0.64, "volatility":1.45, "liquidity":0.40, "market_cap":5_000},
    {"symbol":"USDC", "name":"USD Coin", "themes":"stable", "price":1.00, "momentum30d":0.0, "sentiment":0.50, "volatility":0.02, "liquidity":1.0, "market_cap":32_000},
])

@dataclass(frozen=True)
class BasketConstituent:
    symbol: str
    name: str
    weight: float
    score: float
    rationale: str

@dataclass(frozen=True)
class Basket:
    id: str
    thesis: str
    risk_profile: str
    amount_usd: float
    risk_score: int
    expected_annual_vol: float
    constituents: list[BasketConstituent]
    reasoning: str
    created_at: str


def _softmax(values: Iterable[float], temp: float = 0.55) -> list[float]:
    vals = np.array(list(values), dtype=float)
    if len(vals) == 0:
        return []
    vals = vals - vals.max()
    exp = np.exp(vals / max(temp, 1e-6))
    return (exp / exp.sum()).tolist()


def _cap_weights(weights: list[float], cap: float) -> list[float]:
    w = np.array(weights, dtype=float)
    for _ in range(8):
        overflow = np.maximum(w - cap, 0).sum()
        w = np.minimum(w, cap)
        under = w < cap - 1e-9
        if overflow < 1e-9 or not under.any():
            break
        w[under] += overflow * w[under] / max(w[under].sum(), 1e-9)
    return (w / w.sum()).tolist()


def classify_thesis(thesis: str, risk_profile: str = "balanced") -> tuple[list[tuple[str, float]], list[str]]:
    text = thesis.lower()
    hits: list[tuple[str, int]] = []
    notes: list[str] = []
    for theme, kws in THEMES.items():
        score = 0
        matched = []
        for kw, weight in kws:
            if kw in text:
                score += weight
                matched.append(kw)
        if score:
            hits.append((theme, score))
            notes.append(f"{theme}: matched {', '.join(matched[:4])}")
    if not hits:
        hits = [("defi-bluechip", 1)]
        notes.append("Không thấy keyword rõ, mặc định dùng lõi blue-chip/defi.")
    weights = _softmax([h[1] for h in hits], temp=2.0)
    return [(theme, float(w)) for (theme, _), w in zip(hits, weights)], notes


def build_basket(thesis: str, amount_usd: float = 10_000, risk_profile: str = "balanced") -> Basket:
    risk_profile = risk_profile.lower()
    theme_weights, notes = classify_thesis(thesis, risk_profile)
    rows = []
    for _, t in TOKEN_UNIVERSE.iterrows():
        if t["symbol"] == "USDC":
            continue
        token_themes = set(str(t["themes"]).split(","))
        theme_fit = sum(w for theme, w in theme_weights if theme in token_themes)
        if theme_fit <= 0:
            continue
        if risk_profile == "aggressive":
            factor = t["momentum30d"] * 1.25 + (t["sentiment"] - 0.5) * 0.9 - t["volatility"] * 0.05
            n = 7
            cap = 0.43
        elif risk_profile == "conservative":
            factor = t["liquidity"] * 0.65 + t["momentum30d"] * 0.35 + (t["sentiment"] - 0.5) * 0.3 - t["volatility"] * 0.55
            n = 4
            cap = 0.30
        else:
            factor = t["momentum30d"] * 0.75 + (t["sentiment"] - 0.5) * 0.55 + t["liquidity"] * 0.28 - t["volatility"] * 0.20
            n = 6
            cap = 0.35
        score = theme_fit * 0.65 + factor * 0.35 + 0.35
        rows.append((t, float(max(score, 0.001))))
    rows = sorted(rows, key=lambda x: x[1], reverse=True)[:n]
    raw = _softmax([s for _, s in rows], temp=0.32 if risk_profile == "aggressive" else 0.48)
    weights = _cap_weights(raw, cap)
    cons = []
    for (t, score), w in zip(rows, weights):
        rationales = [f"theme fit với thesis", f"momentum 30d {t['momentum30d']:+.1%}", f"sentiment {t['sentiment']:.2f}", f"liquidity {t['liquidity']:.2f}"]
        cons.append(BasketConstituent(str(t["symbol"]), str(t["name"]), round(float(w), 4), round(score, 3), " • ".join(rationales)))
    exp_vol = sum(c.weight * float(TOKEN_UNIVERSE.loc[TOKEN_UNIVERSE.symbol == c.symbol, "volatility"].iloc[0]) for c in cons)
    concentration = sum(c.weight ** 2 for c in cons)
    risk_score = int(min(100, round(exp_vol * 62 + concentration * 55)))
    bid = hashlib.sha1(f"{thesis}-{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:10]
    reasoning = "\n".join([
        "Thesis được tách thành: " + ", ".join(f"{theme} {w:.0%}" for theme, w in theme_weights),
        "Chọn token theo theme fit + momentum + sentiment + liquidity, sau đó giới hạn concentration cap.",
        "Notes: " + " | ".join(notes),
    ])
    return Basket(bid, thesis, risk_profile, float(amount_usd), risk_score, round(exp_vol, 3), cons, reasoning, datetime.now(timezone.utc).isoformat(timespec="seconds"))


def basket_to_frame(basket: Basket) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": c.symbol,
        "name": c.name,
        "weight_pct": round(c.weight * 100, 2),
        "notional_usd": round(c.weight * basket.amount_usd, 2),
        "score": c.score,
        "rationale": c.rationale,
    } for c in basket.constituents])


def as_dict(basket: Basket) -> dict:
    d = asdict(basket)
    return d
