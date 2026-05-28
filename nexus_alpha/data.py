from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from .connectors import LiveBundle, MarketBackupAPI, SoDEXAPI, SoSoValueAPI, SourceStatus, _first, _num

ASSETS = ["BTC", "ETH", "SOL", "XRP"]


@dataclass(frozen=True)
class AssetSnapshot:
    asset: str
    price: float
    change_24h: float
    volume_ratio: float
    sentiment: float
    volatility: float
    etf_flow_7d: float
    etf_flow_prev_7d: float
    updated_at: str


def _seed_for(asset: str) -> int:
    return sum(ord(c) for c in asset) * 97


def demo_flow_series(asset: str, days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(_seed_for(asset))
    end = datetime.now(timezone.utc).date()
    dates = [end - timedelta(days=i) for i in range(days - 1, -1, -1)]
    base = {"BTC": 75, "ETH": 38, "SOL": 12, "XRP": 8}.get(asset.upper(), 10)
    trend = {"BTC": 1.8, "ETH": .5, "SOL": -0.2, "XRP": .1}.get(asset.upper(), .2)
    noise = rng.normal(0, base * 0.55, days)
    flows = base + trend * np.arange(days) + noise
    if asset.upper() == "SOL":
        flows[-2] -= base * 2.2
    if asset.upper() == "BTC":
        flows[-1] += base * 2.5
    start_price = {"BTC": 96500, "ETH": 3650, "SOL": 170, "XRP": 0.62}.get(asset.upper(), 100)
    ret = rng.normal(0.001, 0.025, days) + np.clip(flows / (base * 10000), -0.004, 0.006)
    price = start_price * np.cumprod(1 + ret)
    volume_ratio = 1 + rng.normal(0, 0.25, days)
    sentiment = np.clip(50 + flows / max(base, 1) * 8 + rng.normal(0, 12, days), 0, 100)
    return pd.DataFrame({
        "date": pd.to_datetime(dates, utc=True),
        "asset": asset.upper(),
        "net_flow_musd": flows.round(2),
        "price": price.round(4),
        "volume_ratio": np.clip(volume_ratio, 0.35, 2.8).round(2),
        "sentiment": sentiment.round(1),
        "source": "demo",
    })


def _complete_frame(asset: str, price_frame: pd.DataFrame | None, flow_frame: pd.DataFrame | None, days: int = 30) -> pd.DataFrame:
    demo = demo_flow_series(asset, days)
    if price_frame is None or price_frame.empty:
        out = demo.copy()
    else:
        pf = price_frame.copy().sort_values("date").tail(days)
        out = pd.DataFrame({"date": pd.to_datetime(pf["date"], utc=True), "asset": asset.upper()})
        out["price"] = pd.to_numeric(pf.get("price"), errors="coerce")
        vol = pd.to_numeric(pf.get("volume", pd.Series([0] * len(pf))), errors="coerce").fillna(0)
        rolling = vol.rolling(7, min_periods=1).mean().replace(0, np.nan)
        out["volume_ratio"] = (vol / rolling).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.35, 3.5)
        out = out.dropna(subset=["price"]).tail(days)
        if len(out) < 3:
            out = demo.copy()
    if flow_frame is not None and not flow_frame.empty:
        ff = flow_frame.copy().sort_values("date").tail(days)
        ff["date_key"] = pd.to_datetime(ff["date"], utc=True).dt.date
        out["date_key"] = pd.to_datetime(out["date"], utc=True).dt.date
        out = out.merge(ff[["date_key", "net_flow_musd"]], on="date_key", how="left")
        out["net_flow_musd"] = out["net_flow_musd"].fillna(0)
        out = out.drop(columns=["date_key"])
    elif "net_flow_musd" not in out.columns:
        out["net_flow_musd"] = demo.tail(len(out))["net_flow_musd"].to_numpy() if len(out) else []
    if "sentiment" not in out.columns:
        returns = pd.to_numeric(out["price"], errors="coerce").pct_change().fillna(0)
        flow_scaled = pd.to_numeric(out["net_flow_musd"], errors="coerce").fillna(0)
        denom = max(float(flow_scaled.abs().quantile(0.75) or 1), 1)
        out["sentiment"] = (50 + returns.rolling(3, min_periods=1).mean() * 800 + (flow_scaled / denom) * 8).clip(0, 100)
    out["source"] = out.get("source", "live")
    return out[["date", "asset", "net_flow_musd", "price", "volume_ratio", "sentiment", "source"]].sort_values("date").tail(days).reset_index(drop=True)


def load_asset_data(asset: str, days: int = 30, prefer_live: bool = True) -> LiveBundle:
    statuses: list[SourceStatus] = []
    asset = asset.upper()
    if not prefer_live:
        return LiveBundle(demo_flow_series(asset, days), [SourceStatus("Demo", True, "demo mode", "", days)])

    soso = SoSoValueAPI()
    sodex = SoDEXAPI()
    backup = MarketBackupAPI()

    flow_frame, flow_status = soso.etf_flow_history(asset, days)
    statuses.append(flow_status)

    price_frame, kline_status = sodex.klines(asset, "1d", days)
    statuses.append(kline_status)

    if price_frame is None or price_frame.empty or price_frame["price"].isna().all():
        soso_price_frame, soso_kline_status = soso.currency_klines(asset, days)
        statuses.append(soso_kline_status)
        if soso_price_frame is not None and not soso_price_frame.empty:
            price_frame = soso_price_frame

    snapshot, snap_status = soso.market_snapshot(asset)
    statuses.append(snap_status)
    if snapshot and (price_frame is None or price_frame.empty):
        price = _num(_first(snapshot, ["price", "currentPrice", "last", "close", "usdPrice"]))
        if price:
            now = pd.Timestamp.utcnow()
            price_frame = pd.DataFrame({"date": [now], "asset": [asset], "price": [price], "volume": [_num(_first(snapshot, ["volume", "volume24h", "totalVolume"]))]})

    if price_frame is None or price_frame.empty or price_frame["price"].isna().all():
        price_frame, backup_status = backup.chart(asset, days)
        statuses.append(backup_status)

    frame = _complete_frame(asset, price_frame, flow_frame, days)
    if frame.empty:
        frame = demo_flow_series(asset, days)
        statuses.append(SourceStatus("Demo", True, "all live sources unavailable", "", len(frame)))
    return LiveBundle(frame, statuses)


def live_market_data(assets: Iterable[str] = ASSETS, days: int = 30, prefer_live: bool = True) -> tuple[dict[str, pd.DataFrame], list[SourceStatus]]:
    data: dict[str, pd.DataFrame] = {}
    statuses: list[SourceStatus] = []
    if prefer_live:
        statuses.append(SoSoValueAPI().config_status())
        statuses.append(SoDEXAPI().auth_status())
    for asset in assets:
        bundle = load_asset_data(asset, days, prefer_live)
        data[asset.upper()] = bundle.frame
        statuses.extend(bundle.statuses)
    return data, statuses


def demo_market_data(assets: Iterable[str] = ASSETS) -> dict[str, pd.DataFrame]:
    return {a.upper(): demo_flow_series(a.upper()) for a in assets}


def snapshot_from_series(df: pd.DataFrame) -> AssetSnapshot:
    df = df.sort_values("date").reset_index(drop=True)
    asset = str(df.loc[df.index[-1], "asset"])
    last_price = float(df.loc[df.index[-1], "price"])
    prev_price = float(df.loc[df.index[-2], "price"]) if len(df) > 1 else last_price
    change_24h = (last_price / prev_price - 1) * 100 if prev_price else 0
    returns = df["price"].pct_change().dropna()
    volatility = float(returns.tail(14).std() * math.sqrt(365) * 100) if len(returns) else 0
    etf_7d = float(df["net_flow_musd"].tail(7).sum()) if "net_flow_musd" in df else 0
    prev_7d = float(df["net_flow_musd"].tail(14).head(7).sum()) if "net_flow_musd" in df else 0
    return AssetSnapshot(
        asset=asset,
        price=last_price,
        change_24h=change_24h,
        volume_ratio=float(df.loc[df.index[-1], "volume_ratio"]),
        sentiment=float(df.loc[df.index[-1], "sentiment"]),
        volatility=volatility,
        etf_flow_7d=etf_7d,
        etf_flow_prev_7d=prev_7d,
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


class SoSoValueClient(SoSoValueAPI):
    """Backward-compatible alias for old imports."""
