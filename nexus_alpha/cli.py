from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from .basket import basket_to_frame, build_basket
from .briefing import generate_brief
from .connectors import SoDEXAPI, SoSoValueAPI, SourceStatus
from .data import ASSETS, live_market_data
from .lab import build_execution_preview, run_backtest, run_monte_carlo, run_stress_tests
from .risk import evaluate_risk
from .signals import generate_signals


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Nexus Alpha CLI")
    parser.add_argument("--asset", default="ALL", help="BTC, ETH, SOL, XRP, or ALL")
    parser.add_argument("--capital", type=float, default=float(os.getenv("DEFAULT_CAPITAL", 10000)))
    parser.add_argument("--risk", type=float, default=float(os.getenv("RISK_PER_TRADE", 0.01)))
    parser.add_argument("--thesis", default="AI infrastructure and bluechip crypto with ETF flow confirmation")
    parser.add_argument("--profile", default="balanced", choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--mode", default="signals", choices=["signals", "basket", "backtest", "stress", "execute-preview", "source-check"])
    parser.add_argument("--demo", action="store_true", help="Force deterministic demo data instead of live API connectors")
    parser.add_argument("--network-check", action="store_true", help="For source-check, also perform live HTTP calls to configured APIs")
    args = parser.parse_args()

    if args.mode == "signals":
        assets = ASSETS if args.asset.upper() == "ALL" else [args.asset.upper()]
        data, statuses = live_market_data(assets, prefer_live=not args.demo)
        signals = generate_signals(data)
        risks = {s.asset: evaluate_risk(s, data[s.asset], args.capital, args.risk) for s in signals}
        for s in signals:
            r = risks[s.asset]
            print("=" * 72)
            print(f"{s.asset} | {s.action} | confidence {s.confidence:.0%} | score {s.score:+.3f}")
            print(f"Entry: {s.entry} | SL: {s.stop_loss} | TP: {s.take_profit}")
            print(f"Risk: {r.risk_level} | Allowed: {r.allowed} | Size: ${r.position_usd:,.2f}")
            print("Thesis:", s.thesis)
            print("Risk notes:", "; ".join(r.notes))
        if len(signals) > 1:
            print("\n" + generate_brief(signals, risks, args.capital))
        print("\nSource status:")
        for st in statuses:
            print(f"- {st.name}: {'OK' if st.ok else 'MISS'} | rows={st.rows} | {st.message}")
        return

    if args.mode == "source-check":
        if args.demo:
            assets = ASSETS if args.asset.upper() == "ALL" else [args.asset.upper()]
            _, statuses = live_market_data(assets, prefer_live=False)
        else:
            statuses: list[SourceStatus] = [
                SoSoValueAPI().config_status(),
                SoDEXAPI().auth_status(),
            ]
            if args.network_check:
                assets = ASSETS if args.asset.upper() == "ALL" else [args.asset.upper()]
                _, live_statuses = live_market_data(assets, prefer_live=True)
                statuses.extend(live_statuses)
            else:
                statuses.append(SourceStatus("Network Check", True, "skipped; add --network-check to call live APIs", "", 0))
        for st in statuses:
            print(f"{st.name:16} | {'OK' if st.ok else 'MISS'} | rows={st.rows:<4} | {st.message} | {st.endpoint}")
        return

    basket = build_basket(args.thesis, args.capital, args.profile)
    print(f"Basket {basket.id} | risk {basket.risk_score}/100 | vol {basket.expected_annual_vol:.2f}")
    print(basket.reasoning)
    print(basket_to_frame(basket).to_string(index=False))

    if args.mode == "backtest":
        _, metrics = run_backtest(basket)
        _, mc = run_monte_carlo(basket)
        print("Backtest metrics:", metrics)
        print("Monte Carlo stats:", mc)
    elif args.mode == "stress":
        print(run_stress_tests(basket).to_string(index=False))
    elif args.mode == "execute-preview":
        print(build_execution_preview(basket).to_string(index=False))


if __name__ == "__main__":
    main()
