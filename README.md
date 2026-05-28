# Nexus Alpha

Nexus Alpha is a crypto/ETF research and paper-trading intelligence dashboard built around SoSoValue and SoDEX connectors.

The app is designed for demo, research, and paper trading. Live order submission is disabled by default.

## What is inside

- **Command Center**: market regime, signal matrix, flow rotation, and readiness checklist.
- **Signal Desk**: BUY/HOLD/SELL signals with confidence, entry, stop-loss, take-profit, risk status, and paper allocation.
- **API Sources**: SoSoValue and SoDEX connector status, row counts, endpoints, and signing-readiness checks.
- **Thesis Basket**: converts an investment thesis into a weighted, explainable basket.
- **Rebalance Plan**: suggests target weights based on signal/risk changes.
- **Backtest & Monte Carlo**: simulated equity curve, max drawdown, Sharpe demo, probability of loss, and CVaR.
- **Stress Test**: scenario testing for liquidity shocks, counterparty shocks, ETF-flow reversals, and risk-on rotations.
- **Execution Preview**: IOC-style paper order plan with notional, price, slippage, and blocked/ready status.
- **Liquidity Map**: execution-quality scoring for basket constituents.
- **Edge Analyzer**: upload trade history CSV to analyze PnL, winrate, expectancy, and best/worst slices.
- **Agent Log / Ledger**: auditable reasoning loop and paper ledger records.
- **Demo Center**: one-minute judge script and walkthrough scenarios.
- **Daily Brief**: AI-style market and basket summary.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment variables

Create `.env` locally, or add the same values to Streamlit Cloud Secrets using TOML format.

```toml
SOSOVALUE_API_KEY = "your_sosovalue_key"
SOSOVALUE_BASE_URL = "https://openapi.sosovalue.com/openapi/v1"

SODEX_NETWORK = "mainnet"
SODEX_MARKET = "spot"
SODEX_API_KEY_NAME = "your_sodex_api_key_name"
SODEX_API_PRIVATE_KEY = "0xYOUR_32_BYTE_PRIVATE_KEY"
```

Do not commit `.env` to GitHub.

## CLI examples

```bash
python -m nexus_alpha.cli --mode source-check --asset BTC --network-check
python -m nexus_alpha.cli --mode signals --asset ALL
python -m nexus_alpha.cli --mode basket --thesis "AI infrastructure and RWA with ETF flow confirmation"
python -m nexus_alpha.cli --mode backtest --thesis "L2 scaling and DeFi bluechip"
python -m nexus_alpha.cli --mode stress --thesis "ETF flow reversal defensive basket"
python -m nexus_alpha.cli --mode execute-preview --thesis "AI infrastructure basket"
```

## Safety model

- Live trading is OFF.
- Execution Preview never submits an order.
- SoDEX private signing is checked only to verify that configuration is ready.
- The dashboard is for research, paper trading, demos, and education.

## Trade history CSV format

The Edge Analyzer expects these columns:

```text
opened_at, closed_at, symbol, side, entry_price, exit_price, size, pnl
```

## Deploy on Streamlit Cloud
https://share.streamlit.io/
https://nexusalpha.streamlit.app/
1. Push this repository to GitHub.
2. Create a Streamlit app using `app.py` as the main file.
3. Add the environment variables in **App settings → Secrets** using TOML.
4. Reboot the app.
