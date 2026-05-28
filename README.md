# Nexus Alpha Lab v3.1 API

Crypto/ETF research and paper-trading lab using **SoSoValue** + **SoDEX** as primary connectors.

## What is inside

- SoSoValue connector using `x-soso-api-key`.
- SoSoValue ETF aggregate flow via `/etfs/summary-history`.
- SoSoValue currency snapshot/klines via `/currencies/{currency_id}/...` when available.
- SoDEX spot/perps market data via `/markets/tickers` and `/markets/candles`.
- SoDEX private signing helper for future authenticated endpoints using API key **name** + API private key.
- Automatic market-data backup when primary APIs fail or return incomplete fields.
- Signal Desk: BUY / HOLD / SELL, confidence, entry, stop-loss, take-profit.
- Risk Sentinel, Thesis Basket Builder, Backtest, Monte Carlo, Stress Test, Execution Preview.
- Edge Analyzer from uploaded trade history CSV.
- Paper Ledger and Agent Log.

Live trading is **OFF by default**. The dashboard only produces paper/simulation output and execution previews.

## Setup

```bash
cd nexus-alpha-lab-v3
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill your API settings in `.env`:

```bash
SOSOVALUE_API_KEY=your_sosovalue_key
SOSOVALUE_BASE_URL=https://openapi.sosovalue.com/openapi/v1
SOSOVALUE_ETF_COUNTRY_CODE=US

SODEX_NETWORK=mainnet
SODEX_MARKET=spot
SODEX_API_KEY_NAME=your_sodex_api_key_name
SODEX_API_PRIVATE_KEY=0xYOUR_32_BYTE_API_KEY_PRIVATE_KEY
```

For SoDEX, the docs distinguish the API key name, public key, and private key. `X-API-Key` carries the **API key name**, while `X-API-Sign` is produced locally with the registered API key private key. The private key is never sent to the server.

## Run dashboard

```bash
streamlit run app.py
```

## Run CLI checks

```bash
python -m nexus_alpha.cli --mode source-check --asset ALL
python -m nexus_alpha.cli --mode signals --asset ALL
python -m nexus_alpha.cli --mode signals --asset BTC --capital 10000
```

Other modes:

```bash
python -m nexus_alpha.cli --mode basket --thesis "AI infrastructure and RWA crypto" --profile balanced
python -m nexus_alpha.cli --mode backtest --thesis "L2 scaling and DeFi bluechip"
python -m nexus_alpha.cli --mode stress --thesis "meme and AI infra aggressive basket"
python -m nexus_alpha.cli --mode execute-preview --thesis "RWA conservative portfolio"
```

Force deterministic demo mode:

```bash
python -m nexus_alpha.cli --mode signals --asset ALL --demo
```

## API configuration notes

### SoSoValue

The connector reads:

- `SOSOVALUE_API_KEY`
- `SOSOVALUE_BASE_URL`, default `https://openapi.sosovalue.com/openapi/v1`
- `SOSOVALUE_ETF_COUNTRY_CODE`, default `US`
- `SOSOVALUE_CURRENCY_ID_BTC`, `SOSOVALUE_CURRENCY_ID_ETH`, etc. optional overrides
- `SOSOVALUE_MARKET_PATHS` optional comma-separated path list
- `SOSOVALUE_ETF_PATHS` optional comma-separated path list

ETF default:

```text
GET /etfs/summary-history?symbol=BTC&country_code=US&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&limit=30
```

Currency klines default:

```text
GET /currencies/{currency_id}/klines?interval=1d&limit=30
```

### SoDEX

The connector reads:

- `SODEX_NETWORK`: `mainnet` or `testnet`
- `SODEX_MARKET`: `spot` or `perps`
- `SODEX_SPOT_REST_URL`, default `https://mainnet-gw.sodex.dev/api/v1/spot`
- `SODEX_PERPS_REST_URL`, default `https://mainnet-gw.sodex.dev/api/v1/perps`
- `SODEX_API_KEY_NAME` or legacy `SODEX_API_KEY`
- `SODEX_API_PRIVATE_KEY` or legacy `SODEX_PRIVATE_KEY` / `SODEX_API_SECRET`
- `SODEX_SYMBOL_BTC`, `SODEX_SYMBOL_ETH`, etc. optional symbol mapping

Public market data is unsigned. Private actions require EIP-712 signatures. This build includes a signing helper and config checker, but does **not** submit live orders.

## Trade-history CSV for Edge Analyzer

Recommended columns:

```csv
timestamp,symbol,side,entry,exit,qty,pnl,r_multiple,session,regime
```

The analyzer can normalize common alternatives such as `asset`, `direction`, `profit`, and `r`.

## Safety

This is a research, analysis and paper-trading tool. It is not financial advice and does not submit live orders.
