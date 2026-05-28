from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from nexus_alpha.basket import basket_to_frame, build_basket
from nexus_alpha.briefing import generate_brief
from nexus_alpha.data import ASSETS, live_market_data
from nexus_alpha.edge import demo_trades, find_edge_notes, normalize_trades, summarize
from nexus_alpha.intelligence import (
    api_usage_scorecard,
    build_submission_checklist,
    demo_scenarios,
    demo_script_text,
    flow_rotation_table,
    liquidity_map,
    market_regime,
    rebalance_plan,
    signal_factor_table,
)
from nexus_alpha.lab import (
    agent_steps_for_basket,
    build_execution_preview,
    export_basket_json,
    paper_log_basket,
    read_ledger,
    run_backtest,
    run_monte_carlo,
    run_stress_tests,
)
from nexus_alpha.portfolio import decision_log, suggested_allocations
from nexus_alpha.risk import evaluate_risk
from nexus_alpha.signals import generate_signals

load_dotenv()

st.set_page_config(page_title="Nexus Alpha", page_icon="⚡", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.25rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.small-note {color: #777; font-size: 0.9rem;}
.safe-box {padding: .75rem 1rem; border-radius: .75rem; background: #f4f7fb; border: 1px solid #e4e9f2;}
.kpi-card {padding: 1rem; border: 1px solid rgba(120,120,120,.25); border-radius: 1rem; background: rgba(120,120,120,.08);}
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚡ Nexus Alpha")
st.caption("SoSoValue + SoDEX data → market intelligence → flow signals → thesis baskets → risk lab → execution preview")
st.info("Educational / paper trading only.", icon="🛡️")

with st.sidebar:
    st.header("Settings")
    assets = st.multiselect("Signal assets", ASSETS, default=ASSETS)
    capital = st.number_input("Simulated Capital USD", min_value=100.0, value=float(os.getenv("DEFAULT_CAPITAL", 10000)), step=500.0)
    risk_per_trade = st.slider("Risk per trade", 0.0025, 0.05, float(os.getenv("RISK_PER_TRADE", 0.01)), 0.0025, format="%.3f")
    st.divider()
    st.subheader("Thesis Basket")
    default_thesis = "AI infrastructure and bluechip crypto with ETF flow confirmation"
    thesis = st.text_area("Investment thesis", value=default_thesis, height=110)
    risk_profile = st.selectbox("Basket risk profile", ["conservative", "balanced", "aggressive"], index=1)
    basket_amount = st.number_input("Basket amount USD", min_value=100.0, value=float(capital), step=500.0)
    use_live = st.toggle("Use live SoSoValue / SoDEX API", value=True)
    st.markdown("<p class='small-note'>If a primary API is missing data or fails, the app fills missing market fields through a secondary market-data route.</p>", unsafe_allow_html=True)

selected_assets = assets or ASSETS
market_data, source_statuses = live_market_data(selected_assets, prefer_live=use_live)
signals = generate_signals(market_data)
risks = {s.asset: evaluate_risk(s, market_data[s.asset], capital, risk_per_trade) for s in signals}
basket = build_basket(thesis, basket_amount, risk_profile)
factors = signal_factor_table(market_data, signals, risks)
regime = market_regime(market_data)
rotations = flow_rotation_table(market_data)

allowed_count = sum(1 for r in risks.values() if r.allowed)
ready_sources = sum(1 for s in source_statuses if s.ok)

summary_cols = st.columns(5)
summary_cols[0].metric("Market regime", regime["mode"], f"score {regime['score']:+.2f}")
summary_cols[1].metric("Tradable setups", allowed_count, f"of {len(signals)}")
summary_cols[2].metric("Basket risk", f"{basket.risk_score}/100", f"{basket.risk_profile}")
summary_cols[3].metric("API checks", ready_sources, f"of {len(source_statuses)}")
summary_cols[4].metric("Paper capital", f"${capital:,.0f}", f"risk {risk_per_trade:.2%}")


tabs = st.tabs([
    "Command Center",
    "Signal Desk",
    "API Sources",
    "Thesis Basket",
    "Backtest & Monte Carlo",
    "Stress Test",
    "Execution Preview",
    "Edge Analyzer",
    "Agent Log / Ledger",
    "Demo Center",
    "Daily Brief",
])

with tabs[0]:
    st.subheader("Command Center")
    st.write(regime["summary"])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Signal factor matrix**")
        st.dataframe(factors, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Flow rotation leaderboard**")
        st.dataframe(rotations, use_container_width=True, hide_index=True)
        if not rotations.empty:
            fig = px.bar(rotations, x="asset", y="flow_acceleration_musd", color="rotation_label", title="7-day flow acceleration")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Buildathon readiness checklist**")
    st.dataframe(build_submission_checklist(source_statuses, basket, signals), use_container_width=True, hide_index=True)

with tabs[1]:
    cols = st.columns(max(1, len(signals)))
    for col, sig in zip(cols, signals):
        risk = risks[sig.asset]
        with col:
            st.metric(sig.asset, sig.action, f"score {sig.score:+.2f}")
            st.write(f"Confidence: **{sig.confidence:.0%}**")
            st.write(f"Entry `{sig.entry}` | SL `{sig.stop_loss}` | TP `{sig.take_profit}`")
            st.write(f"Risk: **{risk.risk_level}** | Size: **${risk.position_usd:,.0f}**")
            for n in risk.notes[:2]:
                st.caption(n)

    st.subheader("30-day ETF Flow / Market Series")
    selected = st.selectbox("Select asset for chart", selected_assets or ASSETS)
    df = market_data[selected]
    chart_fields = st.multiselect("Chart fields", ["net_flow_musd", "sentiment", "price", "volume_ratio"], default=["net_flow_musd", "sentiment"])
    fig = px.line(df, x="date", y=chart_fields, markers=True, title=f"{selected}: market and flow series")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Suggested Paper Allocation")
    st.dataframe(suggested_allocations(signals, risks, capital), use_container_width=True, hide_index=True)
    st.subheader("Decision Log")
    st.dataframe(decision_log(signals, risks), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("API Source Status")
    source_frame = api_usage_scorecard(source_statuses)
    st.dataframe(source_frame, use_container_width=True, hide_index=True)
    ok_count = int((source_frame["status"] == "OK").sum()) if not source_frame.empty else 0
    st.progress(min(ok_count / max(len(source_frame), 1), 1.0), text=f"{ok_count}/{len(source_frame)} checks ready")
    st.caption("SoSoValue uses x-soso-api-key. SoDEX market data is unsigned; private signing configuration is checked separately and live order submission remains disabled.")

with tabs[3]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Basket ID", basket.id)
    c2.metric("Risk score", f"{basket.risk_score}/100")
    c3.metric("Expected vol", f"{basket.expected_annual_vol:.2f}")
    c4.metric("Constituents", len(basket.constituents))
    st.subheader("Basket Proposal")
    basket_frame = basket_to_frame(basket)
    st.dataframe(basket_frame, use_container_width=True, hide_index=True)

    if not basket_frame.empty:
        fig = px.pie(basket_frame, names="symbol", values="weight_pct", title="Basket weights")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rebalance Plan")
    rebalance = rebalance_plan(basket, factors)
    st.dataframe(rebalance, use_container_width=True, hide_index=True)

    st.subheader("Reasoning")
    st.write(basket.reasoning)
    if st.button("Save basket JSON + paper ledger", type="primary"):
        out = export_basket_json(basket)
        entry = paper_log_basket(basket)
        st.success(f"Saved {out} and wrote ledger entry {entry['id']}.")

with tabs[4]:
    c1, c2 = st.columns([1, 1])
    with c1:
        days = st.slider("Backtest days", 30, 180, 90, 15)
        bt, metrics = run_backtest(basket, days)
        st.subheader("Backtest equity curve")
        st.plotly_chart(px.line(bt, x="date", y=[c for c in bt.columns if c != "date"], title="Basket vs benchmark"), use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Return", f"{metrics['total_return_pct']}%")
        m2.metric("Benchmark", f"{metrics['benchmark_return_pct']}%")
        m3.metric("Max DD", f"{metrics['max_drawdown_pct']}%")
        m4.metric("Sharpe demo", metrics["sharpe_demo"])
        st.caption("Backtest uses demo/synthetic history to validate product logic. It is not a profit promise.")
    with c2:
        horizon = st.slider("Monte Carlo horizon", 7, 90, 30, 1)
        paths = st.select_slider("MC paths", options=[250, 500, 1000, 2000], value=1000)
        fan, stats = run_monte_carlo(basket, horizon, paths)
        st.subheader("Monte Carlo fan chart")
        st.plotly_chart(px.line(fan, x="day", y="value", color="path", title="Sampled simulated paths"), use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected terminal", f"${stats['expected_terminal_usd']:,.0f}")
        m2.metric("Prob. loss", f"{stats['prob_loss_pct']}%")
        m3.metric("CVaR 95%", f"{stats['cvar_95_pct']}%")

with tabs[5]:
    st.subheader("Historical regime / scenario stress tests")
    stress = run_stress_tests(basket)
    st.dataframe(stress, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(stress, x="scenario", y="basket_return_pct", title="Scenario return impact"), use_container_width=True)
    st.plotly_chart(px.bar(stress, x="scenario", y="max_drawdown_pct", title="Scenario drawdown impact"), use_container_width=True)

with tabs[6]:
    st.subheader("Execution Preview")
    c1, c2 = st.columns([1, 1])
    with c1:
        max_slippage = st.slider("Max allowed slippage bps", 5, 150, 35, 5)
        confirm_preview = st.checkbox("Confirm preview gate", value=False, help="This only changes status to READY in paper mode. It does not send live orders.")
        preview = build_execution_preview(basket, max_slippage, confirm_preview)
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.warning("Live execution adapter is disabled. This is an IOC limit order preview; SoDEX private signing is checked for readiness but no order is submitted.")
    with c2:
        st.markdown("**Liquidity Map**")
        liq = liquidity_map(basket)
        st.dataframe(liq, use_container_width=True, hide_index=True)
        if not liq.empty:
            st.plotly_chart(px.scatter(liq, x="notional_usd", y="estimated_slippage_bps", size="notional_usd", color="execution_quality", hover_name="symbol", title="Notional vs estimated slippage"), use_container_width=True)

with tabs[7]:
    st.subheader("Conditional Edge Analyzer")
    uploaded = st.file_uploader("Upload trade history CSV", type=["csv"])
    if uploaded:
        trades = pd.read_csv(uploaded)
    else:
        trades = demo_trades()
        st.caption("Using demo trade history. Upload your own CSV to analyze real performance.")
    try:
        trades = normalize_trades(trades)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", len(trades))
        c2.metric("Total PnL", f"${trades['pnl'].sum():,.0f}")
        c3.metric("Winrate", f"{(trades['win'].mean()*100):.1f}%")
        c4.metric("Avg R", f"{trades['r_multiple'].mean():.2f}")
        dim = st.selectbox("Slice by", ["symbol", "side", "session", "regime"])
        sliced = summarize(trades, dim)
        st.dataframe(sliced, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(sliced, x=dim, y="pnl", title=f"PnL by {dim}"), use_container_width=True)
        st.subheader("Edge Notes")
        for note in find_edge_notes(trades):
            st.write("- " + note)
    except Exception as e:
        st.error(str(e))

with tabs[8]:
    st.subheader("Agentic Loop")
    for step in agent_steps_for_basket(basket):
        st.write(f"**{step['stage']}** — {step['message']}")
    st.subheader("Paper Ledger")
    ledger = read_ledger()
    if ledger.empty:
        st.caption("No ledger entries yet. Save a basket in the Thesis Basket tab to create a paper ledger record.")
    else:
        show_cols = [c for c in ["timestamp", "basket_id", "risk_profile", "amount_usd", "risk_score", "status", "note"] if c in ledger.columns]
        st.dataframe(ledger[show_cols], use_container_width=True, hide_index=True)

with tabs[9]:
    st.subheader("Wave 2 Demo Center")
    st.markdown("**One-minute demo script**")
    st.code(demo_script_text(), language="markdown")
    st.markdown("**Judge walkthrough scenarios**")
    st.dataframe(pd.DataFrame([s.__dict__ for s in demo_scenarios()]), use_container_width=True, hide_index=True)
    st.markdown("**Submission readiness**")
    st.dataframe(build_submission_checklist(source_statuses, basket, signals), use_container_width=True, hide_index=True)

with tabs[10]:
    st.subheader("AI-style Daily Brief")
    st.markdown(generate_brief(signals, risks, capital))
    st.divider()
    st.subheader("Basket brief")
    st.write(f"The current thesis basket has a risk score of **{basket.risk_score}/100**. Main allocation: " + ", ".join(f"{c.symbol} {c.weight:.1%}" for c in basket.constituents[:4]))
    for sig in signals:
        with st.expander(f"{sig.asset} thesis & reasons"):
            st.write(sig.thesis)
            for reason in sig.reasons:
                st.write("- " + reason)
