from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from nexus_alpha.basket import basket_to_frame, build_basket
from nexus_alpha.briefing import generate_brief
from nexus_alpha.data import ASSETS, live_market_data
from nexus_alpha.edge import demo_trades, find_edge_notes, normalize_trades, summarize
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

st.set_page_config(page_title="Nexus Alpha Lab v3.1", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.25rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.small-note {color: #777; font-size: 0.9rem;}
.safe-box {padding: .75rem 1rem; border-radius: .75rem; background: #f4f7fb; border: 1px solid #e4e9f2;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Nexus Alpha Lab v3.1")
st.caption("SoSoValue + SoDEX data → flow signal → thesis basket → backtest → Monte Carlo → stress test → execution preview")
st.info("Educational / paper trading only. Live trading mặc định TẮT; mọi execution trong dashboard là preview.", icon="🛡️")

with st.sidebar:
    st.header("Cấu hình")
    assets = st.multiselect("Signal assets", ASSETS, default=ASSETS)
    capital = st.number_input("Vốn mô phỏng USD", min_value=100.0, value=float(os.getenv("DEFAULT_CAPITAL", 10000)), step=500.0)
    risk_per_trade = st.slider("Risk mỗi lệnh", 0.0025, 0.05, float(os.getenv("RISK_PER_TRADE", 0.01)), 0.0025, format="%.3f")
    st.divider()
    st.subheader("Thesis Basket")
    default_thesis = "AI infrastructure and bluechip crypto with ETF flow confirmation"
    thesis = st.text_area("Investment thesis", value=default_thesis, height=110)
    risk_profile = st.selectbox("Basket risk profile", ["conservative", "balanced", "aggressive"], index=1)
    basket_amount = st.number_input("Basket amount USD", min_value=100.0, value=float(capital), step=500.0)
    use_live = st.toggle("Dùng API thật SoSoValue / SoDEX", value=True)
    st.markdown("<p class='small-note'>Nếu API chính thiếu dữ liệu hoặc lỗi, tool tự lấy dữ liệu thị trường phụ để lấp phần còn thiếu.</p>", unsafe_allow_html=True)

market_data, source_statuses = live_market_data(assets, prefer_live=use_live)
signals = generate_signals(market_data)
risks = {s.asset: evaluate_risk(s, market_data[s.asset], capital, risk_per_trade) for s in signals}
basket = build_basket(thesis, basket_amount, risk_profile)

tabs = st.tabs([
    "Signal Desk",
    "API Sources",
    "Thesis Basket",
    "Backtest & Monte Carlo",
    "Stress Test",
    "Execution Preview",
    "Edge Analyzer",
    "Agent Log / Ledger",
    "Daily Brief",
])

with tabs[0]:
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
    selected = st.selectbox("Chọn asset để xem chart", assets or ASSETS)
    df = market_data[selected]
    fig = px.line(df, x="date", y=["net_flow_musd", "sentiment"], markers=True, title=f"{selected}: Net flow and sentiment")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Suggested Paper Allocation")
    st.dataframe(suggested_allocations(signals, risks, capital), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("API Source Status")
    rows = [{"source": stt.name, "ok": stt.ok, "rows": stt.rows, "message": stt.message, "endpoint": stt.endpoint} for stt in source_statuses]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("SoSoValue uses x-soso-api-key. SoDEX market data is unsigned; private signing config is checked separately and live order submission remains disabled.")

with tabs[2]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Basket ID", basket.id)
    c2.metric("Risk score", f"{basket.risk_score}/100")
    c3.metric("Expected vol", f"{basket.expected_annual_vol:.2f}")
    c4.metric("Constituents", len(basket.constituents))
    st.subheader("Basket Proposal")
    st.dataframe(basket_to_frame(basket), use_container_width=True, hide_index=True)
    st.subheader("Reasoning")
    st.write(basket.reasoning)
    if st.button("Save basket JSON + paper ledger", type="primary"):
        out = export_basket_json(basket)
        entry = paper_log_basket(basket)
        st.success(f"Đã lưu {out} và ghi ledger {entry['id']}.")

with tabs[3]:
    c1, c2 = st.columns([1, 1])
    with c1:
        days = st.slider("Backtest days", 30, 180, 90, 15)
        bt, metrics = run_backtest(basket, days)
        st.subheader("Backtest equity curve")
        st.plotly_chart(px.line(bt, x="date", y=[c for c in bt.columns if c != "date"], title="Basket vs benchmark"), use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Return", f"{metrics['total_return_pct']}%")
        m2.metric("Max DD", f"{metrics['max_drawdown_pct']}%")
        m3.metric("Sharpe demo", metrics["sharpe_demo"])
        st.caption("Backtest dùng demo/synthetic history để kiểm tra logic sản phẩm, không phải cam kết lợi nhuận.")
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

with tabs[4]:
    st.subheader("Historical regime / scenario stress tests")
    stress = run_stress_tests(basket)
    st.dataframe(stress, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(stress, x="scenario", y="basket_return_pct", title="Scenario return impact"), use_container_width=True)

with tabs[5]:
    st.subheader("Execution Preview")
    max_slippage = st.slider("Max allowed slippage bps", 5, 150, 35, 5)
    confirm_preview = st.checkbox("Confirm preview gate", value=False, help="Chỉ đổi status sang READY trong paper mode, không gửi order thật.")
    preview = build_execution_preview(basket, max_slippage, confirm_preview)
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.warning("Live execution adapter chưa bật. Đây là IOC limit order preview; SoDEX private signing chỉ được kiểm tra config, không gửi order thật.")

with tabs[6]:
    st.subheader("Conditional Edge Analyzer")
    uploaded = st.file_uploader("Upload trade history CSV", type=["csv"])
    if uploaded:
        trades = pd.read_csv(uploaded)
    else:
        trades = demo_trades()
        st.caption("Đang dùng demo trade history. Upload CSV của bạn để phân tích thật.")
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

with tabs[7]:
    st.subheader("Agentic Loop")
    for step in agent_steps_for_basket(basket):
        st.write(f"**{step['stage']}** — {step['message']}")
    st.subheader("Paper Ledger")
    ledger = read_ledger()
    if ledger.empty:
        st.caption("Chưa có ledger. Bấm save ở tab Thesis Basket để ghi paper ledger.")
    else:
        show_cols = [c for c in ["timestamp", "basket_id", "risk_profile", "amount_usd", "risk_score", "status", "note"] if c in ledger.columns]
        st.dataframe(ledger[show_cols], use_container_width=True, hide_index=True)

with tabs[8]:
    st.subheader("AI-style Daily Brief")
    st.markdown(generate_brief(signals, risks, capital))
    st.divider()
    st.subheader("Basket brief")
    st.write(f"Thesis basket hiện tại có risk score **{basket.risk_score}/100**. Main allocation: " + ", ".join(f"{c.symbol} {c.weight:.1%}" for c in basket.constituents[:4]))
    for sig in signals:
        with st.expander(f"{sig.asset} thesis & reasons"):
            st.write(sig.thesis)
            for reason in sig.reasons:
                st.write("- " + reason)
