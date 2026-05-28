from nexus_alpha.basket import basket_to_frame, build_basket
from nexus_alpha.data import demo_market_data
from nexus_alpha.edge import demo_trades, summarize
from nexus_alpha.lab import build_execution_preview, run_backtest, run_monte_carlo, run_stress_tests
from nexus_alpha.risk import evaluate_risk
from nexus_alpha.signals import generate_signals


def test_signal_and_risk():
    data = demo_market_data(["BTC", "ETH"])
    signals = generate_signals(data)
    assert len(signals) == 2
    for signal in signals:
        risk = evaluate_risk(signal, data[signal.asset])
        assert risk.risk_level in {"LOW", "MEDIUM", "BLOCKED"}


def test_edge_summary():
    trades = demo_trades(50)
    out = summarize(trades, "symbol")
    assert not out.empty
    assert "winrate" in out.columns


def test_basket_lab():
    basket = build_basket("AI infrastructure and RWA bluechip", 10000, "balanced")
    assert basket.constituents
    assert 0 <= basket.risk_score <= 100
    assert not basket_to_frame(basket).empty
    _, metrics = run_backtest(basket, 45)
    assert "max_drawdown_pct" in metrics
    _, stats = run_monte_carlo(basket, 7, 50)
    assert stats["paths"] == 50
    assert not run_stress_tests(basket).empty
    assert not build_execution_preview(basket).empty
