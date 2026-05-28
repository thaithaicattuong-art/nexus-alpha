from __future__ import annotations

from .portfolio import suggested_allocations
from .risk import RiskDecision
from .signals import TradeSignal


def generate_brief(signals: list[TradeSignal], risks: dict[str, RiskDecision], capital: float = 10_000) -> str:
    ordered = sorted(signals, key=lambda x: abs(x.score) * x.confidence, reverse=True)
    top = ordered[0]
    allowed = [s for s in ordered if risks[s.asset].allowed]
    blocked = [s for s in ordered if not risks[s.asset].allowed]
    alloc = suggested_allocations(signals, risks, capital)

    if allowed:
        mode = "có thể giao dịch chọn lọc"
        action_line = ", ".join(f"{s.asset} {s.action}" for s in allowed[:3])
    else:
        mode = "phòng thủ / đứng ngoài"
        action_line = "không có lệnh nào vượt qua risk filter"

    lines = [
        f"Daily Brief: Market mode hiện tại là **{mode}**. Tín hiệu mạnh nhất là {top.asset} với bias {top.action}, confidence {top.confidence:.0%}, score {top.score:+.2f}.",
        f"Action plan: {action_line}. Không vào full size; dùng position sizing theo SL và giới hạn rủi ro mỗi lệnh.",
    ]
    if blocked:
        lines.append("Risk notes: " + " | ".join(f"{s.asset}: {risks[s.asset].notes[0]}" for s in blocked[:3]))
    active_alloc = alloc[alloc["position_usd"] > 0]
    if not active_alloc.empty:
        lines.append("Suggested allocation: " + ", ".join(f"{r.asset} {r.allocation_pct:.1f}% vốn" for r in active_alloc.itertuples()))
    lines.append("Rule: nếu giá chạm SL hoặc sentiment đảo chiều mạnh, đóng lệnh thay vì trung bình giá.")
    return "\n\n".join(lines)
