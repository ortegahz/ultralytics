"""Averaging-down (补仓) math for 159929.SZ at a given spot price.

For a top-up the one number that decides everything is the holding's current
average cost relative to spot:

  * cost ABOVE spot -> adding at spot lowers the blended cost (真摊平)
  * cost BELOW spot -> adding at spot RAISES the blended cost; that is
                        加仓, not 补仓, and it needs a different argument

The question a holder actually asks is not "what is my new average" but
"where do I have to buy to pull my average down to the level I want". That
is what this solves:

    P* = B * T / (B - Q * (T - C))

where B is the money to add, T the target average, C the current average,
Q the current share count and P* the add price that lands the blend exactly
on T with the full budget spent.

Usage:
    python average_down.py [budget]
"""

from __future__ import annotations

import json
import os
import sys

PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_v] = PROXY

from em_data import fetch_quote  # noqa: E402

TARGETS = [1.30, 1.29, 1.28, 1.27, 1.25]


def blended(cost: float, shares: float, add_price: float, add_shares: float) -> float:
    total = shares + add_shares
    return (cost * shares + add_price * add_shares) / total if total else cost


def price_to_hit_target(cost: float, shares: float, budget: float, target: float) -> float | None:
    """Add price that lands the blended average exactly on ``target``.

    Returns None when the target is unreachable by adding money at all (the
    denominator vanishes, i.e. the target equals the current average).
    """
    denom = budget - shares * (target - cost)
    if denom == 0:
        return None
    return budget * target / denom


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 10000.0
    quote = fetch_quote()
    px = quote["last"]

    sizes = [5000.0, 10000.0, 20000.0, 30000.0]
    costs = [1.25, 1.27, 1.29, 1.31, 1.35, 1.43]

    rows = []
    for cost in costs:
        for value in sizes:
            shares = value / cost
            add_shares = int(budget // px)
            new_cost = blended(cost, shares, px, add_shares)
            hit = {}
            for t in TARGETS:
                p_star = price_to_hit_target(cost, shares, budget, t)
                if p_star is None:
                    hit[f"到{t}需在"] = None
                elif p_star < px:
                    hit[f"到{t}需在"] = round(p_star, 4)
                    hit[f"到{t}需在_可否现价达成"] = "现价偏高，达不到"
                else:
                    hit[f"到{t}需在"] = round(p_star, 4)
                    hit[f"到{t}需在_可否现价达成"] = "现价即可超额达成"
            rows.append(
                {
                    "现成本": cost,
                    "持仓市值": value,
                    "持仓股数": int(shares),
                    "浮盈亏%": round((px / cost - 1) * 100, 2),
                    "补10000后均价": round(new_cost, 4),
                    "均价变动": round(new_cost - cost, 4),
                    "补后总市值": round((shares + add_shares) * px, 2),
                    **hit,
                }
            )

    out = {
        "现价": px,
        "昨收": quote.get("prev_close"),
        "今日区间": f"{quote.get('low')}-{quote.get('high')}",
        "补仓预算": budget,
        "说明": (
            "摊平只在【现成本 > 现价】时成立。若现成本已低于现价，补仓是在抬高均价，"
            "那是加仓不是补仓，理由必须换成看多，不能用'摊低成本'说服自己。"
        ),
        "表": rows,
    }

    print("=" * 104)
    print(f"159929 补仓摊平测算（现价 {px}，今日区间 {quote.get('low')}-{quote.get('high')}，每次补 {budget:,.0f} 元）")
    print("=" * 104)
    print(f"{'现成本':>7} {'持仓市值':>9} {'浮亏%':>8} {'补后均价':>9} {'均价变动':>9} "
          f"{'降到1.27需在':>13} {'降到1.25需在':>13}")
    for r in rows:
        p27 = r.get("到1.27需在")
        p25 = r.get("到1.25需在")
        s27 = f"{p27:.4f}" if isinstance(p27, (int, float)) else "—"
        s25 = f"{p25:.4f}" if isinstance(p25, (int, float)) else "—"
        print(f"{r['现成本']:>7.3f} {r['持仓市值']:>9,.0f} {r['浮盈亏%']:>8.2f} "
              f"{r['补10000后均价']:>9.4f} {r['均价变动']:>+9.4f} {s27:>13} {s25:>13}")

    print("\n说明：『降到 X 需在』= 用这笔预算把均价拉到 X，需要在哪个价位买入。")
    print("      若该价位低于现价，说明按现价补也达不到目标 —— 那就别补，等更低价。")

    dest = os.path.join("runs", f"159929_average_down_{px:.3f}.json")
    os.makedirs("runs", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
