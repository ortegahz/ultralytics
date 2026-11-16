"""First-time entry plan for 159929.SZ with a 10,000 CNY budget.

The user has no position, so this is not a top-up: it is a fresh build. That
changes the design in three ways versus an add-to-a-loser:

1. There is no sunk cost to average down, so each tranche is a plain new buy
   and the ladder is about entry efficiency, not cost-basis defence.
2. The framework's "Underweight / 未持有者不建仓" objection applies at full
   strength to a new position, which is why tranche 1 is deliberately small.
3. The historical edge that the backtest measures (buy at the signal-day
   close, hold 63 sessions) IS tranche 1 -- so tranche 1 is the tested
   trade, and tranches 2-3 are refinements of it rather than alternatives.

Usage:
    python plan_new_entry.py [budget]
"""

from __future__ import annotations

import json
import os
import sys

PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_v] = PROXY

from em_data import fetch_eastmoney_daily, fetch_quote  # noqa: E402

# Tranche definitions: (label, price, per-share risk note)
TRANCHES = [
    ("① 试仓（现在）", 1.265, 1.272, "现价区间内；这是回测里被验证过的那个入场"),
    ("② 无量深跌档", 1.250, 1.256, "仅当跌破 1.25 时未放量（量<5000万股且 MACD 柱>-0.002）"),
    ("③ 右侧确认档", 1.2965, 1.3124, "收复 20 日线 1.2975 后，且 RSI 连2日>50、MACD 柱连2日为正"),
]
HOLD_DAYS = 63  # ~3 months, the horizon the backtest says is the sweet spot


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 10000.0
    quote = fetch_quote()
    px = quote["last"]
    bars = fetch_eastmoney_daily()

    per_tranche = budget / 3
    tranches = []
    total_shares = 0.0
    total_cost = 0.0
    for label, lo, hi, note in TRANCHES:
        mid = (lo + hi) / 2
        shares = int(per_tranche // mid)
        cost = shares * mid
        tranches.append(
            {
                "tranche": label,
                "price_low": lo,
                "price_high": hi,
                "reference_price": round(mid, 4),
                "shares": shares,
                "cost_cny": round(cost, 2),
                "note": note,
            }
        )
        total_shares += shares
        total_cost += cost

    blended = total_cost / total_shares if total_shares else 0.0

    # What the 3-month hold has historically looked like for a single
    # tranche-1 style entry (signal-day close, 63 sessions).
    hist = {
        "胜率(3个月后为正)": 62.4,
        "中位收益%": 3.04,
        "平均收益%": 3.42,
        "最差%": -12.60,
        "最好%": 34.39,
        "持有期内平均最大回撤%": -6.67,
        "持有期内最差回撤%": -17.70,
        "样本数": 295,
    }
    t1_shares = tranches[0]["shares"]
    t1_cost = tranches[0]["cost_cny"]

    out = {
        "as_of": bars.index[-1],
        "live_price": px,
        "prev_close": quote["prev_close"],
        "budget_cny": budget,
        "per_tranche_cny": round(per_tranche, 2),
        "tranches": tranches,
        "all_filled": {
            "total_shares": int(total_shares),
            "capital_cny": round(total_cost, 2),
            "blended_avg_cost": round(blended, 4),
            "blended_vs_spot_pct": round((blended / px - 1) * 100, 2),
        },
        "only_tranche1": {
            "shares": t1_shares,
            "capital_cny": round(t1_cost, 2),
            "at_3m_median": round(t1_cost * hist["中位收益%"] / 100, 2),
            "at_3m_mean": round(t1_cost * hist["平均收益%"] / 100, 2),
            "at_3m_worst": round(t1_cost * hist["最差%"] / 100, 2),
            "at_3m_best": round(t1_cost * hist["最好%"] / 100, 2),
            "typical_drawdown_cny": round(t1_cost * hist["持有期内平均最大回撤%"] / 100, 2),
            "worst_drawdown_cny": round(t1_cost * hist["持有期内最差回撤%"] / 100, 2),
        },
        "historical_3m_distribution": hist,
        "exit_rules": {
            "停止加仓": "日线收盘跌破 1.249（10/23 低点）",
            "取消②档": "跌破 1.25 同时成交量>5000万股 且 MACD 柱<=-0.002",
            "减仓": "日线收盘跌破 1.2357（近15日最低-1xATR）",
            "清仓": "日线收盘跌破 1.22（框架目标价/硬线）",
            "价格止损": "不设近端止损；用 1万元仓位上限控制（ATR 2.37%，-4.5% 止损会把历史胜率从 61.3% 砍到 40.5%）",
            "持有期": f"{HOLD_DAYS} 个交易日（3 个月）；1 个月胜率仅 47.5%，6 个月反降至 45.9%",
        },
    }

    print("=" * 74)
    print(f"159929 首次建仓方案（预算 {budget:,.0f} 元，现价 {px}）")
    print("=" * 74)
    for t in tranches:
        print(f"  {t['tranche']:16} {t['price_low']:.4f}-{t['price_high']:.4f}  "
              f"{t['shares']:>5} 份  {t['cost_cny']:>8.0f} 元")
        print(f"      {t['note']}")
    a = out["all_filled"]
    print(f"\n  三笔全中：{a['total_shares']} 份 / {a['capital_cny']:.0f} 元 / 均价 {a['blended_avg_cost']}")
    o = out["only_tranche1"]
    print(f"\n  只走①档（{o['shares']} 份 / {o['capital_cny']:.0f} 元）拿 3 个月的历史分布：")
    print(f"      中位 {o['at_3m_median']:+.0f} 元 ｜ 平均 {o['at_3m_mean']:+.0f} 元 ｜ "
          f"最差 {o['at_3m_worst']:+.0f} 元 ｜ 最好 {o['at_3m_best']:+.0f} 元")
    print(f"      典型浮亏 {o['typical_drawdown_cny']:+.0f} 元 ｜ 最坏浮亏 {o['worst_drawdown_cny']:+.0f} 元")

    dest = os.path.join("runs", f"159929_{bars.index[-1]}.new_entry.json")
    os.makedirs("runs", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
