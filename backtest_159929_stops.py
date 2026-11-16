"""Stop-width sensitivity for the staged-entry plan on 159929.

The 2026-10-14 report had to retract a 1.23 stop (about -4.5%): the ATR on
this ETF is ~2.2-2.4% a day, so a 4.5% stop sits about two ATRs away, and
the backtest showed it firing on 60% of signals -- cutting good trades, not
bad ones, and dropping the 3-month win rate from ~60% to ~38%.

This script re-runs that same staged plan over every stop width from -4.5%
to "no stop", on the tape refreshed through the latest session, so the
stop/position-size trade-off is stated with current numbers rather than
October numbers.

Usage:
    python backtest_159929_stops.py
"""

from __future__ import annotations

import json
import os

PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_v] = PROXY

import pandas as pd  # noqa: E402

import backtest_159929_plan as plan  # noqa: E402
from backtest_159929_winrate import load_pe  # noqa: E402
from em_data import fetch_eastmoney_daily  # noqa: E402

STOPS = [
    ("-4.5% (10/14 报告曾用的 1.23)", -0.045),
    ("-6.0%", -0.060),
    ("-7.0%", -0.070),
    ("-8.0%", -0.080),
    ("-10.0% (10/14 修正版 1.18)", -0.100),
    ("-12.0%", -0.120),
    ("-15.0%", -0.150),
    ("不止损（只靠仓位控制）", -1.0),
]


def signal_indices(bars: pd.DataFrame, pe: pd.Series) -> list[int]:
    close = bars["Close"]
    lo60 = close.rolling(60).min()
    hi60 = close.rolling(60).max()
    pos60 = (close - lo60) / (hi60 - lo60).replace(0, pd.NA)

    dates = sorted(set(bars.index) & set(pe.index))
    pe_pct: dict[str, float] = {}
    for d in dates:
        p = pe.index.get_loc(d)
        w = pe.iloc[max(0, p - 365 * 3) : p + 1].dropna()
        pe_pct[d] = float((w < w.iloc[-1]).sum()) / len(w) * 100 if len(w) else float("nan")

    return [
        bars.index.get_loc(d)
        for d in dates
        if pd.notna(pos60.get(d)) and pd.notna(pe_pct.get(d))
        and pos60[d] <= 0.25 and pe_pct[d] <= 20.0
    ]


def main() -> int:
    bars = fetch_eastmoney_daily()
    pe = load_pe()
    close = bars["Close"]
    signals = signal_indices(bars, pe)

    # Effective independent sample: overlapping 63-session windows inflate N.
    eff = 1
    prev = signals[0]
    for i in signals[1:]:
        if i - prev > plan.HORIZON:
            eff += 1
        prev = i

    rows = []
    for label, stop in STOPS:
        plan.STOP_LOSS = stop
        sims = [s for s in (plan.simulate(close, i) for i in signals) if s]
        r = pd.Series([s["ret"] for s in sims])
        m = pd.Series([s["mdd"] for s in sims])
        rows.append(
            {
                "止损设置": label,
                "止损比例": stop if stop > -0.9 else None,
                "样本数": len(sims),
                "胜率%": round(float((r > 0).mean() * 100), 1),
                "中位收益%": round(float(r.median()), 2),
                "平均收益%": round(float(r.mean()), 2),
                "最差%": round(float(r.min()), 2),
                "最好%": round(float(r.max()), 2),
                "平均最大回撤%": round(float(m.mean()), 2),
                "被止损次数": sum(s["stopped"] for s in sims),
                "三笔全成交": sum(s["n_tranches"] == 3 for s in sims),
            }
        )

    out = {
        "回测区间": f"{bars.index[0]} ~ {bars.index[-1]}",
        "信号条件": "60日区间位置<25% 且 中证医药(000933) PE 处于3年20%分位以下",
        "有效独立样本数(去重叠)": eff,
        "持有期": "63个交易日 (~3个月)",
        "分批规则": f"第1笔 信号日收盘; 第2笔 回撤{plan.DIP_TRIGGER:.0%}/30日内; "
                    f"第3笔 上涨{plan.CONF_TRIGGER:.1%}/30日内",
        "止损敏感性": rows,
    }

    print("=" * 78)
    print(f"止损宽度敏感性（数据至 {bars.index[-1]}，独立样本 {eff} 个）")
    print("=" * 78)
    print(f"{'止损设置':30} {'胜率':>7} {'中位':>8} {'平均':>8} {'最差':>8} {'止损次数':>8}")
    for row in rows:
        print(
            f"{row['止损设置']:30} {row['胜率%']:>6.1f}% {row['中位收益%']:>+8.2f} "
            f"{row['平均收益%']:>+8.2f} {row['最差%']:>+8.2f} {row['被止损次数']:>8}"
        )

    dest = os.path.join("runs", f"159929_{bars.index[-1]}.stop_sensitivity.json")
    os.makedirs("runs", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
