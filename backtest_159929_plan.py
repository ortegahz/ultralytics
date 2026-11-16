"""Simulate the actual staged-entry plan on 159929 and report its win rate.

The plain "buy once at the box bottom" backtest (backtest_159929_winrate.py)
answers a different question. This simulates the recommendation actually given
to the user:

    tranche 1  1/3  at signal close
    tranche 2  1/3  only if price dips >=2.0% within 30 sessions
    tranche 3  1/3  only if price rises >=1.5% above tranche 1 within 30 sessions
    stop       exit everything at -4.5% from the average cost (the 1.23 level)
    horizon    63 sessions (~3 months)

Tranches that never trigger stay in cash, which is the whole point of the
plan, so this measures the recommendation rather than a single entry.
"""

from __future__ import annotations

import json
import os

PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_v] = PROXY

import pandas as pd  # noqa: E402

from backtest_159929_winrate import load_pe  # noqa: E402
from em_data import fetch_eastmoney_daily  # noqa: E402

DIP_TRIGGER = -0.020      # tranche 2 needs a 2% pullback
CONF_TRIGGER = +0.015      # tranche 3 needs a 1.5% reclaim
STOP_LOSS = -0.045# the 1.23 invalidation, ~4.5% from ~1.289
HORIZON = 63
TRIGGER_WINDOW = 30        # tranches 2/3 may fill within this many sessions


def simulate(close: pd.Series, i: int) -> dict | None:
    """Run one staged plan starting at bar ``i``. None if the horizon is short."""
    if i + HORIZON + TRIGGER_WINDOW + 1 >= len(close):
        return None

    px1 = float(close.iloc[i])
    px2 = px3 = None
    filled2 = filled3 = False

    # Resolve tranche fills first, in chronological order, so the average cost
    # is realistic. A later fill at a worse price can be skipped (cash stays).
    for j in range(i + 1, min(i + 1 + TRIGGER_WINDOW, len(close))):
        p = float(close.iloc[j])
        if not filled2 and p <= px1 * (1 + DIP_TRIGGER):
            px2 = p
            filled2 = True
        elif not filled3 and p >= px1 * (1 + CONF_TRIGGER):
            px3 = p
            filled3 = True
        if filled2 and filled3:
            break

    prices = [px1] + ([px2] if px2 else []) + ([px3] if px3 else [])
    shares = 1.0 / len(prices)
    avg = sum(prices) / len(prices)

    # The bar where the last tranche filled: the stop may only apply from there.
    last_fill = i
    for j in range(i + 1, min(i + 1 + TRIGGER_WINDOW, len(close))):
        p = float(close.iloc[j])
        if (px2 and abs(p - px2) < 1e-9) or (px3 and abs(p - px3) < 1e-9):
            last_fill = j

    # Walk forward for the stop, starting once fully deployed.
    stop_i = None
    for j in range(last_fill + 1, min(i + HORIZON + 1, len(close))):
        if float(close.iloc[j]) <= avg * (1 + STOP_LOSS):
            stop_i = j
            break

    if stop_i is not None:
        exit_i = stop_i
        exit_px = avg * (1 + STOP_LOSS)
        stopped = True
    else:
        exit_i = min(i + HORIZON, len(close) - 1)
        exit_px = float(close.iloc[exit_i])
        stopped = False

    ret = (exit_px / avg - 1) * 100
    path = close.iloc[last_fill + 1 : exit_i + 1]
    mdd = (float(path.min()) / avg - 1) * 100 if len(path) else 0.0

    return {
        "ret": ret, "stopped": stopped, "mdd": mdd,
        "n_tranches": len(prices),
    }


def main() -> int:
    bars = fetch_eastmoney_daily()
    pe = load_pe()
    close = bars["Close"]

    lo60 = close.rolling(60).min()
    hi60 = close.rolling(60).max()
    pos60 = (close - lo60) / (hi60 - lo60).replace(0, pd.NA)

    dates = sorted(set(bars.index) & set(pe.index))
    pe_pct = {}
    for d in dates:
        idx = pe.index
        p = idx.get_loc(d)
        w = pe.iloc[max(0, p - 365 * 3) : p + 1].dropna()
        pe_pct[d] = float((w < w.iloc[-1]).sum()) / len(w) * 100 if len(w) else float("nan")

    signals = [
        (bars.index.get_loc(d))
        for d in dates
        if pd.notna(pos60.get(d)) and pd.notna(pe_pct.get(d))
        and pos60[d] <= 0.25 and pe_pct[d] <= 20.0
    ]
    sims = [s for s in (simulate(close, i) for i in signals) if s]

    r = pd.Series([s["ret"] for s in sims])
    m = pd.Series([s["mdd"] for s in sims])
    stopped = sum(s["stopped"] for s in sims)
    all3 = sum(s["n_tranches"] == 3 for s in sims)
    only1 = sum(s["n_tranches"] == 1 for s in sims)

    # Effective independent sample: signals cluster, so overlapping 63-session
    # windows inflate N. Count runs separated by >63 sessions.
    eff, prev = 1, signals[0]
    for i in signals[1:]:
        if i - prev > HORIZON:
            eff += 1
        prev = i

    out = {
        "参数": {"分批1_触发": "信号日收盘", "分批2_触发": f"回撤{DIP_TRIGGER:.0%}/30日内",
                "分批3_触发": f"上涨{CONF_TRIGGER:.1%}/30日内", "止损": f"{STOP_LOSS:.1%}",
                "持有期": f"{HORIZON}个交易日"},
        "信号数": len(sims),
        "有效独立样本数(去重叠)": eff,
        "胜率(3个月后为正)": round(float((r > 0).mean() * 100), 1),
        "中位数收益%": round(float(r.median()), 2),
        "平均收益%": round(float(r.mean()), 2),
        "最差%": round(float(r.min()), 2),
        "最好%": round(float(r.max()), 2),
        "平均最大回撤%": round(float(m.mean()), 2),
        "触发止损次数": stopped,
        "只成交1笔(另2笔未触发)": only1,
        "三笔全成交": all3,
    }

    print("=" * 60)
    print("分批建仓方案回测（2022-02 至今，同一触发条件）")
    print("=" * 60)
    for k, v in out.items():
        print(f"  {k:24} {v}")

    dest = os.path.join("runs", "159929_2026-10-14.plan_winrate.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())