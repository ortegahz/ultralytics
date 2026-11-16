"""Empirical win-rate for the "buy a third at the box bottom" plan on 159929.

Answers the user's question "胜率多少" with a backtest instead of a guess.

Signal (the situation we are actually in on 2026-10-14):
    * price in the bottom X% of its trailing 60-session range, AND
    * index PE in the bottom Y% of its trailing 3-year percentile.

Both series are real: 159929 daily bars from Eastmoney, and the CSI 000933
official daily PE series. Forward returns are measured on the ETF close, and
each sample is dropped if its full holding window is not yet observed, so no
look-ahead leaks in.
"""

from __future__ import annotations

import json
import os

PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_v] = PROXY

import pandas as pd  # noqa: E402

from em_data import _get_json, fetch_eastmoney_daily, fetch_quote  # noqa: E402

PE_URL = "https://www.csindex.com.cn/csindex-home/perf/indexCsiDsPe?indexCode=000933"
HOLD_DAYS = {"1个月(21日)": 21, "3个月(63日)": 63, "6个月(126日)": 126}


def load_pe() -> pd.Series:
    """Official CSI daily PE series for 800医卫 / 中证医药卫生 (index 000933)."""
    rows = _get_json(PE_URL, 30, attempts=6)["data"]
    s = pd.Series(
        [float(r["peg"]) for r in rows],  # field is misnamed 'peg'; endpoint is indexCsiDsPe
        index=pd.to_datetime([r["tradeDate"] for r in rows]).strftime("%Y-%m-%d"),
    )
    return s.sort_index()


def trailing_percentile(series: pd.Series, window_days: int, current_date: str) -> float:
    """Percentile rank of the value at ``current_date`` within its own trailing window."""
    idx = series.index
    pos = idx.get_loc(current_date) if current_date in idx else None
    if pos is None:
        return float("nan")
    lo = max(0, pos - window_days)
    window = series.iloc[lo : pos + 1].dropna()
    if window.empty:
        return float("nan")
    return float((window < window.iloc[-1]).sum()) / len(window) * 100


def main() -> int:
    bars = fetch_eastmoney_daily()
    pe = load_pe()
    close = bars["Close"]

    print(f"ETF bars : {len(bars)}  {bars.index[0]} -> {bars.index[-1]}")
    print(f"PE series: {len(pe)}  {pe.index[0]} -> {pe.index[-1]}\n")

    dates = sorted(set(bars.index) & set(pe.index))
    lo60 = close.rolling(60).min()
    hi60 = close.rolling(60).max()
    rng = (hi60 - lo60).replace(0, pd.NA)
    pos60 = (close - lo60) / rng

    # Cache the PE percentile per date once; it is the slow part.
    pe_pct_3y = {d: trailing_percentile(pe, 365 * 3, d) for d in dates}

    results = {}
    for cond_name, (pos_cap, pe_cap) in {
        "严格匹配当前状态 (区间<25% 且 PE<3年20%分位)": (0.25, 20.0),
        "宽松 (区间<35% 且 PE<3年35%分位)": (0.35, 35.0),
    }.items():
        hits = []
        for d in dates:
            p = pos60.get(d, float("nan"))
            q = pe_pct_3y.get(d, float("nan"))
            if pd.isna(p) or pd.isna(q):
                continue
            if p <= pos_cap and q <= pe_cap:
                hits.append(d)

        cond_res = {"condition": cond_name, "n_signals": len(hits), "first": hits[0] if hits else None,
                    "last": hits[-1] if hits else None, "windows": {}}
        if not hits:
            results[cond_name] = cond_res
            continue

        idx = {d: i for i, d in enumerate(bars.index)}
        for label, n in HOLD_DAYS.items():
            rets, mdds = [], []
            for d in hits:
                i = idx[d]
                if i + n >= len(bars):
                    continue  # window not yet observed -> drop, no look-ahead
                entry, exit_ = close.iloc[i], close.iloc[i + n]
                rets.append((exit_ / entry - 1) * 100)
                path = close.iloc[i + 1 : i + n + 1]
                mdds.append((path.min() / entry - 1) * 100)
            if not rets:
                continue
            s = pd.Series(rets)
            cond_res["windows"][label] = {
                "样本数": len(rets),
                "胜率(收益>0)": round(float((s > 0).mean() * 100), 1),
                "中位数收益%": round(float(s.median()), 2),
                "平均收益%": round(float(s.mean()), 2),
                "最差%": round(float(s.min()), 2),
                "最好%": round(float(s.max()), 2),
                "持有期内平均最大回撤%": round(float(pd.Series(mdds).mean()), 2),
                "持有期内最差回撤%": round(float(pd.Series(mdds).min()), 2),
            }
        results[cond_name] = cond_res

    # Where do we stand today?
    today = max(d for d in dates if d <= bars.index[-1])
    try:
        live = fetch_quote().get("last")
    except Exception:  # noqa: BLE001 - live quote is a nicety, not the analysis
        live = None
    cur = {
        "当前日期": today,
        "最新价": round(float(close.iloc[-1]), 4),
        "实时价": live,
        "60日区间位置%": round(float(pos60.iloc[-1]) * 100, 1),
        "PE": round(float(pe.get(today, float("nan"))), 2),
        "PE_3年分位%": round(pe_pct_3y.get(today, float("nan")), 1),
    }

    out = {"current": cur, "backtest": results}
    dest = os.path.join("runs", "159929_2026-10-14.winrate.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("=" * 66)
    print("当前状态")
    print("=" * 66)
    for k, v in cur.items():
        print(f"  {k:16} {v}")

    for name, r in results.items():
        print("\n" + "=" * 66)
        print(f"{name}\n  触发次数 {r['n_signals']}  ({r['first']} ~ {r['last']})")
        print("=" * 66)
        for label, w in r["windows"].items():
            print(f"  【{label}】 样本 {w['样本数']}")
            print(f"     胜率        {w['胜率(收益>0)']:5.1f}%")
            print(f"     中位收益    {w['中位数收益%']:+6.2f}%   平均 {w['平均收益%']:+6.2f}%")
            print(f"     最差/最好   {w['最差%']:+6.2f}% / {w['最好%']:+6.2f}%")
            print(f"     平均最大回撤 {w['持有期内平均最大回撤%']:+6.2f}%   最差 {w['持有期内最差回撤%']:+6.2f}%")

    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())