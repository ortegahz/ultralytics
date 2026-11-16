"""Deterministic technicals + position math for 159929.SZ, sourced from Eastmoney.

Everything the report states as a number about price, indicators or position
sizing is computed here so it is reproducible and free of LLM arithmetic drift.
Reads the same corrected tape the TradingAgents pipeline saw (yfinance is missing
the 2026-10-13 bar and reports an impossible open>high for 2026-10-14).

Usage:
    python compute_159929_levels.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_v] = PROXY

import pandas as pd  # noqa: E402

from em_data import fetch_eastmoney_daily, fetch_quote  # noqa: E402

# China Standard Time (UTC+8). The A-share session runs 09:30-15:00 CST; a bar
# dated today is therefore partial until shortly after the close.
CST = timezone(timedelta(hours=8))
_SESSION_CLOSE = (15, 5)


def _session_is_open(bar_date: str) -> bool:
    """True when ``bar_date`` is today and the trading session has not closed."""
    now = datetime.now(CST)
    if bar_date != now.strftime("%Y-%m-%d"):
        return False
    return (now.hour, now.minute) < _SESSION_CLOSE


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Standard TA set on the corrected close series."""
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["ema10"] = close.ewm(span=10, adjust=False).mean()
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    out["sma200"] = close.rolling(200).mean()
    out["vwma20"] = (close * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi14"] = 100 - 100 / (1 + gain / loss)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    out["boll_mid"] = mid
    out["boll_ub"] = mid + 2 * sd
    out["boll_lb"] = mid - 2 * sd

    prev_close = close.shift(1)
    tr = pd.concat(
        [(df["High"] - df["Low"]), (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_pct"] = out["atr14"] / close * 100
    out["vol_ma20"] = df["Volume"].rolling(20).mean()
    return out


def main() -> int:
    bars = fetch_eastmoney_daily()
    ind = indicators(bars)

    # The last bar is only a COMPLETED session if the feed carries a bar for a
    # date that has already closed. During the A-share session (09:30-15:00
    # CST) today's bar is partial, so indicators are reported on the previous
    # close and today's partial bar is shown separately.
    today = bars.index[-1]
    last_complete = bars.index[-2] if _session_is_open(today) else today

    quote = {}
    try:
        quote = fetch_quote()
    except Exception as exc:  # noqa: BLE001
        quote = {"error": str(exc)}

    px = quote.get("last") or float(bars["Close"].iloc[-1])

    out: dict = {
        "as_of": today,
        "live_quote": quote,
        "last_complete_bar": last_complete,
        "last_complete_ohlc": {
            k: round(float(v), 4) for k, v in bars.loc[last_complete].items()
        },
        "today_partial_ohlc": {
            k: round(float(v), 4) for k, v in bars.loc[today].items()
        },
        "indicators_on_last_complete": {
            k: (round(float(v), 4) if pd.notna(v) else None)
            for k, v in ind.loc[last_complete].items()
        },
        "indicators_incl_today_partial": {
            k: (round(float(v), 4) if pd.notna(v) else None)
            for k, v in ind.loc[today].items()
        },
    }

    # Where is price versus each average?
    lvl = ind.loc[last_complete]
    out["distance_to"] = {
        name: {
            "value": round(float(lvl[name]), 4),
            "pct_from_last_complete_close": round(
                (float(lvl[name]) / float(lvl["close"]) - 1) * 100, 2
            ),
        }
        for name in ("ema10", "sma20", "sma50", "sma200", "vwma20", "boll_mid", "boll_ub", "boll_lb")
    }

    # Drawdown from the real 3-month high, not just the 9/30 bounce.
    window = bars.loc["2026-07-14":]
    hi_date = window["High"].idxmax()
    lo_date = window["Low"].idxmin()
    out["three_month"] = {
        "high": round(float(window.loc[hi_date, "High"]), 4),
        "high_date": hi_date,
        "low": round(float(window.loc[lo_date, "Low"]), 4),
        "low_date": lo_date,
        "current_vs_high_pct": round((px / float(window.loc[hi_date, "High"]) - 1) * 100, 2),
        "current_vs_low_pct": round((px / float(window.loc[lo_date, "Low"]) - 1) * 100, 2),
        "range_position_pct": round(
            (px - float(window["Low"].min())) / (float(window["High"].max()) - float(window["Low"].min())) * 100,
            1,
        ),
    }

    # Volume: is today's activity confirming anything? Compare against its own
    # partial-session pace rather than a full-day average, which would always
    # look small before the close.
    full_day_vol = bars.loc[last_complete, "Volume"]
    today_vol = bars.loc[today, "Volume"]
    out["volume_context"] = {
        "last_complete_day_volume": int(full_day_vol),
        "today_partial_volume": int(today_vol),
        "avg20_volume": int(ind.loc[last_complete, "vol_ma20"]),
        "today_pct_of_avg_full_day": round(today_vol / float(ind.loc[last_complete, "vol_ma20"]) * 100, 1),
    }

    # ---- Position math ----
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 30000.0
    entry = px
    risk_unit = float(lvl["atr14"])
    out["position_math"] = {
        "entry_price": entry,
        "budget_cny": budget,
        "units_at_entry": int(budget // entry),
        "atr14": round(risk_unit, 4),
        "atr_pct": round(float(lvl["atr_pct"]), 2),
        "staged_plan": [],
    }

    # ATR-derived levels: the only scale-free way to talk about where price is
    # relative to noise, given the hardcoded structural levels from the
    # 2026-10-14 plan (1.261 stop-adding, 1.18 hard stop) may be stale.
    atr = float(lvl["atr14"])
    out["position_math"]["atr_levels"] = {
        f"{k}xATR": {
            "price": round(entry - k * atr, 4),
            "pct_from_entry": round(-k * atr / entry * 100, 2),
        }
        for k in (1.0, 1.5, 2.0, 2.5, 3.0)
    }

    # Structural support: swing lows that actually printed, newest first.
    recent = bars.loc[last_complete:]
    lows = (
        recent["Low"].sort_values().groupby(recent["Low"]).first().sort_values(ascending=False)
    )
    out["position_math"]["swing_lows"] = [
        {"price": round(float(p), 4), "pct_below_entry": round((float(p) / entry - 1) * 100, 2)}
        for p in lows.head(6)
    ]

    # Staged ladder for an ADD: 1/3 now, 1/3 lower into the support band,
    # 1/3 only on a reclaim of the 10-day EMA. Each tranche states the blended
    # average cost so the arithmetic is auditable.
    tranches = [
        ("第一笔 1/3", entry),
        ("第二笔 1/3（回踩支撑）", min(float(lvl["boll_lb"]), float(bars.loc[last_complete, "Low"]))),
        ("第三笔 1/3（确认收复10EMA）", float(lvl["ema10"])),
    ]
    shares = 0
    cost = 0.0
    for label, price in tranches:
        amt = budget / 3
        qty = int(amt // price)
        shares += qty
        cost += qty * price
        out["position_math"]["staged_plan"].append(
            {"tranche": label, "price": round(price, 4), "shares": qty, "amount_cny": round(qty * price, 2)}
        )
    blended = cost / shares if shares else 0.0
    out["position_math"]["blended_avg_cost"] = round(blended, 4)
    out["position_math"]["total_shares"] = shares
    out["position_math"]["capital_deployed_cny"] = round(cost, 2)

    # Invalidation: a fixed structural level is only meaningful while it still
    # sits under the market. Use the swing low that defines the support band,
    # widened by one ATR to tolerate normal noise, and report the ATR-multiple
    # distance from entry so the two framings can be compared.
    support = float(min(bars.loc[last_complete:, "Low"].tail(15)))
    invalidation = round(support - atr, 4)
    out["position_math"]["invalidation_level"] = invalidation
    out["position_math"]["invalidation_basis"] = (
        f"近15日最低 {support:.4f} 减 1×ATR({atr:.4f})"
    )
    out["position_math"]["invalidation_pct_from_entry"] = round(
        (invalidation / entry - 1) * 100, 2
    )
    out["position_math"]["loss_at_invalidation_cny"] = round((invalidation - blended) * shares, 2)
    out["position_math"]["loss_at_invalidation_pct"] = round((invalidation / blended - 1) * 100, 2)

    # 1-2% of account risk rule: what price stop is implied for a given account.
    for account in (100000.0, 300000.0):
        risk_budget = account * 0.01
        per_share_risk = blended - invalidation
        max_shares = int(risk_budget / per_share_risk) if per_share_risk > 0 else 0
        out["position_math"][f"account_{int(account)}_1pct_risk"] = {
            "account_cny": account,
            "risk_budget_cny": round(risk_budget, 2),
            "max_shares_by_stop": max_shares,
            "max_capital_at_blended_cny": round(max_shares * blended, 2),
            "note": "与上方预算取较小者",
        }

    dest = os.path.join("runs", f"159929_{today}.levels.json")
    os.makedirs("runs", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())