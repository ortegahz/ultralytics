"""Replay the 2026-10-14 three-tranche plan for 159929.SZ against the real tape.

The 2026-10-14 plan was:
    tranche 1  1/3 at 1.29-1.295   -- buy that day
    tranche 2  1/3 at 1.265        -- pullback into the support band
    tranche 3  1/3 near 1.312      -- only after the Q3 reporting window
                                      closes (10/31); may go early if WuXi
                                      AppTec (10/27) and Hengrui (10/28) land
    guards     close below 1.261 -> stop adding; below 1.18 -> exit

This script asks four questions against the Eastmoney tape, with no LLM in
the loop:

    1. Did each tranche actually fill, on which session, at what price?
    2. Did the 1.261 "stop adding" guard fire on a closing basis, or only
       intraday? (Those two are very different signals.)
    3. What is the blended cost of the plan as executed, versus today's price?
    4. If tranche 3 had never been filled, would the position look better?

Usage:
    python review_plan_1014.py
"""

from __future__ import annotations

import json
import os

PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_v] = PROXY

import pandas as pd  # noqa: E402

from em_data import fetch_eastmoney_daily, fetch_quote  # noqa: E402

PLAN_START = "2026-10-14"
# direction: "down" = passive limit buy on the way down (tranches 1-2);
#            "up"   = stop buy on the way up, i.e. a breakout confirmation
#            (tranche 3 -- buying a rally to 1.312 is not the same order as
#            buying a pullback to 1.265, and replaying them the same way
#            silently fills tranche 3 on the first bar).
TRANCHES = [
    # label, trigger price, limit, window end, direction, earliest_executable
    # Tranche 3 was gated on the reporting window: the plan said "decide after
    # the Q3 reports are out (10/31), may go early if WuXi AppTec (10/27) and
    # Hengrui (10/28) both land". Both results were public from the morning of
    # 10/28, so 10-28 is the earliest session on which the gate could open.
    ("第1笔 1/3 @1.29-1.295", 1.29, 1.295, "2026-10-14", "down", "2026-10-14"),
    ("第2笔 1/3 @1.265", 1.265, 1.265, "2026-10-31", "down", "2026-10-14"),
    ("第3笔 1/3 @1.312", 1.312, 1.315, "2026-10-31", "up", "2026-10-28"),
]
STOP_ADDING = 1.261
HARD_STOP = 1.18


def fill_price_for(bar: pd.Series, limit: float, direction: str) -> float | None:
    """Fill price for a passive buy at ``limit``, or None if it never traded.

    A limit buy at ``limit`` fills at the open if the open is at or below the
    limit, otherwise at the limit itself if the session traded that low.
    A stop buy at ``limit`` fills at the open if the open is at or above the
    trigger, otherwise at the trigger itself if the session traded that high.
    """
    open_ = float(bar["Open"])
    if direction == "up":
        if float(bar["High"]) < limit:
            return None
        return max(open_, limit) if open_ >= limit else limit
    if float(bar["Low"]) > limit:
        return None
    return open_ if open_ <= limit else limit


def main() -> int:
    bars = fetch_eastmoney_daily()
    window = bars.loc[PLAN_START:]
    try:
        live = fetch_quote().get("last")
    except Exception:  # noqa: BLE001
        live = None

    print(f"计划起点 {PLAN_START}  回放区间 {window.index[0]} -> {window.index[-1]}")
    print(f"实时价 {live}\n")

    results = []
    for label, trigger, limit, end, direction, earliest in TRANCHES:
        sub = window.loc[earliest:end]
        fill_date = fill_px = None
        for date, bar in sub.iterrows():
            px = fill_price_for(bar, limit, direction)
            if px is not None:
                fill_date, fill_px = date, px
                break
        results.append(
            {
                "tranche": label,
                "trigger_price": trigger,
                "limit": limit,
                "direction": direction,
                "earliest_executable": earliest,
                "filled": fill_date is not None,
                "fill_date": fill_date,
                "fill_price": fill_px,
                "note": "条件未触发" if fill_date is None else "已成交",
            }
        )

    # Did the 1.261 guard fire? Closing basis and intraday basis differ.
    closes_below = window.index[window["Close"] < STOP_ADDING].tolist()
    lows_below = window.index[window["Low"] < STOP_ADDING].tolist()

    # Blended cost of the plan as executed, equal thirds by capital.
    filled = [r for r in results if r["filled"]]
    budget_per_tranche = 10000.0
    shares = cost = 0.0
    for r in filled:
        qty = budget_per_tranche // r["fill_price"]
        shares += qty
        cost += qty * r["fill_price"]
    blended = cost / shares if shares else 0.0
    ref = live or float(window["Close"].iloc[-1])

    # Counterfactual A: tranche 3 as a blind stop-order with no earnings gate.
    # On 2026-10-16 the ETF spiked to 1.315 intraday before closing at 1.264,
    # so an ungated order would have filled there.
    ungated = None
    for date, bar in window.iterrows():
        px = fill_price_for(bar, 1.315, "up")
        if px is not None:
            ungated = {"fill_date": date, "fill_price": px}
            break

    # Counterfactual: tranches 1+2 only (i.e. if tranche 3 had not been fired).
    filled_12 = [r for r in filled if r["tranche"].startswith(("第1", "第2"))]
    sh12 = cost12 = 0.0
    for r in filled_12:
        qty = budget_per_tranche // r["fill_price"]
        sh12 += qty
        cost12 += qty * r["fill_price"]
    blended12 = cost12 / sh12 if sh12 else 0.0

    if ungated:
        ug_shares = ug_cost = 0.0
        for r in filled_12:
            qty = budget_per_tranche // r["fill_price"]
            ug_shares += qty
            ug_cost += qty * r["fill_price"]
        qty3 = budget_per_tranche // ungated["fill_price"]
        ug_shares += qty3
        ug_cost += qty3 * ungated["fill_price"]
        ug_blended = ug_cost / ug_shares
    else:
        ug_blended = blended12

    out = {
        "replay_start": PLAN_START,
        "last_bar": window.index[-1],
        "live_price": ref,
        "tranches": results,
        "guards": {
            "stop_adding_level": STOP_ADDING,
            "closes_below_stop_adding": closes_below,
            "intraday_lows_below_stop_adding": lows_below,
            "close_basis_triggered": bool(closes_below),
            "intraday_basis_triggered": bool(lows_below),
            "hard_stop": HARD_STOP,
            "close_below_hard_stop": bool((window["Close"] < HARD_STOP).any()),
        },
        "executed_plan": {
            "tranches_filled": len(filled),
            "total_shares": int(shares),
            "capital_deployed_cny": round(cost, 2),
            "blended_avg_cost": round(blended, 4),
            "unrealised_pct_at_ref": round((ref / blended - 1) * 100, 2),
            "unrealised_cny_at_ref": round((ref - blended) * shares, 2),
        },
        "counterfactual_tranche3_ungated": ungated,
        "counterfactual_tranche3_ungated_blended": round(ug_blended, 4),
        "counterfactual_tranche3_ungated_pct": round((ref / ug_blended - 1) * 100, 2),
        "counterfactual_without_tranche3": {
            "blended_avg_cost": round(blended12, 4),
            "unrealised_pct_at_ref": round((ref / blended12 - 1) * 100, 2),
            "unrealised_cny_at_ref": round((ref - blended12) * sh12, 2),
        },
    }

    print("=" * 72)
    print("三笔计划执行复核")
    print("=" * 72)
    for r in results:
        if r["filled"]:
            print(f"  {r['tranche']:24s} → {r['fill_date']} 成交 @ {r['fill_price']:.4f}")
        else:
            print(f"  {r['tranche']:24s} → 未成交（{r['note']}）")

    g = out["guards"]
    print("\n" + "=" * 72)
    print("风控线触发情况")
    print("=" * 72)
    print(f"  停止加仓线 {STOP_ADDING}：收盘价击穿 {closes_below or '无'}；"
          f"盘中最低击穿 {lows_below or '无'}")
    print(f"  收盘口径触发：{g['close_basis_triggered']}    盘中口径触发：{g['intraday_basis_triggered']}")
    print(f"  硬止损 {HARD_STOP}：收盘击穿 = {g['close_below_hard_stop']}")

    e = out["executed_plan"]
    c = out["counterfactual_without_tranche3"]
    print("\n" + "=" * 72)
    print(f"执行结果（参考价 {ref}）")
    print("=" * 72)
    print(f"  已成交 {e['tranches_filled']}/3 笔，均价 {e['blended_avg_cost']:.4f}"
          f"  浮盈亏 {e['unrealised_pct_at_ref']:+.2f}%（{e['unrealised_cny_at_ref']:+.0f} 元）")
    print(f"  若第3笔没打：均价 {c['blended_avg_cost']:.4f}"
          f"  浮盈亏 {c['unrealised_pct_at_ref']:+.2f}%（{c['unrealised_cny_at_ref']:+.0f} 元）")
    if ungated:
        ug = out["counterfactual_tranche3_ungated"]
        print(f"  若第3笔无条件挂单：{ug['fill_date']} 成交 @ {ug['fill_price']:.4f}"
              f"  → 均价 {out['counterfactual_tranche3_ungated_blended']:.4f}"
              f"  浮盈亏 {out['counterfactual_tranche3_ungated_pct']:+.2f}%")

    dest = os.path.join("runs", f"159929_{window.index[-1]}.plan_review.json")
    os.makedirs("runs", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
