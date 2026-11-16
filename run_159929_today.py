"""Run the full TradingAgents multi-agent pipeline on 159929.SZ (医药ETF汇添富).

Same headless shape as ``run_single_ticker.py``, with two additions:

1. ``em_data.patch_yfinance_download()`` redirects 159929.SZ price history to
   Eastmoney. The yfinance vendor drops the 2026-10-13 bar entirely and reports
   an impossible ``open > high`` for 2026-10-14; the whole read of "is now a good
   place to add" turns on those last two sessions.
2. A wider debate (2 research rounds, 2 risk rounds) plus a structured dump of
   the final state to JSON, so the report can be assembled from the agents'
   own numbers rather than re-derived by hand.

Usage:
    python run_159929_today.py [YYYY-MM-DD]
"""

import json
import os
import socket
import sys

# The local proxy leaks keep-alive connections: a hung LLM call leaves sockets in
# CLOSE-WAIT forever and the run never advances. Bound every socket so a dropped
# connection surfaces as a timeout the retry budget can absorb, rather than an
# indefinite block. 600s is comfortably above a slow long-context completion.
socket.setdefaulttimeout(600)

# This machine has no direct route out, so every vendor and LLM call must go
# through the local proxy. socks:// entries break the OpenAI-compatible client,
# so they are dropped rather than inherited from the shell.
PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")
for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_var] = PROXY
for _socks_var in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_socks_var, None)

import tradingagents  # noqa: E402,F401  (loads .env at import)

from em_data import fetch_eastmoney_daily, fetch_quote, patch_yfinance_download  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402

ANALYSTS = ("market", "social", "news", "fundamentals")


def main() -> int:
    trade_date = sys.argv[1] if len(sys.argv) > 1 else "2026-10-14"

    # Provenance for the report: record the tape the agents will actually see.
    print("=" * 70)
    print("DATA PROVENANCE (Eastmoney, authoritative A-share feed)")
    print("=" * 70)
    try:
        print("live quote :", json.dumps(fetch_quote(), ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        print("live quote : unavailable", exc)
    try:
        bars = fetch_eastmoney_daily()
        print(f"daily bars : {len(bars)} rows, {bars.index[0]} -> {bars.index[-1]}")
        print(bars.tail(6).to_string())
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        print(f"daily bars : unavailable ({exc})")
    print()

    patch_yfinance_download()

    config = DEFAULT_CONFIG.copy()
    config["checkpoint_enabled"] = True
    # 1/1 is the proven-stable setting on this host. 2/2 pushes the risk debate
    # into very long contexts, and the proxy starts dropping those calls.
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    if config.get("llm_max_retries") is None:
        config["llm_max_retries"] = 8

    print(f"ticker=159929.SZ date={trade_date} analysts={ANALYSTS}")
    print(
        f"provider={config['llm_provider']} "
        f"deep={config['deep_think_llm']} quick={config['quick_think_llm']} "
        f"language={config['output_language']} "
        f"debate_rounds={config['max_debate_rounds']} "
        f"risk_rounds={config['max_risk_discuss_rounds']}"
    )

    ta = TradingAgentsGraph(selected_analysts=ANALYSTS, debug=True, config=config)
    final_state, decision = ta.propagate("159929.SZ", trade_date)

    print("\n" + "=" * 60)
    print(f"FINAL DECISION for 159929.SZ @ {trade_date}: {decision}")
    print("=" * 60)

    path = ta.save_reports(final_state, "159929.SZ")
    print(f"reports written to: {path}")

    dump = os.path.join("runs", f"159929_{trade_date}.state.json")
    os.makedirs("runs", exist_ok=True)
    serialisable = {
        k: v
        for k, v in (final_state or {}).items()
        if isinstance(v, (str, int, float, bool, type(None), dict, list))
    }
    with open(dump, "w", encoding="utf-8") as fh:
        json.dump(
            {"trade_date": trade_date, "decision": decision, "state": serialisable},
            fh,
            ensure_ascii=False,
            indent=2,
            # Graph state holds objects (e.g. CheckpointSAVEDict) that JSON cannot
            # encode; render them rather than failing the whole run at the end.
            default=str,
        )
    print(f"state dumped to: {dump}")

    for section in (
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "investment_debate_state",
        "trader_investment_plan",
        "risk_debate_state",
        "final_trade_decision",
    ):
        value = (final_state or {}).get(section)
        if isinstance(value, str) and value.strip():
            print(f"\n----- {section} -----\n{value}")
        elif isinstance(value, dict):
            print(f"\n----- {section} -----\n{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())