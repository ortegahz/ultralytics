"""Run the TradingAgents multi-agent pipeline on one ticker (headless).

Provider, models, output language and retry budget come from .env
(TRADINGAGENTS_* / provider key vars), so this script only adds what a
non-interactive run needs: proxy sanitising, a fixed date, checkpointing and
a printed report tree.

Usage:
    python run_single_ticker.py 512800.SS 2026-10-10 [analyst1,analyst2]
"""

import os
import sys

# This machine has no direct route out (a plain https://www.google.com fetch
# times out), so every vendor and LLM call must go through the local proxy.
# Set it explicitly rather than relying on the inherited shell values, and drop
# the socks:// entries: httpx, which the OpenAI-compatible client uses, raises
# "Unknown scheme for proxy URL" on those while http/https_proxy work fine.
PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")
for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[_var] = PROXY
for _socks_var in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_socks_var, None)

import tradingagents  # noqa: E402  (loads .env at import)

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402


def main() -> int:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "512800.SS"
    trade_date = sys.argv[2] if len(sys.argv) > 2 else "2026-10-10"
    analysts = (
        tuple(a.strip() for a in sys.argv[3].split(",") if a.strip())
        if len(sys.argv) > 3
        else ("market", "social", "news", "fundamentals")
    )

    config = DEFAULT_CONFIG.copy()
    config["checkpoint_enabled"] = True
    if config.get("llm_max_retries") is None:
        config["llm_max_retries"] = 6

    print(f"ticker={ticker} date={trade_date} analysts={analysts}")
    print(
        f"provider={config['llm_provider']} "
        f"deep={config['deep_think_llm']} quick={config['quick_think_llm']} "
        f"language={config['output_language']} "
        f"debate_rounds={config['max_debate_rounds']} "
        f"risk_rounds={config['max_risk_discuss_rounds']}"
    )

    ta = TradingAgentsGraph(selected_analysts=analysts, debug=True, config=config)
    final_state, decision = ta.propagate(ticker, trade_date)

    print("\n" + "=" * 60)
    print(f"FINAL DECISION for {ticker} @ {trade_date}: {decision}")
    print("=" * 60)

    path = ta.save_reports(final_state, ticker)
    print(f"reports written to: {path}")

    for section in ("market_report", "sentiment_report", "news_report",
                    "fundamentals_report", "investment_debate_state",
                    "trader_investment_plan", "risk_debate_state"):
        value = (final_state or {}).get(section)
        if isinstance(value, str) and value.strip():
            print(f"\n----- {section} -----\n{value}")
        elif isinstance(value, dict):
            print(f"\n----- {section} -----\n{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
