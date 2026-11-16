"""一句话结论接口：给一个股票，输出「买不买 / 买多少 / 胜率」。

    python ask.py 512800.SS              # 用今天
    python ask.py 512800.SS 2026-10-10   # 指定分析日
    python ask.py 512800.SS --no-run     # 只按已有决策日志复算，不重跑模型

胜率不是模型自己给的数字，而是决策日志里**已结算**样本的命中率：每次运行
都会写入一条决策，持有窗口（默认 5 个交易日）走完后回填实际收益，再按评级
分组统计。所以新标的首次运行时必然是「样本不足」，随着运行累积才有意义。

跑之前会修一个数据坑：Yahoo 对 A 股 ETF 偶尔返回「有成交量、无 OHLC」的末根
K 线，框架会静默回退到上一根收盘价，当成最新价用（512800.SS 上就发生过，
真实收盘 0.850 被报成 0.859）。这里用境内行情源把那一根补回来。
"""

import os
import re
import sys
from pathlib import Path

import run_single_ticker  # noqa: F401  导入即完成代理设置（剔除 socks 代理）

import tradingagents  # noqa: E402,F401
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.decision_log import TradingMemoryLog  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402

# 评级 -> 动作。框架的五档评级里，Buy/Overweight 都是"买"的意思，只是强度不同。
ACTION = {
    "buy": "买",
    "overweight": "买",
    "hold": "观望",
    "underweight": "不买",
    "sell": "卖",
}
# 单笔风险预算区间：仓位 = 风险预算 /（入场价 - 止损价）
RISK_BUDGET = (0.0025, 0.005)


# --------------------------------------------------------------------------
# 价格修复
# --------------------------------------------------------------------------
def _sina_symbol(ticker: str) -> str | None:
    """512800.SS -> sh512800；非 A 股返回 None。"""
    m = re.fullmatch(r"(\d{6})\.(SS|SZ)", ticker.strip())
    if not m:
        return None
    return ("sh" if m.group(2).upper() == "SS" else "sz") + m.group(1)


def fetch_cn_quote(ticker: str, timeout: float = 15.0) -> dict | None:
    """新浪行情快照。返回 open/high/low/close/volume，失败返回 None。"""
    import requests

    sym = _sina_symbol(ticker)
    if sym is None:
        return None
    try:
        r = requests.get(
            f"https://hq.sinajs.cn/list={sym}",
            headers={"Referer": "https://finance.sina.com.cn"},
            timeout=timeout,
        )
        text = r.content.decode("gbk", errors="ignore")
    except Exception:
        return None
    if '"' not in text:
        return None
    body = text.split('"')[1].split(",")
    if len(body) < 10:
        return None
    try:
        return {
            "open": float(body[1]),
            "close": float(body[3]),
            "high": float(body[4]),
            "low": float(body[5]),
            "volume": float(body[8]),
            "date": body[30] if len(body) > 30 else "",
        }
    except (ValueError, IndexError):
        return None


def repair_trailing_bar(ticker: str, trade_date: str) -> str | None:
    """缓存末根 K 线无收盘价时，用境内行情源补上。返回一行说明，无改动返回 None。"""
    import pandas as pd

    from tradingagents.dataflows.config import get_config
    from tradingagents.dataflows.symbols import safe_ticker_component

    cache = Path(get_config()["data_cache_dir"]) / f"{safe_ticker_component(ticker)}-YFin-data.csv"
    if not cache.exists():
        return None

    df = pd.read_csv(cache, on_bad_lines="skip", encoding="utf-8")
    if df.empty or pd.notna(df["Close"].iloc[-1]):
        return None

    quote = fetch_cn_quote(ticker)
    if quote is None or not quote["close"]:
        return None

    last = df.iloc[-1]
    row_date = str(last["Date"])[:10]
    if quote["date"] and quote["date"] != row_date:
        return None  # 行情源与缓存末根不是同一天，不做替换

    for col in ("Open", "High", "Low", "Close"):
        df.loc[df.index[-1], col] = quote[col.lower()]
    df.loc[df.index[-1], "Volume"] = quote["volume"]
    df.to_csv(cache, index=False, encoding="utf-8")
    return (
        f"价格修复：{row_date} 原为「有量无 OHLC」，已用新浪行情补为 "
        f"O{quote['open']} / H{quote['high']} / L{quote['low']} / C{quote['close']}"
    )


# --------------------------------------------------------------------------
# 结论解析
# --------------------------------------------------------------------------
_NUM = r"(\d+(?:\.\d+)?)"


def parse_levels(text: str) -> dict:
    """从决策正文里抠出入场价、止损价、目标价。抠不到就留空。"""

    def grab(*labels: str, window: int = 400) -> float | None:
        # 每个标签单独匹配：把多个标签塞进一个正则的 alternation 会让捕获组错位
        for label in labels:
            m = re.search(rf"\*\*{label}\*\*\s*[:：]?\s*(.{{0,{window}}})", text, re.S)
            if not m:
                continue
            # 字段正文里往往先出现标的代码（"512800.SS 主入场价为 0.843 元"），
            # 去掉代码再取价。只认 6 位代码：ETF 价格是 3~4 位小数，
            # 用 \d{4,6} 会把 "0.8261" 的小数部分一起吃掉。
            body = re.sub(r"\b\d{6}(?:\.[A-Z]{1,2})?\b", " ", m.group(1))
            price = re.search(_NUM, body)
            if price:
                return float(price.group(1))
        return None

    return {
        "entry": grab("Entry Price", "Entry"),
        "stop": grab("Stop Loss", "Stop"),
        "target": grab("Price Target"),
    }


def position_pct(entry: float | None, stop: float | None) -> tuple[float, float] | None:
    """按单笔风险预算倒算名义仓位区间。"""
    if not entry or not stop or entry <= stop:
        return None
    risk = entry - stop
    return tuple(b / risk * 100 for b in RISK_BUDGET)


def win_rate(rating: str, log_path: str) -> tuple[int, float | None]:
    """该评级在决策日志里已结算样本的命中率。"""
    entries = TradingMemoryLog({"memory_log_path": log_path}).load_entries()
    settled = [
        e for e in entries
        if not e["pending"] and e["rating"].lower() == rating.lower()
        and _parse_alpha(e.get("alpha")) is not None
    ]
    if not settled:
        return 0, None
    direction = 1 if rating.lower() in ("buy", "overweight") else -1
    hits = sum(1 for e in settled if _parse_alpha(e["alpha"]) * direction > 0)
    return len(settled), hits / len(settled)


def _parse_alpha(raw) -> float | None:
    if raw is None:
        return None
    try:
        return float(str(raw).strip().rstrip("%"))
    except ValueError:
        return None


# --------------------------------------------------------------------------
def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    ticker = args[0] if args else "512800.SS"

    from tradingagents.dataflows.date_window import get_current_date

    trade_date = args[1] if len(args) > 1 else get_current_date()
    log_path = DEFAULT_CONFIG["memory_log_path"]

    # 两种模式都要先修价格：复算时也要用真实收盘价，否则结论基于的是回退值
    note = repair_trailing_bar(ticker, trade_date)
    if note:
        print(note)

    if "--no-run" not in flags:
        config = DEFAULT_CONFIG.copy()
        config["checkpoint_enabled"] = True
        if config.get("llm_max_retries") is None:
            config["llm_max_retries"] = 6
        ta = TradingAgentsGraph(debug=False, config=config)
        _, rating = ta.propagate(ticker, trade_date)
        rating = str(rating)
    else:
        # 复算模式：读决策日志里该标的最近一条已结算决策
        rating = "Hold"
        for e in reversed(TradingMemoryLog({"memory_log_path": log_path}).load_entries()):
            if e["ticker"] == ticker:
                rating = e["rating"]
                break

    report = (
        Path(DEFAULT_CONFIG["results_dir"]) / ticker
        / "TradingAgentsStrategy_logs" / f"full_states_log_{trade_date}.json"
    )
    levels = {"entry": None, "stop": None, "target": None}
    if report.exists():
        import json
        blob = json.loads(report.read_text(encoding="utf-8"))
        # 入场/止损写在交易员计划里，评级与目标价写在组合经理决策里，两处都要读
        for source in (blob.get("final_trade_decision", ""),
                       blob.get("trader_investment_decision", "")):
            parsed = parse_levels(source)
            for key, value in parsed.items():
                if levels[key] is None:
                    levels[key] = value
    size = position_pct(levels["entry"], levels["stop"])
    n, hit = win_rate(rating, log_path)
    action = ACTION.get(rating.lower(), rating)

    print(f"\n{'=' * 46}\n{ticker}  {trade_date}\n{'=' * 46}")
    print(f"结论    : {action}（框架评级 {rating}）")
    if levels["entry"]:
        print(f"入场    : {levels['entry']} 元" + (f"   止损 {levels['stop']} 元" if levels["stop"] else ""))
    if levels["target"]:
        print(f"目标    : {levels['target']} 元")
    if size:
        print(f"仓位    : 名义 {size[0]:.1f}%–{size[1]:.1f}%"
              f"（单笔风险 {RISK_BUDGET[0]:.2%}–{RISK_BUDGET[1]:.2%}）")
    print(f"胜率    : " + (f"{hit:.0%}（{n} 个已结算样本）" if hit is not None
                        else f"暂无（{rating} 评级尚无已结算样本）"))
    print(f"{'=' * 46}")
    print(f"决策日志: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
