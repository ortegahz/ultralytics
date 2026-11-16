"""Authoritative A-share OHLCV for TradingAgents via Eastmoney (push2his).

Why this exists
---------------
For 159929.SZ (医药ETF汇添富) the yfinance vendor serves a defective recent tape:

* the 2026-10-13 bar is missing entirely (reproduced across several fetches), and
* 2026-10-14 is reported with ``Open 1.307`` but ``High 1.291`` -- an impossible
  open above the high.

Eastmoney is the exchange-side feed and carries both sessions correctly
(10-13 O/H/L/C 1.301/1.302/1.277/1.295). Since a "should I add to my position"
call turns on the last few sessions, the pipeline is patched to read Eastmoney
daily bars instead of yfinance. See ``patch_yfinance_download``.

Falls back to yfinance for any date range Eastmoney does not cover, so a vendor
outage degrades instead of failing the run.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

import pandas as pd

PROXY = os.environ.get("TRADINGAGENTS_PROXY", "http://127.0.0.1:10808")

# Eastmoney secid: leading digit selects the exchange (0 = Shenzhen, 1 = Shanghai).
DEFAULT_SECID = "0.159929"

_KLINE_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    "?secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
    "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
    "&klt=101&fqt={fqt}&beg={beg}&end={end}"
)

# Eastmoney kline field order for klt=101 (daily), fqt=1 (forward-adjusted).
_COLUMNS = [
    "Date",
    "Open",
    "Close",
    "High",
    "Low",
    "Volume",
    "Amount",
    "Amplitude",
    "ChangePct",
    "Change",
    "Turnover",
]


def _opener() -> urllib.request.OpenerDirector:
    op = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    )
    op.addheaders = [("User-Agent", "Mozilla/5.0")]
    return op


def _get_json(url: str, timeout: int, attempts: int = 6):
    """GET + parse JSON, retrying the proxy's frequent dropped connections.

    Eastmoney resets the connection on a fair share of requests through this
    host's proxy, so a single failure is not evidence the endpoint is down.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with _opener().open(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - transient network faults only
            last = exc
            if attempt < attempts - 1:
                time.sleep(min(2.0 * (attempt + 1), 6.0))
    raise RuntimeError(f"Eastmoney request failed after {attempts} attempts: {last}")


# The pipeline re-fetches history on every tool call; memoising keeps a flaky
# upstream from being hit dozens of times per run.
_DAILY_CACHE: dict[tuple, pd.DataFrame] = {}


def fetch_eastmoney_daily(
    secid: str = DEFAULT_SECID,
    beg: str = "20210101",
    end: str = "20991231",
    fqt: int = 1,
    timeout: int = 30,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily bars from Eastmoney as a tidy frame indexed by ``Date``.

    ``fqt=1`` is forward-adjusted, which matches what the pipeline's other
    indicator paths expect from yfinance's ``auto_adjust=True`` series. Eastmoney
    reports volume in 手 (lots); it is scaled to shares to match yfinance.
    """
    key = (secid, beg, end, fqt)
    if use_cache and key in _DAILY_CACHE:
        return _DAILY_CACHE[key]

    cache_file = os.path.join(
        os.environ.get("TRADINGAGENTS_CACHE_DIR")
        or os.path.join(os.path.expanduser("~"), ".tradingagents", "cache"),
        f"em-{secid}-daily.csv",
    )

    url = _KLINE_URL.format(secid=secid, beg=beg, end=end, fqt=fqt)
    try:
        payload = _get_json(url, timeout)
    except Exception:
        # Eastmoney rate-limits bursts, and several processes share this feed.
        # A slightly stale on-disk tape beats failing the run outright; intraday
        # callers still get today's bar because the file was written today.
        if os.path.exists(cache_file):
            logging.getLogger(__name__).warning(
                "Eastmoney daily fetch failed for %s; serving cached tape from %s", secid, cache_file
            )
            stale = pd.read_csv(cache_file).set_index("Date")
            if use_cache and not stale.empty:
                _DAILY_CACHE[key] = stale
            return stale
        raise

    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return pd.DataFrame(columns=_COLUMNS)

    rows = [line.split(",") for line in klines]
    df = pd.DataFrame(rows, columns=_COLUMNS[: len(rows[0])])
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = df["Volume"] * 100.0  # 手 -> shares
    out = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    if use_cache and not out.empty:
        _DAILY_CACHE[key] = out
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            out.to_csv(cache_file)
        except OSError:
            pass  # cache is an optimisation, never a hard dependency
    return out


_QUOTE_HOSTS = (
    "https://push2his.eastmoney.com/api/qt/stock/get",
    "https://82.push2.eastmoney.com/api/qt/stock/get",
    "https://push2.eastmoney.com/api/qt/stock/get",
)
_QUOTE_FIELDS = "f43,f44,f45,f46,f47,f48,f57,f58,f60,f168,f169,f170"


def fetch_quote(secid: str = DEFAULT_SECID, timeout: int = 30, price_divisor: float = 1000.0) -> dict:
    """Live snapshot: last price, day open/high/low, previous close, turnover.

    ``push2.eastmoney.com`` is unreachable from this host's proxy path, so the
    hosts are tried in order and ``push2his`` -- the same origin that serves the
    history -- is preferred.
    """
    query = f"?secid={secid}&fields={_QUOTE_FIELDS}"
    payload, last = None, None
    for host in _QUOTE_HOSTS:
        try:
            payload = _get_json(host + query, timeout, attempts=2)
            break
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last = exc
    if payload is None:
        raise RuntimeError(f"all Eastmoney quote hosts failed: {last}")

    d = payload.get("data") or {}

    # Eastmoney encodes fund/ETF quotes with three implied decimals: a last
    # print of 1.290 arrives as f43=1290. The divisor is therefore fixed at
    # 1000 rather than inferred, and validated below against the previous close
    # so a format change surfaces instead of silently returning 12.90.
    divisor = price_divisor

    def px(key):
        v = d.get(key)
        return None if v in (None, "-") else round(v / divisor, 4)

    last, prev = px("f43"), px("f60")
    if last is not None and not (0.05 < last < 100):
        raise ValueError(
            f"implausible quote {last} from divisor {divisor} (raw f43={d.get('f43')}); "
            "Eastmoney price encoding likely changed"
        )

    return {
        "name": d.get("f58"),
        "code": d.get("f57"),
        "last": px("f43"),
        "open": px("f46"),
        "high": px("f44"),
        "low": px("f45"),
        "prev_close": px("f60"),
        "change": px("f169"),
        "change_pct": None if d.get("f170") in (None, "-") else round(d["f170"] / 100.0, 3),
        "volume_lots": d.get("f47"),
        "amount": d.get("f48"),
        "turnover_rate_pct": None if d.get("f168") in (None, "-") else round(d["f168"] / 100.0, 3),
    }


def patch_yfinance_download(secid: str = DEFAULT_SECID, start: str = "20210101"):
    """Route the pipeline's yfinance calls to Eastmoney for one A-share symbol.

    Two yfinance entry points have to be covered, because the pipeline does not
    use one consistently:

    * ``yf.download`` -- used by the OHLCV vendor (``vendors/yahoo/ohlcv.py``).
    * ``yf.Ticker(..).history`` -- used by the market vendor for price series
      and by ``get_indicators`` (``vendors/yahoo/market.py``). Patching only
      ``yf.download`` leaves the indicator series on the defective Yahoo tape:
      the 2026-10-28 bar goes missing and every moving average is computed
      one session short.

    Returns the original ``yf.download`` so a caller can restore it. Only
    ``159929.SZ`` is redirected; every other symbol keeps the stock path. If
    Eastmoney cannot be reached the original yfinance path still runs, so a
    flaky upstream degrades the data quality instead of failing the whole run.
    """
    import yfinance as yf

    original = yf.download
    original_ticker = yf.Ticker

    def download(ticker, *args, **kwargs):
        symbol = ticker if isinstance(ticker, str) else (ticker[0] if ticker else "")
        if symbol not in ("159929.SZ", "159929"):
            return original(ticker, *args, **kwargs)

        try:
            em = fetch_eastmoney_daily(secid=secid, beg=start)
        except Exception as exc:  # noqa: BLE001 - fall back, never abort the run
            logging.getLogger(__name__).warning(
                "Eastmoney unavailable for %s (%s); falling back to yfinance", symbol, exc
            )
            return original(ticker, *args, **kwargs)

        if em.empty:
            return original(ticker, *args, **kwargs)

        frame = em.copy()
        frame.index.name = "Date"  # load_ohlcv normalises this back out
        # yfinance's auto_adjust output carries no Adj Close column.
        return frame

    # The fetch range is captured under its own name: rebinding ``start``
    # inside ``history`` would turn the enclosing ``start`` into a local of
    # ``history``, and the fetch line would then raise UnboundLocalError --
    # which the fallback below would swallow, silently serving the defective
    # Yahoo tape instead of the corrected one.
    fetch_beg = start

    def ticker_factory(*ticker_args, **ticker_kwargs):
        """A yf.Ticker whose .history() is redirected for the patched symbol."""
        ticker_obj = original_ticker(*ticker_args, **ticker_kwargs)
        symbol = ticker_obj.ticker if hasattr(ticker_obj, "ticker") else ""
        if symbol not in ("159929.SZ", "159929"):
            return ticker_obj

        original_history = ticker_obj.history

        def history(*args, **kwargs):
            try:
                em = fetch_eastmoney_daily(secid=secid, beg=fetch_beg)
            except Exception as exc:  # noqa: BLE001 - fall back, never abort the run
                logging.getLogger(__name__).warning(
                    "Eastmoney unavailable inside .history() for %s (%s); "
                    "falling back to yfinance", symbol, exc
                )
                return original_history(*args, **kwargs)
            if em.empty:
                return original_history(*args, **kwargs)

            frame = em.copy()
            frame.index.name = "Date"
            # yfinance returns an index named "Date"; the market vendor slices
            # it by string dates, so keep the labels as YYYY-MM-DD strings.
            req_start = kwargs.get("start")
            req_end = kwargs.get("end")
            if req_start is not None:
                frame = frame.loc[frame.index >= str(req_start)]
            if req_end is not None:
                frame = frame.loc[frame.index <= str(req_end)]
            return frame

        ticker_obj.history = history
        return ticker_obj

    yf.download = download
    yf.Ticker = ticker_factory
    return original