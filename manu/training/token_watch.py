#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**token / 上下文监控** —— 盯住「还能撑多久、离上下文上限还有多远」。

为什么要有它（2026-11-06）
--------------------------
宿主 **18:30 关机**，这一轮对话的 token 预算有限 ⇒ 需要一个
**不依赖我还在不在**、随时能跑的监控程序。

盯的三个量
----------
1. **上下文占用**（最要命）：当前消息的 prompt_tokens 相对上下文窗口
   （`llm-call.json` 的 `maxTokens`，本机 = 128000）⇒ 越接近越快触发压缩/截断。
2. **累计消耗 + 速率**：最近若干轮的 token/min ⇒ 外推还能撑多久。
3. **服务器流水线状态**：关机前最后一眼（4 个进程是否都活着、收割链走到哪）。

数据来源
--------
MCode 会话目录 `~/.minimax/v2/sessions/**/`：
* `messages.jsonl` —— 每条消息的 usage（prompt/completion/total tokens）
* `llm-call.json` —— `maxTokens`（上下文窗口）
* 取**最新活跃**的会话目录（按 messages.jsonl 的 mtime）。

用法
----
    python manu/training/token_watch.py            # 一次快照
    python manu/training/token_watch.py --watch 60 # 每 60 s 刷新
    python manu/training/token_watch.py --no-server
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

MINIMAX_DIR = Path(os.environ.get("MINIMAX_DIR", str(Path.home() / ".minimax")))
SESSION_ROOT = MINIMAX_DIR / "v2" / "sessions"


# ==============================================================================
# 找活跃会话 + 读 usage
# ==============================================================================

def active_session() -> Path | None:
    """mtime 最新的 messages.jsonl 所在目录 = 活跃会话。"""
    cands = glob.glob(str(SESSION_ROOT / "**" / "messages.jsonl"), recursive=True)
    if not cands:
        return None
    newest = max(cands, key=os.path.getmtime)
    return Path(newest).parent


def _iter_usage(obj):
    """在任意嵌套结构里找 usage 字典。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("usage", "tokenUsage", "token_usage") and isinstance(v, dict):
                yield v
            else:
                yield from _iter_usage(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_usage(v)


def read_usage(sess: Path) -> dict:
    """⇒ {ctx, last, used_total, calls, avg_last10}。

    ★ **实测 schema**（2026-11-06）：MCode 的 `messages.jsonl` 里 usage 字典是
    `{"input":…, "output":…, "cacheRead":…, "cacheWrite":…, "totalTokens":…}`。
    ⇒ **上下文占用 = input + output + cacheRead + cacheWrite**
    （`cacheRead` 是缓存命中的前缀，**也要算进上下文**，否则会严重低估）。
    ⚠️ `llm-call.json` 的 `maxTokens`(=128000) **不是上下文窗口**（实测上下文已 300k+），
    ⇒ 报告里把它单列，别当成上限。
    ⚠️ `messages.jsonl` **没有时间戳** ⇒ 算不了「token/min」时间速率
    ⇒ 改报「每次调用均值」+ 调用次数。
    """
    out = {"ctx": None, "last": None, "used_total": 0, "calls": 0,
           "avg_last10": None, "max_tokens": None, "breakdown": None}

    lc = sess / "llm-call.json"
    if lc.is_file():
        try:
            j = json.loads(lc.read_text(encoding="utf-8", errors="ignore"))

            def find_max(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k == "maxTokens" and isinstance(v, int):
                            return v
                        r = find_max(v)
                        if r:
                            return r
                elif isinstance(o, list):
                    for v in o:
                        r = find_max(v)
                        if r:
                            return r
                return None

            out["max_tokens"] = find_max(j)
        except Exception:
            pass

    mj = sess / "messages.jsonl"
    if not mj.is_file():
        return out
    totals: list[int] = []
    last = None
    try:
        with mj.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                for u in _iter_usage(obj):
                    t = u.get("totalTokens", u.get("total_tokens"))
                    if not isinstance(t, int):
                        continue
                    out["used_total"] += t
                    out["calls"] += 1
                    totals.append(t)
                    last = u
    except Exception:
        pass
    if last:
        g = lambda k: (last.get(k) if isinstance(last.get(k), int) else 0)
        bd = {"input": g("input"), "output": g("output"),
              "cacheRead": g("cacheRead"), "cacheWrite": g("cacheWrite")}
        out["breakdown"] = bd
        out["ctx"] = sum(bd.values())
        out["last"] = last.get("totalTokens")
    if len(totals) >= 10:
        out["avg_last10"] = sum(totals[-10:]) / 10
    return out


def fmt_int(v):
    return f"{v:,}" if isinstance(v, int) else "-"


# ==============================================================================
# 服务器流水线
# ==============================================================================

def server_status() -> str:
    ssh = ["ssh", "-p", "32222", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes",
           "huangzhe@192.168.99.40"]
    cmd = ("echo -n '存活进程数='; ps -ef | grep -E 'optuna_mix_ratio_search.py --output|"
           "npu20ms_watchdog.sh|run_seed_round_relay.sh|final_harvest.sh' | grep -v grep | wc -l; "
           "echo -n '完成标记='; cat /mnt/data/siping/runs/npu20ms/harvest_done.json 2>/dev/null || echo '未完成'; "
           "echo -n '收割进度='; tail -1 /mnt/data/siping/runs/npu20ms/final_harvest.log 2>/dev/null | tail -c 110")
    try:
        r = subprocess.run(ssh + [cmd], capture_output=True, text=True, timeout=25)
        return (r.stdout or "").strip() or "(服务器无响应)"
    except Exception as exc:  # noqa: BLE001
        return f"(查不到：{exc})"


# ==============================================================================

def snapshot() -> dict:
    sess = active_session()
    if sess is None:
        return {"sess": None, "u": {}}
    return {"sess": str(sess), "u": read_usage(sess)}


def main() -> int:
    ap = argparse.ArgumentParser(description="token / 上下文 / 流水线监控")
    ap.add_argument("--watch", type=int, default=0, help="轮询间隔秒（0=只打一次）")
    ap.add_argument("--no-server", action="store_true")
    args = ap.parse_args()

    prev = None
    while True:
        s = snapshot()
        u = s["u"]
        print(f"\n===== {time.strftime('%F %T')} =====")
        print(f"会话       : {s['sess']}")
        bd = u.get("breakdown") or {}
        print(f"上下文占用 : {fmt_int(u.get('ctx'))} token"
              f"（input {fmt_int(bd.get('input'))} + cacheRead {fmt_int(bd.get('cacheRead'))}"
              f" + output {fmt_int(bd.get('output'))}）")
        print(f"累计消耗   : {fmt_int(u.get('used_total'))} token / {u.get('calls')} 次调用")
        if u.get("avg_last10"):
            print(f"近 10 次   : 平均 {fmt_int(int(u['avg_last10']))} token/次")
        if prev and u.get("used_total") and prev.get("used_total"):
            dt = time.time() - prev["t"]
            dd = u["used_total"] - prev["used_total"]
            if dt > 0 and dd > 0:
                print(f"本轮速率   : {dd/dt*60:.0f} token/min")
        prev = {"t": time.time(), "used_total": u.get("used_total")}
        print(f"参考       : llm-call.json 的 maxTokens={fmt_int(u.get('max_tokens'))}"
              f"（**不是上下文上限**，实测上下文已 300k+）")
        if not args.no_server:
            print("--- 服务器流水线 ---")
            print(server_status())
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
